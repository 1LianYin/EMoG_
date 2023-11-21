import torch
import torch.nn.functional as F
from torch import nn
from models.wav2vec import *

import math
from torch.autograd import Variable

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class AdaLN(nn.Module):

    def __init__(self, latent_dim, time_embed_dim, dropout):
        super().__init__()
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )

    def forward(self, h, emb):
        """
        h: B, T, D
        emb: B, D
        """
        # B, 1, 2D
        emb_out = self.emb_layers(emb).unsqueeze(1)
        # scale: B, 1, D / shift: B, 1, D
        scale, shift = torch.chunk(emb_out, 2, dim=2)
        h = self.norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return h

class FFN(nn.Module):

    def __init__(self, latent_dim, ffn_dim, dropout, time_embed_dim):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, ffn_dim)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj_out = AdaLN(latent_dim, time_embed_dim, dropout)

    def forward(self, x, emb):
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        y = x + self.proj_out(y, emb)
        return y

class TemporalSelfAttention(nn.Module):

    def __init__(self, latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = AdaLN(latent_dim, time_embed_dim, dropout)
    
    def forward(self, x, emb, src_mask):
        """
        x: B, T, D
        """
        B, T, D = x.shape
        H = self.num_head
        # B, T, 1, D
        query = self.query(self.norm(x)).unsqueeze(2)
        # B, 1, T, D
        key = self.key(self.norm(x)).unsqueeze(1)
        query = query.view(B, T, H, -1)
        key = key.view(B, T, H, -1)
        # B, T, T, H
        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H)
        attention = attention + (1 - src_mask.unsqueeze(-1)) * -100000

        weight = self.dropout(F.softmax(attention, dim=2))
        value = self.value(self.norm(x)).view(B, T, H, -1)
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, T, D)
        y = x + self.proj_out(y, emb)
        return y

class TemporalCrossAttention(nn.Module):

    def __init__(self, latent_dim, audio_latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.audio_norm = nn.LayerNorm(audio_latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(audio_latent_dim, latent_dim)
        self.value = nn.Linear(audio_latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = AdaLN(latent_dim, time_embed_dim, dropout)
    
    def forward(self, x, xf, emb, src_mask):
        """
        x: B, T, D
        xf: B, N, L
        """
        B, T, D = x.shape
        N = xf.shape[1]
        H = self.num_head
        # B, T, 1, D
        query = self.query(self.norm(x)).unsqueeze(2)
        # B, 1, N, D
        key = self.key(self.audio_norm(xf)).unsqueeze(1)
        query = query.view(B, T, H, -1)
        key = key.view(B, N, H, -1)
        # B, T, N, H
        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H)
        attention = attention + (1 - src_mask.unsqueeze(-1)) * -100000
        
        weight = self.dropout(F.softmax(attention, dim=2))
        value = self.value(self.audio_norm(xf)).view(B, N, H, -1)

        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, T, D)
        y = x + self.proj_out(y, emb)
        return y

class TemporalCrossAttention_motion(nn.Module):

    def __init__(self, latent_dim, audio_latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.audio_norm = nn.LayerNorm(audio_latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(audio_latent_dim, latent_dim)
        # self.value = nn.Linear(audio_latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = AdaLN(latent_dim, time_embed_dim, dropout)
    
    def forward(self, x, xf, emb, src_mask):
        """
        x: B, T, D
        xf: B, N, L
        """
        B, T, D = x.shape
        N = xf.shape[1]
        H = self.num_head
        # B, T, 1, D
        query = self.query(self.norm(x)).unsqueeze(2)
        # B, 1, N, D
        key = self.key(self.audio_norm(xf)).unsqueeze(1)
        query = query.view(B, T, H, -1)
        key = key.view(B, N, H, -1)
        # B, T, N, H
        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H)
        attention = attention + (1 - src_mask.unsqueeze(-1)) * -100000
        
        weight = self.dropout(F.softmax(attention, dim=2))
        value = self.value(self.norm(x)).view(B, T, H, -1)

        y = torch.einsum('bnmh,bnhd->bnhd', weight, value).reshape(B, T, D)
        y = x + self.proj_out(y, emb)
        return y

class TemporalDiffusionTransformerDecoderLayer(nn.Module):

    def __init__(self,
                 latent_dim=32,
                 audio_latent_dim=512,
                 time_embed_dim=128,
                 ffn_dim=256,
                 num_head=4,
                 dropout=0.1):
        super().__init__()
        self.sa_block = TemporalSelfAttention(
             latent_dim, num_head, dropout, time_embed_dim)
        self.ca_block = TemporalCrossAttention(
             latent_dim, audio_latent_dim, num_head, dropout, time_embed_dim)
        self.ffn = FFN(latent_dim, ffn_dim, dropout, time_embed_dim)

    def forward(self, x, xf, emb, src_mask):
        x = self.sa_block(x, emb, src_mask)
        x = self.ca_block(x, xf, emb, src_mask)
        x = self.ffn(x, emb)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class GestureTransformer(nn.Module):
    def __init__(self,
                 input_feats,
                 num_frames=150,
                 latent_dim=512,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0,
                 activation="gelu", 
                 num_audio_layers=4,
                 audio_latent_dim=256,
                 audio_ff_size=2048,
                 audio_num_heads=4,
                 spatial_dim=64,
                 emotion_f=8,
                 speaker_dims=30,
                 speaker_f=8,
                 **kargs):
        super().__init__()
        
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation  
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim * 4
        self.sequence_embedding = nn.Parameter(torch.randn(150, latent_dim)) # learned absolute position encode
        self.position_embedding =  PositionalEncoding(spatial_dim, dropout)
        self.emotion_f = emotion_f
        self.speaker_dims = speaker_dims
        self.speaker_f = speaker_f
        self.spatial_dim = spatial_dim

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.spatial_dim))
        
        self.emotion_embedding_tail = nn.Sequential( 
            nn.Conv1d(self.emotion_f, 8, 9, 1, 4),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(8, 16, 9, 1, 4),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(16, 16, 9, 1, 4),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(16, self.emotion_f, 9, 1, 4),
            nn.BatchNorm1d(self.emotion_f),
            nn.LeakyReLU(0.3, inplace=True),
        )
        self.audio_feature_map = nn.Linear(768, audio_latent_dim)
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_latent_dim, self.time_embed_dim)
        )

        spatialTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=spatial_dim,
            nhead=audio_num_heads,
            dim_feedforward=audio_ff_size,
            dropout=dropout,
            activation=activation)
        self.spatialTransEncoder = nn.TransformerEncoder(
            spatialTransEncoderLayer,
            num_layers=num_audio_layers)

        self.emotion_embedding = nn.Embedding(self.emotion_f, self.emotion_f)
        self.emo_proj = nn.Sequential(
            nn.Linear(self.emotion_f, audio_latent_dim),
            nn.SiLU(),
            nn.Linear(audio_latent_dim, self.time_embed_dim),
        )

        # Input Embedding
        self.joint_embed = nn.Linear(self.input_feats, self.latent_dim)
        self.patch_embed = nn.Linear(3, self.spatial_dim)
        self.temp_embed = nn.Linear(num_frames, 1)

        self.spatial_proj = nn.Linear(self.spatial_dim, latent_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(self.spatial_dim, self.spatial_dim),
            nn.SiLU(),
            nn.Linear(self.spatial_dim, self.spatial_dim),
        )

        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
        self.temporal_decoder_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.temporal_decoder_blocks.append(
                TemporalDiffusionTransformerDecoderLayer(
                    seq_len=num_frames,
                    latent_dim=latent_dim,
                    audio_latent_dim=audio_latent_dim,
                    time_embed_dim=self.time_embed_dim,
                    ffn_dim=ff_size,
                    num_head=num_heads,
                    dropout=dropout
                )
            )
        
        # Output Module
        self.out = zero_module(nn.Linear(self.latent_dim, self.input_feats))

        # audio wav2vec_encoder
        self.wave2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.wave2vec.feature_extractor._freeze_parameters()

        self.speaker_embedding = nn.Sequential(
                nn.Embedding(self.speaker_dims, self.speaker_f),
                nn.Linear(self.speaker_f, audio_latent_dim), 
                nn.LeakyReLU(True),
                nn.Linear(audio_latent_dim, self.time_embed_dim), 
            )
    def encode_emo(self, in_emo):
        emo_feat_seq = self.emotion_embedding(in_emo.long()) # B, T, 8
        emo_feat_seq = emo_feat_seq.permute([0,2,1])
        emo_feat_seq = self.emotion_embedding_tail(emo_feat_seq)
        emo_feat_seq = emo_feat_seq.permute([0,2,1])
        emo_feat_seq = self.emo_proj(emo_feat_seq.mean(1)) # 
        return emo_feat_seq
        
    def encode_audio(self, audio, length, device):
        audio_feat = self.wave2vec(audio, frame_num=length).last_hidden_state # [128, 34, 768]
        audio_out = self.audio_feature_map(audio_feat)
        audio_proj = self.audio_proj(torch.mean(audio_out, dim=1))
        return audio_proj, audio_out

    def generate_src_mask(self, T, length):
        B = len(length)
        src_mask = torch.ones(B, T)
        for i in range(B):
            for j in range(length[i], T):
                src_mask[i, j] = 0
        return src_mask

    def forward(self, x, timesteps, in_emo=None, in_id=None, emo_proj=None, is_train=False, length=None, pre_seq=None, audio=None, audio_proj=None, audio_out=None, ref_seq=None):
        """
        x: B, T, D
        timesteps: B
        in_emo: B, T
        in_id: B, 1
        audio: B, 160000
        """
        B, T = x.shape[0], x.shape[1]
        # get mask
        src_mask = self.generate_src_mask(T, length).to(x.device).unsqueeze(-1)
        # calculate connections between joints
        cls_tokens = self.cls_token.expand(B, -1, -1) + self.time_mlp(timestep_embedding(timesteps, self.spatial_dim)).unsqueeze(1)
        try:    
            spatial_x = self.patch_embed(self.temp_embed(x.transpose(1, 2)).reshape(B, -1, 3))
        except:
            padding = torch.zeros(x.shape[0], 150 - x.shape[1], x.shape[2]).cuda()
            x_padding = torch.cat((x, padding), dim=1)
            spatial_x = self.patch_embed(self.temp_embed(x_padding.transpose(1, 2)).reshape(B, -1, 3))

        spatial_x = self.position_embedding(torch.cat((cls_tokens, spatial_x), dim=1))
        spatial_x = self.spatialTransEncoder(spatial_x)[:, :1]
        cls_tokens = self.spatial_proj(cls_tokens)
        if audio is not None and len(audio) != B:
            index = x.device.index
            audio = audio[index * B: index * B + B]
        if audio_proj is None or audio_out is None:
            audio_proj, audio_out = self.encode_audio(audio, T, x.device) # [34, 128, 2048]
        if emo_proj is None:
            emo_proj = self.encode_emo(in_emo) # [1, 128, 2048]
        id_proj = self.speaker_embedding(in_id.long()).squeeze(1)
        
        emb = self.time_embed(timestep_embedding(timesteps, self.latent_dim)) + id_proj + emo_proj #+ id_proj#+ audio_proj
        # B, T, latent_dim
        h = self.joint_embed(x)
        pos = self.sequence_embedding.unsqueeze(0)[:, :T, :]
        h = h + pos

        for idx, module in enumerate(self.temporal_decoder_blocks):
            if idx == self.num_layers // 2:
                h = h + cls_tokens
            h = module(h, audio_out, emb, src_mask)

        output = self.out(h).view(B, T, -1).contiguous()
        return output


if __name__ =='__main__':
    x = torch.randn([128, 150, 141])
    timesteps = torch.randn([128])
    in_emo = torch.full((128, 150), 0)
    in_id = torch.full((128, 1), 2)
    audio = torch.randn([128, 160000])
    length = torch.full((128, 1), 150)

    model = GestureTransformer(141)

    output = model(x, timesteps, in_emo, in_id, audio=audio, length=length)
    print(output.shape)
    for name, parameters in model.named_parameters():
        print(name,':',parameters.size())

