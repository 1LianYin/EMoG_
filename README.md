
<h1 style="text-align: center;">
EMoG: Synthesizing Emotive Co-speech 3D Gesture with Diffusion Model 
</h1>

# Dataset
In this project, we harness the power of the BEAT dataset to train our models. The dataset is meticulously partitioned and processed in accordance with the methodologies detailed at BEAT([https://example.com]).
    
# Reproduction
### Train EMoG
0. `python == 3.7`
1. build folders like:
    ```
    EMoG
    ├── codes
    │   └── EMoG
    ├── datasets
    │   ├── beat_raw_data
    │   ├── beat_annotations
    │   └── beat_cache
    └── outputs
        └── EMoG
    ```
2. download the scripts to `codes/EMoG/`
3. download full dataset to `datasets/beat`, process datas via method of BEAT(([https://example.com])).
7. run ```python -u /code/MotionDiffuse/text2motion/tools/train.py \
    --name  EMoG \
    --batch_size  128 \
    --num_epochs 150 \
    --gpu_id 0 \
    --dataset_name beat \
    --pose_cache bvh_rot_cache \
    --latent_dim 512 \
    --lr 1e-4 \
    --pose_length 150\
    --dim_pose 141 \
    --multi_length_training 0.2 0.4 0.6 0.8 1.0 \``` 


