export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python train.py \
--config configs/generation/ss_scenegen_flow_img_train.json \
--output_dir output/ss_scenegen_flow_img_train \
--data_dir TRELLIS-500K/3D-FUTURE \
--num_gpus 8 \
--load_dir output/ss_scenegen_flow_img_train \
# --tryrun

# data_dir: Path to the 3D-FUTURE dataset