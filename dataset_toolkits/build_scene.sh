#!/bin/bash

OUTPUT_DIR="TRELLIS-500K/3D-FUTURE" # Path to the 3D-FUTURE dataset
BLENDER_PATH="/tmp/blender-3.0.1-linux-x64" # Path to your Blender installation
NUM_GPUS=8
NUM_WORKERS_PER_GPU=2

echo "Starting scene building in parallel on ${NUM_GPUS} GPUs, with ${NUM_WORKERS_PER_GPU} workers per GPU..."

pids=()

for gpu_id in $(seq 0 $((${NUM_GPUS}-1)))
do
    for worker_id in $(seq 0 $((${NUM_WORKERS_PER_GPU}-1)))
    do
        echo "Launching build process on GPU ${gpu_id}, Worker ${worker_id}..."
        CUDA_VISIBLE_DEVICES=${gpu_id} python dataset_toolkits/build_scene.py 3D-FUTURE \
            --output_dir ${OUTPUT_DIR} \
            --set test \
            --blender_path ${BLENDER_PATH} \
            --gpu_num ${NUM_GPUS} \
            --gpu_id ${gpu_id} \
            --num_workers ${NUM_WORKERS_PER_GPU} \
            --worker_id ${worker_id} &
        
        pids+=($!)
    done
done

echo "Waiting for all build processes to complete..."
for pid in "${pids[@]}"; do
    wait "$pid"
done

echo "All scene building tasks are complete!"