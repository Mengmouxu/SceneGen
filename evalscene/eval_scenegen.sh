#!/bin/bash

# Configuration
INPUT_DIR="TRELLIS-500K/3D-FUTURE/scene_test_SceneGen"
OUTPUT_DIR="TRELLIS-500K/3D-FUTURE/scene_test"
GPU_COUNT=4

echo "Starting evaluation in parallel on ${GPU_COUNT} GPUs..."

# Array to hold process IDs
declare -a pids

# Launch one process on each GPU
for gpu_id in $(seq 0 $(($GPU_COUNT - 1)))
do
    echo "Launching evaluation process on GPU ${gpu_id}..."
    
    # Run the evaluation script in the background for the current GPU
    # Pass gpu_num and gpu_id using the Hydra override syntax (e.g., system.gpu_num)
    CUDA_VISIBLE_DEVICES=${gpu_id} python evaluate.py \
        --config-name scene_evaluation_scenegen \
        system.input_dir=${INPUT_DIR} \
        system.output_dir=${OUTPUT_DIR} \
        system.gpu_num=${GPU_COUNT} \
        system.gpu_id=${gpu_id} &
    
    # Store the process ID
    pids[${gpu_id}]=$!
    echo "Process ID for GPU ${gpu_id}: ${pids[${gpu_id}]}"
done

# Wait for all background processes to complete
echo "Waiting for all evaluation processes to complete..."
for pid in ${pids[*]}; do
    wait $pid
    echo "Process $pid has completed."
done

echo "All evaluation tasks are complete!"