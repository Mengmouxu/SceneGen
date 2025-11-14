#!/bin/bash

OUTPUT_DIR="TRELLIS-500K/3D-FUTURE" # Path to the 3D-FUTURE dataset

export TMPDIR=tmp

mkdir -p $TMPDIR

echo "Starting TRELLIS evaluation in parallel on 8 GPUs..."

# Launch one process on each GPU
for gpu_id in {0..7}
do
    # Set the corresponding GPU for each process
    echo "Launching evaluation process on GPU ${gpu_id}..."
    CUDA_VISIBLE_DEVICES=${gpu_id} python scenegen_eval.py 3D-FUTURE \
        --output_dir ${OUTPUT_DIR} \
        --set test \
        --gpu_num 8 \
        --gpu_id ${gpu_id} &
    
    # Record process ID
    pid=$!
    echo "Process ID for GPU ${gpu_id}: ${pid}"
    pids[${gpu_id}]=${pid}
done

# Wait for all processes to finish
echo "Waiting for all evaluation processes to complete..."
for pid in ${pids[*]}
do
    wait $pid
    echo "Process $pid has completed"
done

echo "All GPU evaluation tasks are complete!"

cd evalscene
echo "Starting SceneGen evaluation script..."
bash eval_scenegen.sh
