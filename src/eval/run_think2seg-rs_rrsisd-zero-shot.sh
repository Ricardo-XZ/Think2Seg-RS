export model_root=../src/open-r1-multimodal/output_ultra    # <your_output_directory_path>
export RUN_NAME=GEO-Qwen-3B-EarthReason-GRPO-small           # <your_model_run_name>
export RRSISD_ROOT=<your_rrsisd_root_path_here>   # <your_rrsisd_root_path>
export CUDA_VISIBLE_DEVICES=0,1,2,3
export N_PER_NODE=4
export MASTER_PORT=12347
export SAM_SIZE=small
export SAM_ROOT=../../sam2  # <your_sam_root_path_here>

# define checkpoint list, final or specific checkpoints
# e.g., checkpoints=(final 2000 1000 500)

checkpoints=(final)
for i in "${checkpoints[@]}"
do
    if [ "$i" = "final" ]; then
        export target_dir=$model_root/$RUN_NAME
    else
        export target_dir=$model_root/$RUN_NAME/checkpoint-$i
    fi
    echo "Running test for $target_dir"
    torchrun --nproc_per_node=$N_PER_NODE \
        --master_port $MASTER_PORT \
      test_think2seg-rs_ref.py \
      --seed 0 \
      --model_path $target_dir \
      --output_dir ./geo_rrsisd_results/test/$RUN_NAME \
      --num_samples 3481 \
      --batch_size 20 \
      --sam_device cuda:1 \
      --sam_model_size $SAM_SIZE \
      --dataset rrsisd \
      --rrsisd_root $RRSISD_ROOT \
      --IoU_threshold 0.5 \
      --split test \
      --visualize_num 50 \
      --save_results
done
