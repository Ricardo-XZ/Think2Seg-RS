export model_root=../src/open-r1-multimodal/output_ultra    # <your_output_directory_path>
export RUN_NAME=GEO-Qwen-3B-EarthReason-GRPO-small           # <your_model_run_name>
export EARTHREASON_ROOT=<your_earthreason_root_path_here>   # <your_earthreason_root_path>
export CUDA_VISIBLE_DEVICES=0,1,2,3
export N_PER_NODE=4
export MASTER_PORT=12347
export SAM_SIZE=small
export SAM_ROOT=../../sam2  # <your_sam_root_path_here>

# mkdir geo_ultra_results/test/$RUN_NAME
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
      test_think2seg-rs_llava.py.py \
      --seed 0 \
      --model_path $target_dir \
      --output_dir ./geo_ultra_results/test/$RUN_NAME \
      --num_samples 1928 \
      --batch_size 20 \
      --sam_device cuda:1 \
      --sam_model_size $SAM_SIZE \
      --sam_root $SAM_ROOT \
      --dataset earthreason \
      --earthreason_root $EARTHREASON_ROOT \
      --resize_size 840 \
      --split test \
      --visualize_num 50 \
      --save_results
done


# for val_split visualizations
checkpoints=(1100)
for i in "${checkpoints[@]}"
do
    if [ "$i" = "final" ]; then
        export target_dir=$model_root/$RUN_NAME
    else
        export target_dir=$model_root/$RUN_NAME/checkpoint-$i
    fi
    # export target_dir=$model_root/$RUN_NAME/checkpoint-$i
    echo "Running test for $target_dir"
    torchrun --nproc_per_node=$N_PER_NODE \
        --master_port $MASTER_PORT \
      test_think2seg-rs_llava.py.py \
      --seed 0 \
      --model_path $target_dir \
      --output_dir ./geo_ultra_results/test/$RUN_NAME \
      --num_samples 1135 \
      --batch_size 20 \
      --sam_device cuda:1 \
      --sam_model_size $SAM_SIZE \
      --sam_root $SAM_ROOT \
      --dataset earthreason \
      --earthreason_root $EARTHREASON_ROOT \
      --resize_size 840 \
      --split val \
      --visualize_num 50 \
      --save_results
done