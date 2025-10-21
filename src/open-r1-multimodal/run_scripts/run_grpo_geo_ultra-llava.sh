# export MODEL_NAME=llava-hf/llava-v1.6-vicuna-7b-hf
export MODEL_NAME=llava-hf/llava-1.5-7b-hf
export EARTHREASON_ROOT=<your_earthreason_root_path_here>
export SAM_SIZE=small
export SAM_ROOT=../../sam2
export RUN_NAME=GEO-LLaVA-7B-EarthReason-GRPO-$SAM_SIZE-$(date +%Y%m%d_%H%M%S)
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_API_KEY=<your_wandb_api_key_here>
# set --report_to to "none" to disable wandb logging

# Debug mode, log the details of the reward, set to "false" to disable
export DEBUG_MODE="true"
export LOG_PATH="output_ultra/$RUN_NAME/debug_log.txt"

export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export TORCH_NUM_THREADS=32
export NUMEXPR_NUM_THREADS=32

torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo_geo_ultra.py \
    --deepspeed local_scripts/zero3.json \
    --dataset_name none \
    --output_dir output_ultra/$RUN_NAME \
    --model_name_or_path $MODEL_NAME \
    --max_prompt_length 1024 \
    --max_completion_length 1024 \
    --num_generations 12 \
    --per_device_train_batch_size 12 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 8 \
    --run_name $RUN_NAME \
    --sam_model_size $SAM_SIZE \
    --sam_root $SAM_ROOT \
    --sam_device cuda:1 \
    --earthreason_root $EARTHREASON_ROOT \
    --save_steps 100 \
    --save_only_model true \
    --save_total_limit 3 \
    --earthreason_resize_size 672 \
    --use_datasets earthreason \
    --freeze_vision_modules false  \
    --beta 0.001

# If you want to only finetune the language model, set freeze_vision_modules to true. 

