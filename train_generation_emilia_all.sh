devices=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
accelerate launch --mixed_precision=bf16 --num_processes=$devices --main_process_port=12345 \
    vibevoice/finetune/train_vibevoice.py \
    --model_name_or_path vibevoice/VibeVoice-1.5B \
    --train_jsonl data/manifest/emilia/train.jsonl \
    --text_column_name text \
    --audio_column_name audio \
    --voice_prompts_column_name audio \
    --output_dir exp/vibevoice_generation_emilia_all \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-5 \
    --num_train_epochs 5 \
    --logging_steps 10 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --report_to tensorboard \
    --remove_unused_columns False \
    --train_connectors True \
    --bf16 True \
    --do_train \
    --gradient_clipping \
    --gradient_checkpointing False \
    --ddpm_batch_mul 4 \
    --diffusion_loss_weight 1 \
    --train_diffusion_head True \
    --ce_loss_weight 1 \
    --voice_prompt_drop_rate 1.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0 \
    --max_grad_norm 0.8
    # --resume_from_checkpoint ./vibevoice_generation_emilia/checkpoint-10 \
