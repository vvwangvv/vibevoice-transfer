. path.sh
devices=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
accelerate launch --mixed_precision=bf16 --num_processes=$devices --main_process_port=12345 \
    vibevoice/finetune/train_vibevoice.py \
    --dataloader_num_workers 0 \
    --model_name_or_path vibevoice/VibeVoice-1.5B \
    --train_jsonl data/manifest/emilia_spk5000_en/train_u4500_g4500.jsonl \
    --text_column_name text \
    --audio_column_name audio \
    --voice_prompts_column_name audio \
    --output_dir exp/debug \
    --per_device_train_max_samples 16 \
    --per_device_train_max_tokens 8192 \
    --learning_rate 3e-5 \
    --num_train_epochs 50 \
    --voice_input_use_semantic False \
    --logging_steps 100 \
    --save_steps 10000 \
    --eval_steps 10000 \
    --report_to none \
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
    --lr_scheduler_type constant \
    --warmup_ratio 0 \
    --max_grad_norm 0.8 \
    --generation_use_semantic_only True \
    --multiple_choice_version 2  # use label in speaker text \
    # --resume_from_checkpoint ./vibevoice_generation_emilia/checkpoint-10 \
