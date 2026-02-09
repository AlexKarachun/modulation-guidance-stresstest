ACCELERATE_CONFIG="configs/default_config.yaml"
PORT=$(( ((RANDOM<<15)|RANDOM) % 27001 + 2000 ))
echo $PORT

for w in 0; do
  for start_layer in 0; do
      CUDA_VISIBLE_DEVICES=1,2 accelerate launch \
  --num_processes=2 \
  --multi_gpu \
  --mixed_precision bf16 \
  --main_process_port $PORT \
  main.py \
    --pretrained_model_name_or_path "nvidia/Cosmos-Predict2-2B-Text2Image" \
    --pretrained_model_name_or_path_clip "black-forest-labs/FLUX.1-schnell" \
    --gradient_checkpointing True \
    --apply_clip_pooled True \
    --cfg_scale 7.0 \
    --seed 10 \
    --do_init_run \
    --output_dir 'validation' \
    --height 768 \
    --width 768 \
    --w $w \
    --start_layer $start_layer \
    --shift_type 'realism'
  done
done
