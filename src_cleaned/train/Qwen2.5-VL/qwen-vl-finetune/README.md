##### 


./train_raw.sh \
    --dataset "aicitydataset_train" \
    --model "Qwen/Qwen2.5-VL-7B-Instruct" \
    --project "AICity_QwenVL7B_LoRA_Run1" \
    --gpu 0 \
    --lora \
    --rank 64 \
    --lr 1e-4 \
    --epochs 3 \
    --batch-size 2 \
    --grad-accum 8 \
    

