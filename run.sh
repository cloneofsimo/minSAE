python sae.py \
    --model-name meta-llama/Llama-2-7b-hf \
    --layer-idx 8 \
    --cache-dir activation_cache \
    --learning-rate 1e-4 \
    --batch-size 2048 \
    --wandb-project test_sae_project \
    --num-train-samples 1000 \
    --num-val-samples 100 \
    --wandb-run-name test_sae_run_relu \
    --sae-type topk \
    --l1-coef 0.01 \
    --num-epochs 10
    
