for lr in 0.002 0.004 0.001
do
    for eps in 0.3 0.2 0.6
    do
        CUDA_VISIBLE_DEVICES=0 python -m train --dataset tiny_in --epoch 1000 --lr $lr --epsilon $eps --emb 128 --method maxent
    done
done