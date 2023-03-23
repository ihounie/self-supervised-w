for eps in 0.3 0.2 0.4
do
    CUDA_VISIBLE_DEVICES=1 python -m train --dataset stl10 --epoch 2000 --lr 0.002 --epsilon $eps --bs 256 --emb 128 --w_size 256 --method maxent
done