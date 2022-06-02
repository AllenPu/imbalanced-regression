#! /bin/bash 

for i in $(seq 1000 +500 5000)
do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --data_dir ./data/ --reweight none --train_number $i 
done