#!/bin/bash
#while true; do
    python train_torcs.py \
        --save-path mpc_10_torcs \
        --env torcs \
        --learning-freq 10 \
        --num-train-steps 1 \
        --simple-seg \
        --continuous \
        --one-hot \
        --use-seg \
        --lstm2 \
        --num-total-act 2 \
        --pred-step 10 \
        --buffer-size 50000 \
        --epsilon-frames 100000 \
        --batch-size 16 \
        --use-collision \
        --use-offroad \
        --use-speed \
        --sample-with-collision \
        --sample-with-offroad \
        --num-same-step 1 \
        --data-parallel \
        --id 25 \
        --resume
#done
