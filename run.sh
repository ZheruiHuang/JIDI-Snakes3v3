#!/bin/bash

evaluate_freq=2000
save_freq=100
eval_times=2
episode_max_steps=200
seed=42
hidden_dim=256
hidden_state_dim=128
gamma=0.9

main_output_dir="./models/main/"
league_output_dir="./models/league/"

export PYTHONPATH=$PYTHONPATH:$(pwd)


python ./src/main.py \
    --max_train_steps 4000000 \
    --evaluate_freq $evaluate_freq \
    --save_freq $save_freq \
    --episode_max_steps $episode_max_steps \
    --seed $seed \
    --eval_times $eval_times \
    --batch_size 120 \
    --mini_batch_size 120 \
    --hidden_dim $hidden_dim \
    --hidden_state_dim $hidden_state_dim \
    --lr 0.001 \
    --gamma $gamma \
    --K_epochs 10 \
    --save_dir $main_output_dir


python ./src/league.py \
    --max_train_steps 4000000 \
    --evaluate_freq $evaluate_freq \
    --save_freq $save_freq \
    --episode_max_steps $episode_max_steps \
    --seed $seed \
    --eval_times $eval_times \
    --batch_size 120 \
    --mini_batch_size 120 \
    --hidden_dim $hidden_dim \
    --hidden_state_dim $hidden_state_dim \
    --lr 0.0003 \
    --gamma $gamma \
    --K_epochs 10 \
    --load_model \
    --load_dir $main_output_dir \
    --save_dir $league_output_dir