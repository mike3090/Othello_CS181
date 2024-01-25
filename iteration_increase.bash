#!/bin/bash

iterations=(10 20 50 100 200)
mkdir -p results

for i in "${iterations[@]}"
do
    echo "Running duel with MCTS iterations: $i"
    python duel.py -w MCTS --wi $i -b DQN -r 50 > results/MCTS_white_${i}_iterations.txt & 
    python duel.py -b MCTS --bi $i -w DQN -r 50 > results/MCTS_black_${i}_iterations.txt & 
done

wait