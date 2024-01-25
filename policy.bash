#!/bin/bash

policies=('random' 'greedy' 'DQN' )
#mkdir -p results/policy

for ((i=1; i<${#policies[@]}; i++))
do
    for ((j=i+1; j<${#policies[@]}; j++))
    do
        echo "Running duel with MCTS default policies: ${policies[i]} vs ${policies[j]}"
        python duel.py -w MCTS --wp ${policies[i]} -b MCTS --bp ${policies[j]} -r 25 |tee results/policy/MCTS_${policies[i]}_vs_${policies[j]}.txt &
        python duel.py -w MCTS --wp ${policies[j]} -b MCTS --bp ${policies[i]} -r 25 |tee results/policy/MCTS_${policies[i]}_vs_${policies[j]}.txt 
    done
done