#!/usr/bin/bash

num_rounds=$1

if [ -z "$num_rounds" ]
then
    num_rounds=10
    echo "Number of training rounds not provided; using default value '$num_rounds'"
fi

for scenario in "best" "mid" "worst"
    do
        echo "\nSTARTING '$scenario' SCENARIO\n"
        ./main.py --strat=$scenario --num_rounds=$num_rounds
        echo "\nENDING '$scenario' SCENARIO\n"
done