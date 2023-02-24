#!/usr/bin/bash

num_rounds=$1
time_stamp=$(date +"%m-%d-%Y_at_%H-%M-%S")

if [ -z "$num_rounds" ]
then
    num_rounds=10
    echo "Number of training rounds not provided; using default value '$num_rounds'"
fi

for scenario in "best" "mid" "worst"
    do

        echo -e "\nSTARTING '$scenario' SCENARIO\n"
        ./main.py --strat=$scenario --num_rounds=$num_rounds --time_stamp=$time_stamp
        echo -e "\nENDING '$scenario' SCENARIO\n"
done