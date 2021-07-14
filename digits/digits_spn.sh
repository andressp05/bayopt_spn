#!/bin/bash
num_executions=3
echo "START"
echo "--- Digits test with a total of $num_executions executions"
python3 digits_dataset.py
echo "--- Digits dataset charged"
for (( counter=0; counter<num_executions; counter++ ))
do
    echo "----- Execution $counter"
    echo "Execution $counter" >> digits_execution.txt
    bo=$(python3 digits_bo.py $counter)
    echo "$bo" >> digits_execution.txt
    rs=$(python3 digits_rs.py $counter)
    echo "$rs" >> digits_execution.txt
done
echo "END"