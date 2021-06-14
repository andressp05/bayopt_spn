#!/bin/bash
num_executions=10
echo "START"
echo "--- Boston test with a total of $num_executions executions"
for (( counter=0; counter<num_executions; counter++ ))
do
    echo "----- Execution $counter"
    echo "Execution $counter" >> boston_execution.txt
    bo=$(python3 boston_bo.py $counter)
    echo "$bo" >> boston_execution.txt
    rs=$(python3 boston_rs.py $counter)
    echo "$rs" >> boston_execution.txt
done
echo "END"