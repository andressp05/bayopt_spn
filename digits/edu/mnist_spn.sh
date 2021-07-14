#!/bin/bash
num_executions=2
echo "START"
echo "--- Mnist test with a total of $num_executions executions"
for (( counter=0; counter<num_executions; counter++ ))
do
    echo "----- Execution $counter"
    echo "Execution $counter" >> mnist_execution.txt
    bo=$(python3 mnist_bo.py $counter)
    echo "$bo" >> mnist_execution.txt
    rs=$(python3 mnist_rs.py $counter)
    echo "$rs" >> mnist_execution.txt
done
echo "END"
