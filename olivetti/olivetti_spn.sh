#!/bin/bash
num_executions=10
echo "START"
echo "--- Olivetti test with a total of $num_executions executions"
for (( counter=0; counter<num_executions; counter++ ))
do
    echo "----- Execution $counter"
    echo "Execution $counter" >> olivetti_execution.txt
    bo=$(python3 olivetti_bo.py $counter)
    echo "$bo" >> olivetti_execution.txt
    rs=$(python3 olivetti_rs.py $counter)
    echo "$rs" >> olivetti_execution.txt
done
echo "END"