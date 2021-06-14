#!/bin/bash
num_executions=10
echo "START"
echo "--- Iris test with a total of $num_executions executions"
python3 iris_dataset.py
echo "--- Iris dataset charged"
for (( counter=0; counter<num_executions; counter++ ))
do
    echo "----- Execution $counter"
    echo "Execution $counter" >> iris_execution.txt
    bo=$(python3 iris_bo.py $counter)
    echo "$bo" >> iris_execution.txt
    rs=$(python3 iris_rs.py $counter)
    echo "$rs" >> iris_execution.txt
done
echo "END"