#!/bin/bash
num_executions=3
echo "START"
echo "--- Synthetic Classification test with a total of $num_executions executions"
python3 synthetic_classification_dataset.py
echo "--- Synthetic Classification dataset charged"
for (( counter=0; counter<num_executions; counter++ ))
do
    echo "----- Execution $counter"
    echo "Execution $counter" >> classification_execution.txt
    bo=$(python3 synthetic_classification_bo.py $counter)
    echo "$bo" >> classification_execution.txt
    rs=$(python3 synthetic_classification_rs.py $counter)
    echo "$rs" >> classification_execution.txt
done
echo "END"