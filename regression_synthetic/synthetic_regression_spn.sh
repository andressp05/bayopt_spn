#!/bin/bash
num_executions=3
echo "START"
echo "--- Synthetic Regression test with a total of $num_executions executions"
python3 synthetic_regression_dataset.py
echo "--- Synthetic Regression dataset charged"
for (( counter=0; counter<num_executions; counter++ ))
do
    echo "----- Execution $counter"
    echo "Execution $counter" >> regression_execution.txt
    bo=$(python3 synthetic_regression_bo.py $counter)
    echo "$bo" >> regression_execution.txt
    rs=$(python3 synthetic_regression_rs.py $counter)
    echo "$rs" >> regression_execution.txt
done
echo "END"