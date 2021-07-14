#!/bin/bash
num_executions=3
echo "START"
echo "--- Wine test with a total of $num_executions executions"
python3 wine_dataset.py
echo "--- Wine dataset charged"
for (( counter=0; counter<num_executions; counter++ ))
do
    echo "----- Execution $counter"
    echo "Execution $counter" >> wine_execution.txt
    bo=$(python3 wine_bo.py $counter)
    echo "$bo" >> wine_execution.txt
    rs=$(python3 wine_rs.py $counter)
    echo "$rs" >> wine_execution.txt
done
echo "END"