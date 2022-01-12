#!/bin/bash
# args: eval script, test file, output file

for i in *.tsv
do
    python3 tsv_to_gs.py $i ${i%.*}.wa
done