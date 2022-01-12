#!/bin/bash
# args: eval script, test file, output file

for i in score*.wa
do
    echo $i >> $3
    $1 $2 $i >> $3
done
