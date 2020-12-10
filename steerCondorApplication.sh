#!/bin/bash

shiftList=("JECup" "JECdown" "JERup" "JERdown")

for sft in ${shiftList[@]}; do
  echo 'submitApplication ${sft}'
  python -u submitApplication.py ${sft}
done


