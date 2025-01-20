#!/bin/bash
#$ -S /bin/bash
#$ -N DANI010
#$ -cwd

#==============================
#== Evaluation of Individual ==
#==============================
python3 /Exacutables/Eval_Individual.py $1 $2 $3 $4

