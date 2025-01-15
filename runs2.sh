#!/usr/bin/env bash

set -o xtrace

EPOCHS=80
TRIALS=1

python -m fo.scripts.train -a o3 -e $EPOCHS -t $TRIALS ;
echo "run1";
# python -m fo.scripts.train -a resnet18 -e $EPOCHS -t $TRIALS ;
# echo "run2";
python -m fo.scripts.train -a resnet18 -e $EPOCHS -t $TRIALS -p;
echo "run3";

python -m fo.scripts.test -a o3 > results_o3.txt ;
echo "run4";
python -m fo.scripts.test -a resnet18 > results_resnet.txt ;
echo "run5";
