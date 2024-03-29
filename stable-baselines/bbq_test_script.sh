#!/bin/bash

if [ "$#" -ne 1 ]
then
	echo "====================================================================="
    echo "This script will test trained model                                  "
    echo "Usage : source test_script.sh <model path>"
    echo "e.g., source test_script.sh ./model                                  "
    echo "====================================================================="
    #exit 1
elif [ "$#" -eq 1 ]
then
	RL_BASELINES_ZOO_PATH=/home/enddl22/workspace/rl-baselines-zoo
	INPUT_MODEL_PATH1=$1
    INPUT_MODEL_PATH2=${INPUT_MODEL_PATH1#"./"}
    CURR_DIR=$PWD
	cd $RL_BASELINES_ZOO_PATH
	MODEL_PATH=$(printf "%s/%s" "$CURR_DIR" "$INPUT_MODEL_PATH2")
    $echo $MODEL_PATH
	python enjoy.py --algo ppo2 --folder $MODEL_PATH -n 50000 --env BallBouncingQuad-v0
    
    cd $CURR_DIR
fi
