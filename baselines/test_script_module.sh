#!/bin/bash

if [ "$#" -ne 1 ]
then
	echo "====================================================================="
    echo "This script will test trained model                                  "
    echo "Usage : source test_script.sh <model path>"
    echo "e.g., source test_script.sh ~/workspace/model/quad-10M               "
    echo "====================================================================="
    #exit 1
elif [ "$#" -eq 1 ]
then
	MODEL_PATH=$1
	python -m baselines.run --num_timesteps=0 --play --env=QuadRate-v0
fi