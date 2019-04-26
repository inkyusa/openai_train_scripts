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
	N_LAYERS=2
    N_HIDDEN=128
	MODEL_PATH=$1
	python -m baselines.run --alg=ppo2 --num_timesteps=0 --play --env=QuadRate-v0 --load_path=$MODEL_PATH --num_layers=$N_LAYERS --num_hidden=$N_HIDDEN
fi
