#!/bin/bash

# if [ "$#" -ne 4 ]
# then
#   echo "================================================================================================"
#     echo "Script for training a model using ppo2 (default) given some parameters.                          "
#     echo "Usage : source ./train_script.sh <model save path and name> <log dir path> <# steps> <# env>    "
#     echo "e.g., source ./train_script.sh ./model/quad-10M ./log/mylog 1e3 2          "
#     echo ""
#     echo "Note that save model and log are optional but recommended to validate the trained model after   "
#     echo "training and debugging purposes.                                                                "
#     echo "trained model after training and debugging purposes                                             "
#     echo "================================================================================================"
#     #exit 1
# elif [ "$#" -eq 4 ]
# then
    # MODEL_SAVE_PATH=$1
    # LOG_PATH=$2
    # N_STEP=$3
    # N_ENV=$4
    #TEST_NAME="test06"
    TEST_NAME="$(date +"%F-%H-%M-%S")"
    #echo $TEST_NAME
    MODEL_SAVE_PATH=$(printf "%s/%s" "./model" "$TEST_NAME")
    LOG_PATH=$(printf "%s/%s" "./log" "$TEST_NAME")
    echo $MODEL_SAVE_PATH
    echo $LOG_PATH
    rm -rf $LOG_PATH

    # N_STEP=256
    # N_TIMESTEPS=2e6
    # N_ENV=32
    # N_MINIBATCHES=1024
    # N_OPTEPOCHS=8

    # N_STEP=2048
    # N_TIMESTEPS=50e6 #50e6
    # N_ENV=32
    # N_MINIBATCHES=64
    # N_OPTEPOCHS=10

    N_STEP=8000
    N_TIMESTEPS=50e6 #50e6
    N_ENV=32
    N_MINIBATCHES=128
    N_OPTEPOCHS=10

    #activation = tf.tanh or tf.nn.relu

    #python ./train_hovering.py --save_path=$MODEL_SAVE_PATH --num_timesteps=$N_STEP --num_env=$N_ENV --play=False --env=MujocoQuadQuat-v0 --logdir=$LOG_PATH --reward_scale=1.0 --nsteps=4096 --nminibatches=16 --noptepochs=16
    #python ./train_hovering.py --save_path=$MODEL_SAVE_PATH --num_timesteps=$N_STEP --num_env=$N_ENV --play=False --env=MujocoQuadQuat-v0 --logdir=$LOG_PATH --reward_scale=1.0 --nsteps=2048 --nminibatches=64 --noptepochs=16
    OPENAI_LOG_FORMAT=stdout,tensorboard OPENAI_LOGDIR=$LOG_PATH python -m baselines.run --env=QuadRate-v0 --network=mlp --num_timesteps=$N_TIMESTEPS --num_env=$N_ENV --save_path=$MODEL_SAVE_PATH --seed=13 --alg=ppo2 --nsteps=$N_STEP --nminibatches=$N_MINIBATCHES --noptepochs=$N_OPTEPOCHS --reward_scale=1 --num_layers=2 --num_hidden=256 --activation=tf.tanh
# fi

