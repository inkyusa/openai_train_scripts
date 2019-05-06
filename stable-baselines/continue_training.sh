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
    MODEL_LOAD_PATH=$1
    # LOG_PATH=$2
    # N_STEP=$3
    # N_ENV=$4
    #TEST_NAME="$(date +"%F-%H-%M-%S")"
    #echo $TEST_NAME
    #MODEL_SAVE_PATH=$(printf "%s/%s" "./model" "$TEST_NAME")
    RL_BASELINES_ZOO_PATH=/home/enddl22/workspace/rl-baselines-zoo
    #LOG_PATH=$(printf "%s/%s" "./log" "$TEST_NAME")
    #echo $MODEL_SAVE_PATH
    #echo $LOG_PATH
    #rm -rf $LOG_PATH
    CURR_DIR=$PWD
    INPUT_MODEL_PATH2=${INPUT_MODEL_PATH1#"./"}
    ABS_MODEL_PATH=$(printf "%s/%s" "$CURR_DIR" "${MODEL_LOAD_PATH#"./"}")
    echo $ABS_MODEL_PATH

    lockFile="/home/enddl22/workspace/venv-p3.6/lib/python3.6/site-packages/mujoco_py/generated/mujocopy-buildlock.lock"
    if [ -f $lockFile ] ; then
    rm $lockFile
    fi
    cd $RL_BASELINES_ZOO_PATH
    python train.py --algo ppo2 --env QuadRate-v0 --tensorboard-log ~/workspace/openai_train_scripts/stable-baselines/log/ -i $ABS_MODEL_PATH -n 1000000
    cd $CURR_DIR
    #activation = tf.tanh or tf.nn.relu

    #python ./train_hovering.py --save_path=$MODEL_SAVE_PATH --num_timesteps=$N_STEP --num_env=$N_ENV --play=False --env=MujocoQuadQuat-v0 --logdir=$LOG_PATH --reward_scale=1.0 --nsteps=4096 --nminibatches=16 --noptepochs=16
    #python ./train_hovering.py --save_path=$MODEL_SAVE_PATH --num_timesteps=$N_STEP --num_env=$N_ENV --play=False --env=MujocoQuadQuat-v0 --logdir=$LOG_PATH --reward_scale=1.0 --nsteps=2048 --nminibatches=64 --noptepochs=16
    #OPENAI_LOG_FORMAT=$LOG_FORMAT OPENAI_LOGDIR=$LOG_PATH python -m baselines.run --env=QuadRate-v0 --network=mlp --num_timesteps=$N_TIMESTEPS --num_env=$N_ENV --save_path=$MODEL_SAVE_PATH --alg=ppo2 --nsteps=$N_STEP --nminibatches=$N_MINIBATCHES --noptepochs=$N_OPTEPOCHS --seed=$SEED --reward_scale=$RW_SCALE --num_layers=$N_LAYERS --num_hidden=$N_HIDDEN --activation=$ACTIVATION --ent_coef=$ENT_COEF --value_network=$VN
# fi

