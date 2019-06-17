RL_BASELINES_ZOO_PATH=/home/enddl22/workspace/rl-baselines-zoo
TENSORBOARD_LOG=~/workspace/openai_train_scripts/stable-baselines/log/
LOG_DIR=~/workspace/openai_train_scripts/stable-baselines/model
CURR_DIR=$PWD
lockFile="/home/enddl22/workspace/mujoco-py/mujoco_py/generated/mujocopy-buildlock.lock"
if [ -f $lockFile ] ; then
rm $lockFile
fi
cd $RL_BASELINES_ZOO_PATH
OPENAI_LOG_FORMAT=tensorboard,stdout python train.py --algo ppo2 --env QuadRate-v0 --tensorboard-log $TENSORBOARD_LOG --log-folder $LOG_DIR
cd $CURR_DIR
