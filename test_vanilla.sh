#!/usr/bin/env bash
#$ -binding linear:4 # request 4 cpus (8 with Hyperthreading) (some recommend 4 per GPU)
#$ -N ex       # set consistent base name for output and error file (allows for easy deletion alias)
#$ -q all.q    # don't fill the qlogin queue (can some add why and when to use?)
#$ -cwd        # change working directory (to current)
#$ -V          # provide environment variables
# -t 1-10    # start 100 instances: from 1 to 100
# if you also want to request a GPU, add the following line to the above block:
#$ -l cuda=1   # request one GPU
#$ -j y
#$ -l h='!node4*'
echo "I am a job task with ID $SGE_TASK_ID."
export CUDA_LAUNCH_BLOCKING=1

dataset=$1
real_data=$2
pretrained=$3

source /home/salnabulsi/.thesis-py38/bin/activate && python test_vanilla.py --dataset ${dataset} --real-data ${real_data} --pretrained ${pretrained} >> output-test-${timestamp}.txt 