#!/usr/bin/env bash
#$ -binding linear:4 # request 4 cpus (8 with Hyperthreading) (some recommend 4 per GPU)
#$ -N ex       # set consistent base name for output and error file (allows for easy deletion alias)
#$ -q all.q    # don't fill the qlogin queue (can some add why and when to use?)
#$ -cwd        # change working directory (to current)
#$ -V          # provide environment variables
# -t 1-10    # start 100 instances: from 1 to 100
# if you also want to request a GPU, add the following line to the above block:
#$ -l cuda=1   # request one GPU
#$ -o task_out
#$ -j y
echo "I am a job task with ID $SGE_TASK_ID."
source /home/salnabulsi/.thesis-py38/bin/activate && python train_baseline.py > output.txt
