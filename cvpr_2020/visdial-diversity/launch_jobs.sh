#!/bin/bash
INPUT=$1
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
while read LINE; do
     sbatch -p short --gres=gpu:1 -J visdial-diversity -o /srv/share2/vmurahari3/visdial-rl/logs_slurm/slurm-%j.out -x calculon run_job.sh  "$LINE"
done < $INPUT
