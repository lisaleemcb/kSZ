#!/bin/bash

#SBATCH --job-name=run_kSZ
#SBATCH --time=02-00:00:00
#SBATCH --output=output.log
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=lisaleemcb@gmail.com
#SBATCH --ntasks-per-node=12

# use the bash shell
set -x
# echo each command to standard out before running it
date

# source bash profile
source /home/emc-brid/.bashrc
source ~/venvs/riddler/bin/activate

# run the Unix 'date' command
echo "Hello world, from the Cluster!"
# run the Unix 'echo' command
# which mamba
# mamba activate kSZ
which python
python -u /home/emc-brid/kSZ/scripts/make_kSZ.py --file /home/emc-brid/sims_check.npy 
