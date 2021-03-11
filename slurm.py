from sim_logistic import grid
import numpy as np
import itertools
import subprocess
import os, sys

with open('parms.txt', 'r') as f:
    pars = f.readline()

pars = eval(pars)
parameters = grid(n_signal = pars['n_signal'], n_pi = pars['n_pi'],\
     n_kappa = pars['n_kappa'], n_theta = pars['n_theta'], n_sim = pars['n_sim'])

n_sims = len(parameters)
n_jobs = int(n_sims / 5000) + 1
job_log_file = 'job-logs.txt'
last_array_size = n_sims - int(n_sims / 5000) * 5000
for job in range(n_jobs):

    job_file = f'job_files/job_{job}.sh'

    # content of batch file
    job_string = f'#!/bin/bash\n'
    job_string += f'#SBATCH --job-name=sim{job}\n'
    job_string += f'#SBATCH --output=logs/sim_%A_%a.out\n'
    if job < (n_jobs - 1):
        job_string += f'#SBATCH --array=0-4999\n'
    else: 
        job_string += f'#SBATCH --array=0-{last_array_size-1}\n'
    job_string += f'#SBATCH --nodes=1\n#SBATCH --cpus-per-task=1\n#SBATCH --mem-per-cpu=6gb\n'
    job_string += f'#SBATCH --time=1:00:00\n#SBATCH --account=yuekai1\n#SBATCH --mail-type=NONE\n'
    job_string += f'#SBATCH --mail-user=smaity@umich.edu\n#SBATCH --partition=standard\n'
    job_string += f'echo "SLURM_JOBID: " $SLURM_JOBID\necho "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID\n'
    job_string += f'echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID\n'
    job_string += f'python3 grid_sim.py $(($SLURM_ARRAY_TASK_ID+{job * 5000}))'


    with open(job_file, 'w') as jf:
        jf.write(job_string)

    os.system('sbatch ' + job_file)
    
