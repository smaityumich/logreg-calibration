from sim_logistic import logreg_calib, grid
import numpy as np
import itertools
import subprocess
import os, sys


n = 10000
ps = np.linspace(0.1, 0.9, 9)

with open('parms.txt', 'r') as f:
    pars = f.readline()

pars = eval(pars)
parameters = grid(n_signal = pars['n_signal'], n_pi = pars['n_pi'],\
     n_kappa = pars['n_kappa'], n_theta = pars['n_theta'], n_sim = pars['n_sim'])

i = int(float(sys.argv[1]))
n_sims = len(parameters)

if i <= n_sims:
    s, pi, k, theta, penalty, ITER = parameters[i]
    [score0, score1], [norm_b, theta0, theta1], [ce0, ce1]\
         = logreg_calib(n = n, s = s, pi = pi, k = k, theta = theta, penalty = penalty)
    return_dict = dict()
    return_dict['score0'] = score0
    return_dict['score1'] = score1
    return_dict['norm-beta-hat'] = norm_b
    return_dict['theta0'] = theta0
    return_dict['theta1'] = theta1
    return_dict['job-id'] = os.getenv('SLURM_JOBID')
    return_dict['job-array-id'] = os.getenv('SLURM_ARRAY_JOB_ID')
    return_dict['calib-error-0'] = [ce0(p) for p in ps]
    return_dict['calib-error-1'] = [ce1(p) for p in ps]
    filename = f'SNR_{s}_pi_{pi}_k_{k}_theta_{theta}_pen_{penalty}_iter_{ITER}.txt'
    with open('out/'+filename, 'w') as f:
        f.writelines(str(return_dict))
else:
    print('Job out of bound.\n')





        





