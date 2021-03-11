# Effect of distribution shift on model calibration

To run a demo experiment:
``` bash
python3 sim_logistic.py --n 2000 --s 2 --pi 0.9 --k 0.5 --theta 1 --penality l1 --p 0.2 0.4 0.6 0.8
```

The arguments are tabulated below. 

|Args| Description | 
| --- | --- |
| `--n` | Total sample size |
|`--s` | Signal strength (SNR) |
|`--pi` | Proportion of group 1 sample |
| `--k` | Dimension / (sample  size) | 
| `--theta` | Angle between $\beta_0$ and $\beta_1$ | 
| `--penalty` | Penalty for logistic regression|
|`--p` | $p$'s for which calibration errors to be calculated | 

