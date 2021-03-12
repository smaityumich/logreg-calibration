import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm
import argparse, itertools, functools


def calib_error(p, norm_b, theta, s):

    z = np.random.normal(size = (100, ))
    calib = (s/norm_b) * np.cos(theta) * (p/(1-p)) + s * np.sin(theta) * z
    calib = 1/(1+np.exp(-calib))
    return p - np.mean(calib)

def logreg_calib(n, s = 1, pi = 0.5, k = 0.5, theta = np.pi/2, penalty = 'l1'):
    
    # parameters 

    ## model pars
    d = int(n * k)
    n0, n1 = n - int(n * pi), int(n * pi)
    sigma = 0.2
    n_test = 1000

    ## coefficients
    beta0, beta1 = np.zeros(shape = (d, )), np.zeros(shape = (d, ))
    beta0[0] = s
    beta1[0], beta1[1] = s * np.cos(theta), s * np.sin(theta)

    ## regularizer 
    if penalty == 'l1':
        regularizer = sigma * np.sqrt(np.log(d)/n)
    elif penalty == 'l2':
        regularizer = sigma ** 2 * np.log(d)/n
    elif penalty == 'none':
        regularizer = 1
    else: 
        raise ValueError('Not implemented for ' + penalty + '.\n')

    ## solver 
    if penalty == 'l1':
        solver = 'liblinear'
    elif penalty == 'l2':
        solver = 'lbfgs'
    elif penalty == 'none':
        solver = 'lbfgs'
    else: 
        raise ValueError('Not implemented for ' + penalty + '.\n')

    # data
    x0, x1 = np.random.normal(size = (n0, d)), np.random.normal(size = (n1, d))
    h0, h1 = x0 @ beta0.reshape((-1, 1)) + sigma * np.random.normal(size = (n0, 1)),\
         x1 @ beta1.reshape((-1, 1)) + sigma * np.random.normal(size = (n1, 1))
    h0, h1 = h0.reshape((-1, )), h1.reshape((-1, ))
    p0, p1 = 1/(1+np.exp(-h0)), 1/(1+np.exp(-h1))
    y0, y1 = np.random.binomial(1, p0, size = (n0, )), np.random.binomial(1, p1, size = (n1, ))
    x_train, y_train = np.concatenate((x0, x1), axis = 0), np.concatenate((y0, y1))

    x0_test, x1_test = np.random.normal(size = (n_test, d)), np.random.normal(size = (n_test, d))
    h0_test, h1_test = x0_test @ beta0.reshape((-1, 1)) + sigma * np.random.normal(size = (n_test, 1)),\
         x1_test @ beta1.reshape((-1, 1)) + sigma * np.random.normal(size = (n_test, 1))
    h0_test, h1_test = h0_test.reshape((-1, )), h1_test.reshape((-1, ))
    p0_test, p1_test = 1/(1+np.exp(-h0_test)), 1/(1+np.exp(-h1_test))
    y0_test, y1_test = np.random.binomial(1, p0_test, size = (n_test, )),\
     np.random.binomial(1, p1_test, size = (n_test, ))

    # model 
    cl = LogisticRegression(penalty = penalty, C = 1/regularizer, fit_intercept = False, solver = solver)
    cl.fit(x_train, y_train)


    # evaluation
    
    ## data: 0, 1
    score0, score1 = cl.score(x0_test, y0_test), cl.score(x1_test, y1_test)
    b = cl.coef_.reshape((-1, ))
    norm_b = np.linalg.norm(b)
    theta0, theta1 = np.arccos(np.dot(b, beta0)/ (s * norm_b)), np.arccos(np.dot(b, beta1)/ (s * norm_b))

    ## calibration error

    calibration_error0 = functools.partial(calib_error, norm_b = norm_b, theta = theta0, s = s)
    calibration_error1 = functools.partial(calib_error, norm_b = norm_b, theta = theta1, s = s)

    # return
    return [score0, score1], [norm_b, theta0, theta1], [calibration_error0, calibration_error1]



## parameter grid
def grid(n_signal = 5, n_pi = 5, n_kappa = 5, n_theta = 5):
    
    signals = np.logspace(-1, 0, n_signal) * 5
    pis = np.linspace(0.1, 0.9, n_pi)
    kappas = np.logspace(-1, 1, n_kappa, base = 4)
    thetas = np.linspace(0, 1, n_theta) * np.pi
    penaltys = ['l1', 'l2', 'none']

    return list(itertools.product(signals, pis, kappas, thetas, penaltys))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Arguments for logreg-calibration')

    parser.add_argument('--n', dest='n', type=int, nargs = 1,\
         default = 5000, help='total sample size')
    parser.add_argument('--s', dest='s', type=float, nargs = 1,\
         default = 2, help='signal strength')
    parser.add_argument('--pi', dest='pi', type=float, nargs = 1,\
         default = 0.9, help='proportion of group 1')
    parser.add_argument('--k', dest='k', type=float, nargs = 1,\
         default = 0.5, help='overparametrization parameter')
    parser.add_argument('--theta', dest='theta', type=float, nargs = 1,\
         default = 1.57, help='angle between beta0 and beta1 in radian')
    parser.add_argument('--penalty', dest='penalty', type=str, nargs = 1,\
         default = 'l1', help='penalty for logistic regression')
    parser.add_argument('--p', dest = 'p', type = float, nargs = '*',\
         default = [0.5], help = 'original calibration')

    args = parser.parse_args()
    n = args.n
    s = args.s
    pi = args.pi
    k = args.k
    theta = args.theta
    penalty = args.penalty


    [score0, score1], [norm_b, theta0, theta1], [ce0, ce1] =\
         logreg_calib(n = n, s = s, pi = pi, k = k, theta = theta, penalty = penalty)
    
    print('\n'+'-' * 100 + '\n')
    print(f'Test accuracy of group 0: {score0}\n')
    print(f'Test accuracy of group 1: {score1}\n'+ '-' * 100 + '\n')

    print(f'Norm of estimated coefficient vector: {norm_b}\n')
    print(f'Angle (radian) between estimated coefficient vector and beta_0: {theta0}\n')
    print(f'Angle (radian) between estimated coefficient vector and beta_1: {theta1}\n'+ '-' * 100 + '\n')

    for p in args.p:
        print(f'For p = {p} ')
        print(f'calibration error\n' + '-' * 30 + '\n') 
        print(f'Group 0: {ce0(p)}\n')
        print(f'Group 1: {ce1(p)}\n'+ '-' * 50 + '\n')

    print('-' * 100 + '\n')



    


