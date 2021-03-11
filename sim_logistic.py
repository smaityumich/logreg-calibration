import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm
import argparse, itertools


def logreg_calib(n, s = 1, pi = 0.5, k = 0.5, theta = np.pi/2, penalty = 'l1'):
    

    # input sanitization
    if type(n) != int or n < 0:
        raise ValueError('n must be non-negative integer.\n')

    if type(s) != float or s <= 0:
        raise ValueError('s must be positive number.\n')

    if type(pi) != float or pi < 0 or pi > 1 :
        raise ValueError('pi must be in [0, 1].\n')

    if type(k) != float or k <= 0:
        raise ValueError('k must be positive number.\n')


    if type(theta) != float or theta < 0 or theta > np.pi :
        raise ValueError('pi must be in [0, np.pi].\n')

    if penalty not in {'l1', 'l2'}:
        raise ValueError('Not implemented for ' + penalty + ' penalty.\n')



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
    else: 
        raise ValueError('Not implemented for ' + penalty + '.\n')

    ## solver 
    if penalty == 'l1':
        solver = 'liblinear'
    elif penalty == 'l2':
        solver = 'lbfgs'
    else: 
        raise ValueError('Not implemented for ' + penalty + '.\n')

    # data
    x0, x1 = np.random.normal(size = (n0, d)), np.random.normal(size = (n1, d))
    h0, h1 = x0 @ beta0.reshape((-1, 1)) + sigma * np.random.normal(size = (n0, 1)),\
         x1 @ beta1.reshape((-1, 1)) + sigma * np.random.normal(size = (n1, 1))
    y0, y1 = (h0.reshape((-1, )) > 0).astype('float32'), (h1.reshape((-1, )) > 0).astype('float32')
    x_train, y_train = np.concatenate((x0, x1), axis = 0), np.concatenate((y0, y1))

    x0_test, x1_test = np.random.normal(size = (n_test, d)), np.random.normal(size = (n_test, d))
    h0_test, h1_test = x0_test @ beta0.reshape((-1, 1)) + sigma * np.random.normal(size = (n_test, 1)),\
         x1_test @ beta1.reshape((-1, 1)) + sigma * np.random.normal(size = (n_test, 1))
    y0_test, y1_test = (h0_test.reshape((-1, )) > 0).astype('float32'),\
         (h1_test.reshape((-1, )) > 0).astype('float32')

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

    def calibration_error0(p):
        n_calibration = 500
        x1 = norm.ppf(p)/s * np.ones((n_calibration, 1))
        x_rest = np.random.normal(size = (n_calibration, d-1))
        x = np.concatenate((x1, x_rest), axis = 1)
        h = (x @ b.reshape((-1, 1))).reshape((-1, ))
        return p - np.mean((1 / (1 + np.exp(-h))))

    def calibration_error1(p):
        n_calibration = 500
        if theta < np.pi :
            x1 = norm.ppf(p)/s * np.ones((n_calibration, 1))
            x2 = norm.ppf(p)/s * np.tan(theta/2) * np.ones((n_calibration, 1))
        else: 
            x1 = np.zeros((n_calibration, 1))
            x2 = norm.ppf(p)/s * np.ones((n_calibration, 1))
        x_rest = np.random.normal(size = (n_calibration, d-2))
        x = np.concatenate((x1, x2, x_rest), axis = 1)
        h = (x @ b.reshape((-1, 1))).reshape((-1, ))
        return p - np.mean((1 / (1 + np.exp(-h))))

        

    
    # return
    return [score0, score1], [norm_b, theta0, theta1], [calibration_error0, calibration_error1]



## parameter grid
def grid(n_signal = 5, n_pi = 5, n_kappa = 5, n_theta = 5, n_sim = 100):
    
    signals = np.logspace(0.5, 5, n_signal)
    pis = np.linspace(0.1, 0.9, n_pi)
    kappas = np.logspace(0.1, 10, n_kappa)
    thetas = np.linspace(0, 1, n_theta) * np.pi
    penaltys = ['l1', 'l2']
    iters = range(n_sim)

    return list(itertools.product(signals, pis, kappas, thetas, penaltys, iters))


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



    


