import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm


def sim(n, s = 1, pi = 0.5, k = 0.5, theta = np.pi/2, penalty = 'l1'):
    

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
        x1 = norm.ppf(p)/s * np.ones((n_calibration, 1))
        x2 = norm.ppf(p)/s * np.tan(theta/2) * np.ones((n_calibration, 1))
        x_rest = np.random.normal(size = (n_calibration, d-2))
        x = np.concatenate((x1, x2, x_rest), axis = 1)
        h = (x @ b.reshape((-1, 1))).reshape((-1, ))
        return p - np.mean((1 / (1 + np.exp(-h))))

        

    
    # return
    return [score0, score1], [norm_b, theta0, theta1], [calibration_error0, calibration_error1]

