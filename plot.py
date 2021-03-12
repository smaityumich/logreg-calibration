import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd
from matplotlib.lines import Line2D

ps = np.array(range(1, 10))/10

def edit_dict(d):
    c0 = d['calib-error-0']
    c1 = d['calib-error-1']
    for i, p in enumerate(ps):
        d[f'cal-error-0-p-{p}'] = c0[i]
        d[f'cal-error-1-p-{p}'] = c1[i]
    return d




def plot(df, xaxis = 'k', pi = 0.1, k = 0.05, s = 0.5, theta = np.pi/4,\
 p_choice = [0.2, 0.8], penalty = 'l1', measure = 'cal-error-0'):

    parameters = ['pi', 'k', 'theta', 's']
    par_choice = [pi, k, theta, s]
    


    ms = ['score0', 'score1', 'norm-beta-hat', 'theta0', 'theta1']
    ms += [f'cal-error-0-p-{p}' for p in ps] + [f'cal-error-1-p-{p}' for p in ps]
    agg_dict = dict()
    for key in ms:
        agg_dict[key] = ['mean', 'std']
    result = df.groupby(['s', 'pi', 'k', 'theta', 'penalty'], as_index=False).agg(agg_dict)


    result_selected = result
    for pm, ch in zip(parameters, par_choice):
        if pm != xaxis:
            result_selected = result_selected.loc[result_selected[pm] == ch]

    result_selected = result_selected.loc[result_selected['penalty'] == penalty]


    ltys = ['-', '--', ':', '-.']
    markers = ['*', 'x', '+', 'o']
    cols = ['blue', 'orange', 'black', 'green']

    if measure in ['cal-error-0', 'cal-error-1']:
        g = re.split('-', measure)[2]
        measures = [measure + f'-p-{p}' for p in p_choice]
        pn = len(p_choice)
        ltys = ltys[:pn]
        markers = markers[:pn]
        cols = cols[:pn]

        lines = []
        labels = []

        for p, m, c, lty, mk in zip(p_choice, measures, cols, ltys, markers):
            x = result_selected[xaxis]
            m_mean, m_std = result_selected[m]['mean'], result_selected[m]['std']
            plt.errorbar(x, m_mean, m_std, color = c, linestyle = lty,\
                marker = mk, markersize = 8, lw = 1, alpha = 1)
            lines.append(Line2D([0], [0], color=c, linestyle=lty, marker=mk, lw = 1, alpha = 1))
            labels.append(f'p = {p}')
        plt.legend(lines, labels)
        plt.xlabel(xaxis, size = 'large')
        plt.ylabel('$p - P_{'+str(g)+ r'}[Y = 1\| \hat f(X) = p ]$')
        if xaxis in ['s', 'k']:
            plt.xscale('log')



        

    elif measure in ['theta', 'score']:
        measures = [measure + f'{g}' for g in range(2)]
        ltys = ltys[:2]
        markers = markers[:2]
        cols = cols[:2]

        lines = []
        labels = []
        for g, (m, c, lty, mk) in enumerate(zip(measures, cols, ltys, markers)):
            x = result_selected[xaxis]
            m_mean, m_std = result_selected[m]['mean'], result_selected[m]['std']
            plt.errorbar(x, m_mean, m_std, color = c, linestyle = lty,\
                marker = mk, markersize = 8, lw = 1, alpha = 1)
            lines.append(Line2D([0], [0], color=c, linestyle=lty, marker=m, lw = 1, alpha = 1))
            labels.append(f'g = {g}')
        plt.legend(lines, labels, title = measure)
        plt.xlabel(xaxis, size = 'large')
        if xaxis in ['s', 'k']:
            plt.xscale('log')
        if measure == 'score':
            plt.ylabel(f'Test accuracy')
        else:
            plt.ylabel('$\\cos(\\hat\\beta, \\beta)$')
    else:
        raise ValueError('Wrong measure.\n')



    
    
    