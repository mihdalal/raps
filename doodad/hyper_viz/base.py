from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


class Experiment(object):
    def __init__(self, params_dict, values, performance=0.0):
        self.params = params_dict
        self.values = values
        self.performance = performance

    def satisfies_param(param_k, param_v):
        return self.params[param_k] == param_v


def find_unique_params(experiments):
    params_dict = defaultdict(set)
    for experiment in experiments:
        exp_params = experiment.params
        for k in exp_params:
            v = exp_params[k]
            if isinstance(v, set):
                raise NotImplementedError()
            params_dict[k].add(v)
    for k in list(params_dict.keys()):
        if len(params_dict[k]) == 1:
            del params_dict[k]
    return params_dict


def make_3d_plot(experiments, xkey, ykey, logx=True, logy=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    z = [exp.performance for exp in experiments]
    x = [exp.params[xkey] for exp in experiments]
    y = [exp.params[ykey] for exp in experiments]
    ax.set_xlabel(xkey)
    ax.set_ylabel(ykey)
    ax.set_zlabel("Performance")

    if logx:
        x = np.log(np.array(x) + 1e-5) / np.log(10)
        ax.set_xlabel("log " + xkey)
    if logy:
        y = np.log(np.array(y) + 1e-5) / np.log(10)
        ax.set_ylabel("log " + ykey)

    ax.scatter(x, y, z)
    plt.show()


def resize_ticks(data, maxlen):
    ldata = len(data)
    ticks_per_data = maxlen / ldata
    resized = []
    j = 0
    for i in range(maxlen):
        if i % ticks_per_data == 0:
            resized.append(data[j])
            j += 1
        else:
            resized.append("")
    return resized


def make_2d_plot(experiments, xkey, ykey, title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    z = [exp.performance for exp in experiments]
    x = [exp.params[xkey] for exp in experiments]
    y = [exp.params[ykey] for exp in experiments]
    ax.set_xlabel(xkey)
    ax.set_ylabel(ykey)
    x_vals_uniq = sorted(list(set([exp.params[xkey] for exp in experiments])))
    y_vals_uniq = sorted(list(set([exp.params[ykey] for exp in experiments])))

    xticks = ax.get_xticks().tolist()
    ax.set_xticklabels(resize_ticks(x_vals_uniq, len(xticks)))
    yticks = ax.get_yticks().tolist()
    ax.set_yticklabels(resize_ticks(y_vals_uniq, len(yticks)))

    data = np.zeros((len(x_vals_uniq), len(y_vals_uniq)))
    for i in range(len(x_vals_uniq)):
        for j in range(len(y_vals_uniq)):
            exps = [
                exp
                for exp in experiments
                if (exp.params[xkey] == x_vals_uniq[i])
                and (exp.params[ykey] == y_vals_uniq[j])
            ]
            avg_perf = np.mean([exp.performance for exp in exps])
            data[i, j] = avg_perf
    ax.imshow(data.T, cmap="YlOrRd")
    for i in range(len(x_vals_uniq)):
        for j in range(len(y_vals_uniq)):
            ax.annotate("%.2f" % data[i, j], xy=(i - 0.3, j))

    if title is not None:
        plt.title(title)
    plt.show()
