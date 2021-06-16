import csv
import json
import os
from collections import defaultdict

import numpy as np
from base import Experiment

N_PERF = 5


def get_experiments(dirname, perf_key="AverageReturn"):
    # look recursively for directories containing a params.json file
    print("get_experiments looking in %s" % dirname)
    lsdir = list(os.listdir(dirname))
    if any(params_file in lsdir for params_file in ["variant.json", "params.json"]):
        print("\t Found experiment directory")
        params_file_name = "params.json"
        if "variant.json" in lsdir:
            params_file_name = "variant.json"
        return [parse_exp_dir(dirname, params_file_name, perf_key=perf_key)]
    else:
        exps = []
        for item in lsdir:
            full_path = os.path.join(dirname, item)
            if os.path.isdir(full_path):
                exps.extend(get_experiments(full_path))
        return exps


def flatten_kv_dict(orig_dict, join_char=":"):
    """
    >>> flatten_kv_dict({'a': {'b': 2, 'c': 3}, 'd': 4})
    """
    flat_dict = {}
    for k in orig_dict:
        v = orig_dict[k]
        if isinstance(v, dict):
            flattened_dict = flatten_kv_dict(v, join_char=join_char)
            for k_ in flattened_dict:
                flat_dict["%s%s%s" % (k, join_char, k_)] = flattened_dict[k_]
        else:
            flat_dict[k] = v
    return flat_dict


def parse_exp_dir(dirname, params_file_name, perf_key="AverageReturn"):
    with open(os.path.join(dirname, params_file_name)) as f:
        params_dict = json.loads(f.read())
        params_dict = flatten_kv_dict(params_dict)
    with open(os.path.join(dirname, "progress.csv")) as f:
        readCSVFile = csv.reader(f, delimiter=",")
        for i, row in enumerate(readCSVFile):
            if i == 0:
                headers = row
                keyValueMap = [[] for _ in range(len(row))]
            else:
                for j, rowItem in enumerate(row):
                    keyValueMap[j].append(float(rowItem))
        keyValueMap = dict(zip(headers, keyValueMap))
    return Experiment(
        params_dict, keyValueMap, performance=np.mean(keyValueMap[perf_key][-N_PERF:])
    )
