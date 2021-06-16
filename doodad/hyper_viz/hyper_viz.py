import argparse

from base import *
from rllab_interface import get_experiments

if __name__ == "__main__":
    # exps = get_experiments('hopper_monotone', perf_key='Returns Average')
    exps = get_experiments("walker_09_06_17", perf_key="Returns Average")
    env_name = exps[0].params["env_params:gym_name"]
    make_2d_plot(
        exps,
        xkey="algo_params:n_updates_per_time_step",
        ykey="algo_params:monotone_constraint_wt",
        title=env_name,
    )
