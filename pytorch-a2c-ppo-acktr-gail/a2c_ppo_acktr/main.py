import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rlkit.core import logger as rlkit_logger
from rlkit.core.eval_util import create_stats_ordered_dict

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.evaluation import evaluate
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage


def experiment(variant):
    env_name = variant["env_name"]
    env_suite = variant["env_suite"]
    env_kwargs = variant["env_kwargs"]
    multi_step_horizon = variant.get('multi_step_horizon', 1)
    seed = variant["seed"]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    log_dir = os.path.expanduser(rlkit_logger.get_snapshot_dir())
    utils.cleanup_log_dir(log_dir)

    device = torch.device("cuda:0")

    envs = make_vec_envs(
        env_suite,
        env_name,
        env_kwargs,
        seed,
        variant["num_processes"],
        variant["rollout_kwargs"]["gamma"],
        rlkit_logger.get_snapshot_dir(),
        device,
        False,
    )

    eval_env_args = (env_suite,
        env_name,
        env_kwargs,
        seed,
        1,
        variant["rollout_kwargs"]["gamma"],
        rlkit_logger.get_snapshot_dir(),
        device,
        False,)

    eval_env_kwargs = dict()

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs=variant['actor_kwargs'],
        discrete_continuous_dist=variant.get("discrete_continuous_dist", False),
        env = envs,
        multi_step_horizon=multi_step_horizon,
    )
    actor_critic.to(device)

    agent = algo.PPO(actor_critic, **variant["algorithm_kwargs"])
    torch.backends.cudnn.benchmark = True
    rollouts = RolloutStorage(
        variant["num_steps"],
        variant["num_processes"],
        envs.observation_space.shape,
        envs.action_space,
        actor_critic.recurrent_hidden_state_size,
    )

    obs = envs.reset()
    policy_step_obs = obs
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = (
        int(variant["num_env_steps"])
        // variant["num_steps"]
        // variant["num_processes"]
    )
    num_train_calls = 0
    total_train_expl_time = 0
    for j in range(num_updates):
        epoch_start_time = time.time()
        train_expl_st = time.time()
        if variant['use_linear_lr_decay']:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer,
                j,
                num_updates,
                variant['algorithm_kwargs']['lr'],
            )

        for step in range(variant["num_steps"]):
            # Sample actions
            with torch.no_grad():
                (
                    value,
                    action,
                    action_log_prob,
                    recurrent_hidden_states,
                ) = actor_critic.act(
                    rollouts.obs[step],
                    rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step],
                    index = step % multi_step_horizon
                )
            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            # for info in infos:
                # if "episode" in info.keys():
                #     episode_rewards.append(info["episode"]["r"])

            # for r in reward:
            #     episode_rewards.append(r)

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if "bad_transition" in info.keys() else [1.0] for info in infos]
            )
            if (step+1) % multi_step_horizon != 0:
                # +1 because you would take the action from the reset state
                obs = policy_step_obs
            else:
                policy_step_obs = obs


            if all(done):
                obs = envs.reset()
                policy_step_obs = obs

            rollouts.insert(
                obs,
                recurrent_hidden_states,
                action,
                action_log_prob,
                value,
                reward,
                masks,
                bad_masks,
                torch.tensor([step%multi_step_horizon]*variant["num_processes"]),
            )

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1],
                rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1],
                step%multi_step_horizon
            ).detach()
        rollouts.compute_returns(next_value, **variant["rollout_kwargs"])

        value_loss, action_loss, dist_entropy, num_calls = agent.update(rollouts)
        num_train_calls += num_calls

        rollouts.after_update()

        total_train_expl_time += time.time()-train_expl_st
        if variant["eval_interval"] is not None and j % variant["eval_interval"] == 0:
            total_num_steps = (j + 1) * variant["num_processes"] * variant["num_steps"]
            obs_rms = utils.get_vec_normalize(envs).obs_rms
            evaluate(
                actor_critic,
                eval_env_args,
                eval_env_kwargs,
                obs_rms,
                5,
                device,
            )
            rlkit_logger.record_tabular(
                "time/epoch (s)", time.time() - epoch_start_time
            )
            rlkit_logger.record_tabular("time/total (s)", time.time() - start)
            rlkit_logger.record_tabular("time/training and exploration (s)", total_train_expl_time)
            rlkit_logger.record_tabular("exploration/num steps total", total_num_steps)
            rlkit_logger.record_tabular("trainer/num train calls", num_train_calls)
            rlkit_logger.record_tabular("Epoch", j // variant["eval_interval"])
            rlkit_logger.dump_tabular(with_prefix=False, with_timestamp=False)
