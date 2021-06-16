import numpy as np
import torch
from rad.kitchen_train import compute_path_info
from rlkit.core import logger as rlkit_logger

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs


def evaluate(actor_critic, eval_env_args, eval_env_kwargs, obs_rms, num_episodes, device):
    eval_envs = make_vec_envs(
        *eval_env_args, disable_time_limit_mask=True, **eval_env_kwargs
    )
    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms
    rewards = 0
    all_infos = []
    for i in range(num_episodes):
        obs = eval_envs.reset()
        eval_recurrent_hidden_states = torch.zeros(
            1, actor_critic.recurrent_hidden_state_size, device=device
        )
        eval_masks = torch.zeros(1, 1, device=device)
        done = [False] * 1
        ep_infos = []
        while not all(done):
            with torch.no_grad():
                _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                    obs, eval_recurrent_hidden_states, eval_masks, deterministic=True
                )

            # Obser reward and next obs
            obs, reward, done, infos = eval_envs.step(action)

            eval_masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device,
            )
            info = infos[0]
            if "terminal_observation" in info:
                del info["terminal_observation"]
            ep_infos.append(infos[0])

            rewards += reward
        all_infos.append(ep_infos)
    mean_ep_reward = rewards.sum().item() / num_episodes
    rlkit_logger.record_dict({"Average Returns": mean_ep_reward}, prefix="evaluation/")
    statistics = compute_path_info(all_infos)
    rlkit_logger.record_dict(statistics, prefix="evaluation/")
    print(
        " Evaluation using {} episodes: mean reward {:.5f}\n".format(
            num_episodes, mean_ep_reward
        )
    )
