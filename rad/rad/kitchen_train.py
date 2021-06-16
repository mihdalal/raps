import argparse

# import dmc2gym
import copy
import json
import os
import time
from collections import OrderedDict

import gym
import numpy as np
import rlkit.pythonplusplus as ppp
import torch
from d4rl.kitchen.kitchen_envs import (
    KitchenHingeCabinetV0,
    KitchenHingeSlideBottomLeftBurnerLightV0,
    KitchenKettleV0,
    KitchenLightSwitchV0,
    KitchenMicrowaveKettleLightTopLeftBurnerV0,
    KitchenMicrowaveV0,
    KitchenSlideCabinetV0,
    KitchenTopLeftBurnerV0,
)
from rlkit.core import logger as rlkit_logger
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.envs.primitives_make_env import make_base_kitchen_env, make_base_metaworld_env
from rlkit.envs.primitives_wrappers import (
    ActionRepeat,
    ImageEnvMetaworld,
    ImageUnFlattenWrapper,
    NormalizeActions,
    TimeLimit,
)
from torchvision import transforms
import rlkit.envs.primitives_make_env as primitives_make_env
import rad.utils as utils
from rad.curl_sac import RadSacAgent
from rad.logger import Logger
from rad.video import VideoRecorder


def parse_args():
    parser = argparse.ArgumentParser()
    # train
    parser.add_argument("--agent", default="rad_sac", type=str)
    parser.add_argument("--hidden_dim", default=1024, type=int)
    # critic
    parser.add_argument("--critic_beta", default=0.9, type=float)
    parser.add_argument("--critic_tau", default=0.01, type=float)  # try 0.05 or 0.1
    parser.add_argument(
        "--critic_target_update_freq", default=2, type=int
    )  # try to change it to 1 and retain 0.01 above
    # actor
    parser.add_argument("--actor_beta", default=0.9, type=float)
    parser.add_argument("--actor_log_std_min", default=-10, type=float)
    parser.add_argument("--actor_log_std_max", default=2, type=float)
    parser.add_argument("--actor_update_freq", default=2, type=int)
    parser.add_argument("--discrete_continuous_dist", default=0, type=int)
    # encoder
    parser.add_argument("--encoder_feature_dim", default=50, type=int)
    parser.add_argument("--encoder_tau", default=0.05, type=float)
    parser.add_argument("--num_layers", default=4, type=int)
    parser.add_argument("--num_filters", default=32, type=int)
    parser.add_argument("--latent_dim", default=128, type=int)
    # sac
    parser.add_argument("--init_temperature", default=0.1, type=float)
    parser.add_argument("--alpha_lr", default=1e-4, type=float)
    parser.add_argument("--alpha_beta", default=0.5, type=float)
    # misc
    parser.add_argument("--save_tb", default=False, action="store_true")
    parser.add_argument("--save_buffer", default=False, action="store_true")
    parser.add_argument("--save_video", default=False, action="store_true")
    parser.add_argument("--save_model", default=False, action="store_true")
    parser.add_argument("--detach_encoder", default=False, action="store_true")

    parser.add_argument("--log_interval", default=100, type=int)
    args = parser.parse_args()
    return args


def compute_path_info(infos):
    all_env_infos = [ppp.list_of_dicts__to__dict_of_lists(ep_info) for ep_info in infos]
    statistics = OrderedDict()
    stat_prefix = ""
    for k in all_env_infos[0].keys():
        final_ks = np.array([info[k][-1] for info in all_env_infos])
        first_ks = np.array([info[k][0] for info in all_env_infos])
        all_ks = np.concatenate([info[k] for info in all_env_infos])
        statistics.update(
            create_stats_ordered_dict(
                stat_prefix + k,
                final_ks,
                stat_prefix="{}/final/".format("env_infos"),
            )
        )
        statistics.update(
            create_stats_ordered_dict(
                stat_prefix + k,
                first_ks,
                stat_prefix="{}/initial/".format("env_infos"),
            )
        )
        statistics.update(
            create_stats_ordered_dict(
                stat_prefix + k,
                all_ks,
                stat_prefix="{}/".format("env_infos"),
            )
        )
    return statistics


def evaluate(
    env,
    agent,
    video,
    num_episodes,
    L,
    step,
    encoder_type,
    data_augs,
    image_size,
    pre_transform_image_size,
    env_name,
    action_repeat,
    work_dir,
    seed,
):
    all_ep_rewards = []
    all_infos = []

    def run_eval_loop(sample_stochastically=True):
        start_time = time.time()
        prefix = "stochastic_" if sample_stochastically else ""
        for i in range(num_episodes):
            obs = env.reset()
            video.init(enabled=(i == 0))
            done = False
            episode_reward = 0
            ep_infos = []
            while not done:
                # center crop image
                if encoder_type == "pixel" and "crop" in data_augs:
                    obs = utils.center_crop_image(obs, image_size)
                if encoder_type == "pixel" and "translate" in data_augs:
                    # first crop the center with pre_transform_image_size
                    obs = utils.center_crop_image(obs, pre_transform_image_size)
                    # then translate cropped to center
                    obs = utils.center_translate(obs, image_size)
                with utils.eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs / 255.0)
                    else:
                        action = agent.select_action(obs / 255.0)
                obs, reward, done, info = env.step(action)
                video.record(env)
                episode_reward += reward
                ep_infos.append(info)

            video.save("%d.mp4" % step)
            L.log("eval/" + prefix + "episode_reward", episode_reward, step)
            all_ep_rewards.append(episode_reward)
            all_infos.append(ep_infos)

        L.log("eval/" + prefix + "eval_time", time.time() - start_time, step)
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        std_ep_reward = np.std(all_ep_rewards)
        L.log("eval/" + prefix + "mean_episode_reward", mean_ep_reward, step)
        L.log("eval/" + prefix + "best_episode_reward", best_ep_reward, step)
        rlkit_logger.record_dict(
            {"Average Returns": mean_ep_reward}, prefix="evaluation/"
        )
        statistics = compute_path_info(all_infos)
        rlkit_logger.record_dict(statistics, prefix="evaluation/")

        filename = (
            work_dir
            + "/"
            + env_name
            + "-"
            + data_augs
            + "--s"
            + str(seed)
            + "--eval_scores.npy"
        )
        key = env_name + "-" + data_augs
        try:
            log_data = np.load(filename, allow_pickle=True)
            log_data = log_data.item()
        except:
            log_data = {}

        if key not in log_data:
            log_data[key] = {}

        log_data[key][step] = {}
        log_data[key][step]["step"] = step
        log_data[key][step]["mean_ep_reward"] = mean_ep_reward
        log_data[key][step]["max_ep_reward"] = best_ep_reward
        log_data[key][step]["std_ep_reward"] = std_ep_reward
        log_data[key][step]["env_step"] = step * action_repeat

        np.save(filename, log_data)

    run_eval_loop(sample_stochastically=False)
    L.dump(step)


def make_agent(
    obs_shape,
    continuous_action_dim,
    discrete_action_dim,
    args,
    agent_kwargs,
    device,
):
    if args.agent == "rad_sac":
        return RadSacAgent(
            obs_shape=obs_shape,
            continuous_action_dim=continuous_action_dim,
            discrete_action_dim=discrete_action_dim,
            device=device,
            hidden_dim=args.hidden_dim,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            latent_dim=args.latent_dim,
            **agent_kwargs,
        )
    else:
        assert "agent is not supported: %s" % args.agent


def experiment(variant):
    gym.logger.set_level(40)
    work_dir = rlkit_logger.get_snapshot_dir()
    args = parse_args()
    seed = int(variant["seed"])
    utils.set_seed_everywhere(seed)
    os.makedirs(work_dir, exist_ok=True)
    agent_kwargs = variant["agent_kwargs"]
    data_augs = agent_kwargs["data_augs"]
    encoder_type = agent_kwargs["encoder_type"]
    discrete_continuous_dist = agent_kwargs["discrete_continuous_dist"]

    env_suite = variant["env_suite"]
    env_name = variant["env_name"]
    env_kwargs = variant["env_kwargs"]
    pre_transform_image_size = variant["pre_transform_image_size"]
    image_size = variant["image_size"]
    frame_stack = variant["frame_stack"]
    batch_size = variant["batch_size"]
    replay_buffer_capacity = variant["replay_buffer_capacity"]
    num_train_steps = variant["num_train_steps"]
    num_eval_episodes = variant["num_eval_episodes"]
    eval_freq = variant["eval_freq"]
    action_repeat = variant["action_repeat"]
    init_steps = variant["init_steps"]
    log_interval = variant["log_interval"]
    use_raw_actions = variant["use_raw_actions"]
    pre_transform_image_size = (
        pre_transform_image_size if "crop" in data_augs else image_size
    )
    pre_transform_image_size = pre_transform_image_size

    if data_augs == "crop":
        pre_transform_image_size = 100
        image_size = image_size
    elif data_augs == "translate":
        pre_transform_image_size = 100
        image_size = 108

    if env_suite == 'kitchen':
        env_kwargs['imwidth'] = pre_transform_image_size
        env_kwargs['imheight'] = pre_transform_image_size
    else:
        env_kwargs['image_kwargs']['imwidth'] = pre_transform_image_size
        env_kwargs['image_kwargs']['imheight'] = pre_transform_image_size

    expl_env = primitives_make_env.make_env(env_suite, env_name, env_kwargs)
    eval_env = primitives_make_env.make_env(env_suite, env_name, env_kwargs)
    # stack several consecutive frames together
    if encoder_type == "pixel":
        expl_env = utils.FrameStack(expl_env, k=frame_stack)
        eval_env = utils.FrameStack(eval_env, k=frame_stack)

    # make directory
    ts = time.gmtime()
    ts = time.strftime("%m-%d", ts)
    env_name = env_name
    exp_name = (
        env_name
        + "-"
        + ts
        + "-im"
        + str(image_size)
        + "-b"
        + str(batch_size)
        + "-s"
        + str(seed)
        + "-"
        + encoder_type
    )
    work_dir = work_dir + "/" + exp_name

    utils.make_dir(work_dir)
    video_dir = utils.make_dir(os.path.join(work_dir, "video"))
    model_dir = utils.make_dir(os.path.join(work_dir, "model"))
    buffer_dir = utils.make_dir(os.path.join(work_dir, "buffer"))

    video = VideoRecorder(video_dir if args.save_video else None)

    with open(os.path.join(work_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if use_raw_actions:
        continuous_action_dim = expl_env.action_space.low.size
        discrete_action_dim = 0
    else:
        num_primitives = expl_env.num_primitives
        max_arg_len = expl_env.max_arg_len
        if discrete_continuous_dist:
            continuous_action_dim = max_arg_len
            discrete_action_dim = num_primitives
        else:
            continuous_action_dim = max_arg_len + num_primitives
            discrete_action_dim = 0

    if encoder_type == "pixel":
        obs_shape = (3 * frame_stack, image_size, image_size)
        pre_aug_obs_shape = (
            3 * frame_stack,
            pre_transform_image_size,
            pre_transform_image_size,
        )
    else:
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape

    replay_buffer = utils.ReplayBuffer(
        obs_shape=pre_aug_obs_shape,
        action_size=continuous_action_dim + discrete_action_dim,
        capacity=replay_buffer_capacity,
        batch_size=batch_size,
        device=device,
        image_size=image_size,
        pre_image_size=pre_transform_image_size,
    )

    agent = make_agent(
        obs_shape=obs_shape,
        continuous_action_dim=continuous_action_dim,
        discrete_action_dim=discrete_action_dim,
        args=args,
        device=device,
        agent_kwargs=agent_kwargs,
    )

    L = Logger(work_dir, use_tb=args.save_tb)

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()
    epoch_start_time = time.time()
    train_expl_st = time.time()
    total_train_expl_time = 0
    all_infos = []
    ep_infos = []
    num_train_calls = 0
    for step in range(num_train_steps):
        # evaluate agent periodically

        if step % eval_freq == 0:
            total_train_expl_time += time.time()-train_expl_st
            L.log("eval/episode", episode, step)
            evaluate(
                eval_env,
                agent,
                video,
                num_eval_episodes,
                L,
                step,
                encoder_type,
                data_augs,
                image_size,
                pre_transform_image_size,
                env_name,
                action_repeat,
                work_dir,
                seed,
            )
            if args.save_model:
                agent.save_curl(model_dir, step)
            if args.save_buffer:
                replay_buffer.save(buffer_dir)
            train_expl_st = time.time()
        if done:
            if step > 0:
                if step % log_interval == 0:
                    L.log("train/duration", time.time() - epoch_start_time, step)
                    L.dump(step)
            if step % log_interval == 0:
                L.log("train/episode_reward", episode_reward, step)
            obs = expl_env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            if step % log_interval == 0:
                all_infos.append(ep_infos)

                L.log("train/episode", episode, step)
                statistics = compute_path_info(all_infos)

                rlkit_logger.record_dict(statistics, prefix="exploration/")
                rlkit_logger.record_tabular(
                    "time/epoch (s)", time.time() - epoch_start_time
                )
                rlkit_logger.record_tabular("time/total (s)", time.time() - start_time)
                rlkit_logger.record_tabular("time/training and exploration (s)", total_train_expl_time)
                rlkit_logger.record_tabular("trainer/num train calls", num_train_calls)
                rlkit_logger.record_tabular("exploration/num steps total", step)
                rlkit_logger.record_tabular("Epoch", step // log_interval)
                rlkit_logger.dump_tabular(with_prefix=False, with_timestamp=False)
                all_infos = []
                epoch_start_time = time.time()
            ep_infos = []


        # sample action for data collection
        if step < init_steps:
            action = expl_env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs / 255.0)

        # run training update
        if step >= init_steps:
            num_updates = 1
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)
                num_train_calls += 1

        next_obs, reward, done, info = expl_env.step(action)
        ep_infos.append(info)
        # allow infinit bootstrap
        done_bool = (
            0 if episode_step + 1 == expl_env._max_episode_steps else float(done)
        )
        episode_reward += reward
        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        obs = next_obs
        episode_step += 1
