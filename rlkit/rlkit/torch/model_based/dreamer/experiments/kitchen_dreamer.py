def experiment(variant):
    import os

    os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

    import torch

    import rlkit.envs.primitives_make_env as primitives_make_env
    import rlkit.torch.pytorch_util as ptu
    from rlkit.envs.mujoco_vec_wrappers import DummyVecEnv, StableBaselinesVecEnv
    from rlkit.torch.model_based.dreamer.actor_models import (
        ActorModel,
        ConditionalActorModel,
    )
    from rlkit.torch.model_based.dreamer.dreamer import DreamerTrainer
    from rlkit.torch.model_based.dreamer.dreamer_policy import (
        ActionSpaceSamplePolicy,
        DreamerPolicy,
    )
    from rlkit.torch.model_based.dreamer.dreamer_v2 import DreamerV2Trainer
    from rlkit.torch.model_based.dreamer.episode_replay_buffer import (
        EpisodeReplayBuffer,
    )
    from rlkit.torch.model_based.dreamer.kitchen_video_func import video_post_epoch_func
    from rlkit.torch.model_based.dreamer.mlp import Mlp
    from rlkit.torch.model_based.dreamer.path_collector import VecMdpPathCollector
    from rlkit.torch.model_based.dreamer.world_models import WorldModel
    from rlkit.torch.model_based.plan2explore.actor_models import (
        ConditionalContinuousActorModel,
    )
    from rlkit.torch.model_based.rl_algorithm import TorchBatchRLAlgorithm

    env_suite = variant.get("env_suite", "kitchen")
    env_name = variant["env_name"]
    env_kwargs = variant["env_kwargs"]
    use_raw_actions = variant["use_raw_actions"]
    num_expl_envs = variant["num_expl_envs"]
    actor_model_class_name = variant.get("actor_model_class", "actor_model")

    if num_expl_envs > 1:
        env_fns = [
            lambda: primitives_make_env.make_env(env_suite, env_name, env_kwargs)
            for _ in range(num_expl_envs)
        ]
        expl_env = StableBaselinesVecEnv(env_fns=env_fns, start_method="fork")
    else:
        expl_envs = [primitives_make_env.make_env(env_suite, env_name, env_kwargs)]
        expl_env = DummyVecEnv(
            expl_envs, pass_render_kwargs=variant.get("pass_render_kwargs", False)
        )
    if env_suite != "real_robot":
        eval_envs = [
            primitives_make_env.make_env(env_suite, env_name, env_kwargs)
            for _ in range(1)
        ]
        eval_env = DummyVecEnv(
            eval_envs, pass_render_kwargs=variant.get("pass_render_kwargs", False)
        )
    else:
        eval_envs = expl_envs
        eval_env = expl_env
    if use_raw_actions:
        discrete_continuous_dist = False
        continuous_action_dim = eval_env.action_space.low.size
        discrete_action_dim = 0
        use_batch_length = True
        action_dim = continuous_action_dim
    else:
        discrete_continuous_dist = variant["actor_kwargs"]["discrete_continuous_dist"]
        continuous_action_dim = eval_envs[0].max_arg_len
        discrete_action_dim = eval_envs[0].num_primitives
        if variant["env_kwargs"]["fixed_schema"]:
            discrete_action_dim = 0
        else:
            if not discrete_continuous_dist:
                continuous_action_dim = continuous_action_dim + discrete_action_dim
                discrete_action_dim = 0
        action_dim = continuous_action_dim + discrete_action_dim
        use_batch_length = False
    world_model_class = WorldModel
    obs_dim = expl_env.observation_space.low.size
    if actor_model_class_name == "conditional_actor_model":
        actor_model_class = ConditionalActorModel
    elif actor_model_class_name == "continuous_conditional_actor_model":
        actor_model_class = ConditionalContinuousActorModel
    elif actor_model_class_name == "actor_model":
        actor_model_class = ActorModel
    if variant.get("load_from_path", False):
        filename = variant["models_path"] + variant["pkl_file_name"]
        print("loaded models from: %s" % filename)
        data = torch.load(filename)
        actor = data["trainer/actor"]
        vf = data["trainer/vf"]
        target_vf = data["trainer/target_vf"]
        world_model = data["trainer/world_model"]
    else:
        world_model = world_model_class(
            action_dim,
            image_shape=eval_envs[0].image_shape,
            **variant["model_kwargs"],
        )
    if variant.get("retrain_actor_and_vf", True):
        actor = actor_model_class(
            variant["model_kwargs"]["model_hidden_size"],
            world_model.feature_size,
            hidden_activation=torch.nn.functional.elu,
            discrete_action_dim=discrete_action_dim,
            continuous_action_dim=continuous_action_dim,
            env=eval_envs[0],
            **variant["actor_kwargs"],
        )
        vf = Mlp(
            hidden_sizes=[variant["model_kwargs"]["model_hidden_size"]]
            * variant["vf_kwargs"]["num_layers"],
            output_size=1,
            input_size=world_model.feature_size,
            hidden_activation=torch.nn.functional.elu,
        )
        target_vf = Mlp(
            hidden_sizes=[variant["model_kwargs"]["model_hidden_size"]]
            * variant["vf_kwargs"]["num_layers"],
            output_size=1,
            input_size=world_model.feature_size,
            hidden_activation=torch.nn.functional.elu,
        )

    expl_policy = DreamerPolicy(
        world_model,
        actor,
        obs_dim,
        action_dim,
        exploration=True,
        expl_amount=variant.get("expl_amount", 0.3),
        discrete_action_dim=discrete_action_dim,
        continuous_action_dim=continuous_action_dim,
        discrete_continuous_dist=discrete_continuous_dist,
    )
    eval_policy = DreamerPolicy(
        world_model,
        actor,
        obs_dim,
        action_dim,
        exploration=False,
        expl_amount=0.0,
        discrete_action_dim=discrete_action_dim,
        continuous_action_dim=continuous_action_dim,
        discrete_continuous_dist=discrete_continuous_dist,
    )

    rand_policy = ActionSpaceSamplePolicy(expl_env)

    expl_path_collector = VecMdpPathCollector(
        expl_env,
        expl_policy,
        save_env_in_snapshot=False,
    )

    eval_path_collector = VecMdpPathCollector(
        eval_env,
        eval_policy,
        save_env_in_snapshot=False,
    )

    replay_buffer = EpisodeReplayBuffer(
        variant["replay_buffer_size"],
        expl_env,
        variant["algorithm_kwargs"]["max_path_length"] + 1,
        obs_dim,
        action_dim,
        replace=False,
        use_batch_length=use_batch_length,
        batch_length=50,
    )
    if variant.get("models_path", None):
        replay_buffer = load_replay_buffer(replay_buffer, variant.get("replay_path"))
        variant["algorithm_kwargs"]["min_num_steps_before_training"] = 0
        print("loaded replay buffer")
    trainer_class_name = variant.get("algorithm", "DreamerV2")
    if trainer_class_name == "DreamerV2":
        trainer_class = DreamerV2Trainer
    else:
        trainer_class = DreamerTrainer
    trainer = trainer_class(
        eval_env,
        actor,
        vf,
        target_vf,
        world_model,
        eval_envs[0].image_shape,
        **variant["trainer_kwargs"],
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        pretrain_policy=rand_policy,
        **variant["algorithm_kwargs"],
    )

    if variant.get("save_video", False):
        algorithm.post_epoch_funcs.append(video_post_epoch_func)
    print("TRAINING")
    algorithm.to(ptu.device)
    algorithm.train()
    if variant.get("save_video", False):
        video_post_epoch_func(algorithm, -1)


def save_replay_buffer(replay_buffer, path):
    import h5py
    import numpy as np

    with h5py.File(path + "/replay_buffer.h5", "w") as hf:
        hf.create_dataset(
            "obs",
            data=replay_buffer._observations,
            compression="gzip",
            compression_opts=9,
        )
        hf.create_dataset(
            "actions",
            data=replay_buffer._actions,
            compression="gzip",
            compression_opts=9,
        )
        hf.create_dataset(
            "rewards",
            data=replay_buffer._rewards,
            compression="gzip",
            compression_opts=9,
        )
        hf.create_dataset(
            "terminals",
            data=replay_buffer._terminals,
            compression="gzip",
            compression_opts=9,
        )
        hf.create_dataset(
            "size",
            data=np.array([replay_buffer._size]),
            compression="gzip",
            compression_opts=9,
        )
        hf.create_dataset(
            "top",
            data=np.array([replay_buffer._top]),
            compression="gzip",
            compression_opts=9,
        )
        hf.close()


def load_replay_buffer(replay_buffer, path):
    import h5py
    import numpy as np

    with h5py.File(path + "/replay_buffer.h5", "r") as hf:
        obs = np.array(hf.get("obs"))
        actions = np.array(hf.get("actions"))
        rewards = np.array(hf.get("rewards"))
        terminals = np.array(hf.get("terminals"))
        size = np.array(hf.get("size"))[0]
        top = np.array(hf.get("top"))[0]
    replay_buffer._observations = obs
    replay_buffer._actions = actions
    replay_buffer._rewards = rewards
    replay_buffer._terminals = terminals
    replay_buffer._top = top
    replay_buffer._size = size
    return replay_buffer
