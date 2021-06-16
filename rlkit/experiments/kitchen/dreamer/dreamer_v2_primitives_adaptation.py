import argparse
import json
import os
import random
import subprocess
import time

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.model_based.dreamer.experiments.experiment_utils import (
    preprocess_variant,
)
from rlkit.torch.model_based.dreamer.experiments.kitchen_dreamer import experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_prefix", type=str, default="test")
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--mode", type=str, default="local")
    parser.add_argument("--env", type=str, default="")
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    if args.debug:
        algorithm_kwargs = dict(
            num_epochs=5,
            num_eval_steps_per_epoch=10,
            num_expl_steps_per_train_loop=50,
            min_num_steps_before_training=10,
            num_pretrain_steps=10,
            num_train_loops_per_epoch=1,
            num_trains_per_train_loop=10,
            batch_size=30,
            max_path_length=5,
            use_pretrain_policy_for_initial_data=False,
        )
        exp_prefix = "test" + args.exp_prefix
    else:
        algorithm_kwargs = dict(
            num_epochs=25,
            num_eval_steps_per_epoch=30,
            min_num_steps_before_training=2500,
            num_pretrain_steps=100,
            max_path_length=5,
            batch_size=417,
            num_expl_steps_per_train_loop=30,
            num_train_loops_per_epoch=40,
            num_trains_per_train_loop=5,
            use_pretrain_policy_for_initial_data=False,
        )
        exp_prefix = args.exp_prefix
    variant = dict(
        algorithm="DreamerV2",
        version="normal",
        replay_buffer_size=int(5e5),
        algorithm_kwargs=algorithm_kwargs,
        env_class="hinge_cabinet",
        env_kwargs=dict(
            dense=False,
            image_obs=True,
            fixed_schema=False,
            action_scale=1.4,
            use_combined_action_space=True,
            proprioception=False,
            wrist_cam_concat_with_fixed_view=False,
            use_wrist_cam=False,
            normalize_proprioception_obs=True,
            use_workspace_limits=True,
        ),
        actor_kwargs=dict(
            discrete_continuous_dist=True,
            init_std=0.0,
            num_layers=4,
            min_std=0.1,
            dist="tanh_normal_dreamer_v1",
        ),
        vf_kwargs=dict(
            num_layers=3,
        ),
        model_kwargs=dict(
            model_hidden_size=400,
            stochastic_state_size=50,
            deterministic_state_size=200,
            embedding_size=1024,
            rssm_hidden_size=200,
            reward_num_layers=2,
            pred_discount_num_layers=3,
            gru_layer_norm=True,
            std_act="sigmoid2",
        ),
        trainer_kwargs=dict(
            adam_eps=1e-5,
            discount=0.99,
            lam=0.95,
            forward_kl=False,
            free_nats=1.0,
            pred_discount_loss_scale=10.0,
            kl_loss_scale=0.0,
            transition_loss_scale=0.8,
            actor_lr=8e-5,
            vf_lr=8e-5,
            world_model_lr=3e-4,
            reward_loss_scale=2.0,
            use_pred_discount=True,
            policy_gradient_loss_scale=1.0,
            actor_entropy_loss_schedule="1e-4",
            target_update_period=100,
            detach_rewards=False,
            imagination_horizon=5,
        ),
        num_expl_envs=5,
        num_eval_envs=1,
        expl_amount=0.3,
        load_from_path=True,
        models_path="/path/to/p2exp_experiment",
        retrain_actor_and_vf=False,
        pkl_file_name="/itr_400.pkl",
    )

    search_space = {
        "env_class": [
            "microwave",
            "kettle",
            "slide_cabinet",
            "top_left_burner",
            "hinge_cabinet",
            "light_switch",
        ],
        "trainer_kwargs.discount": [0.8],
        "retrain_actor_and_vf": [False],
        "num_actor_vf_pretrain_iters": [1000, 10000],
        "algorithm_kwargs.use_pretrain_policy_for_initial_data": [True],
        "algorithm_kwargs.num_pretrain_steps": [0],
        "algorithm_kwargs.num_trains_per_train_loop": [50],
        "trainer_kwargs.world_model_lr": [3e-4],
        "pkl_file_name": [
            "/itr_100.pkl",
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space,
        default_parameters=variant,
    )
    num_exps_launched = 0
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        variant = preprocess_variant(variant, args.debug)
        models_path = variant["models_path"]
        seed = int(json.load(open(models_path + "/variant.json", "r"))["seed"])
        variant["seed"] = seed
        variant["exp_id"] = exp_id
        run_experiment(
            experiment,
            exp_prefix=args.exp_prefix,
            mode=args.mode,
            variant=variant,
            use_gpu=True,
            snapshot_mode="none",
            python_cmd=subprocess.check_output("which python", shell=True).decode(
                "utf-8"
            )[:-1],
            seed=seed,
            exp_id=exp_id,
        )
        time.sleep(1)
        num_exps_launched += 1
    print("Num exps launched: ", num_exps_launched)
