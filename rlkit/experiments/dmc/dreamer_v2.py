import argparse
import random
import subprocess

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.model_based.dreamer.experiments.dmc_dreamer import experiment

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
            num_eval_steps_per_epoch=1000,
            min_num_steps_before_training=1000,
            num_pretrain_steps=1,
            max_path_length=500,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1,
            num_train_loops_per_epoch=1,
            batch_size=50,
        )
        exp_prefix = "test" + args.exp_prefix
    else:
        algorithm_kwargs = dict(
            num_epochs=50,
            num_eval_steps_per_epoch=2500,
            min_num_steps_before_training=5000,
            num_pretrain_steps=100,
            max_path_length=500,
            num_expl_steps_per_train_loop=500,
            num_trains_per_train_loop=100,
            num_train_loops_per_epoch=20,
            batch_size=50,
        )
        exp_prefix = args.exp_prefix
    variant = dict(
        algorithm="DreamerV2",
        version="normal",
        replay_buffer_size=int(1e4),
        algorithm_kwargs=algorithm_kwargs,
        actor_kwargs=dict(
            init_std=0.0,
            num_layers=4,
            min_std=0.1,
            dist="trunc_normal",
        ),
        vf_kwargs=dict(
            num_layers=3,
        ),
        model_kwargs=dict(
            model_hidden_size=400,
            stochastic_state_size=32,
            deterministic_state_size=200,
            embedding_size=32 * 48,
            rssm_hidden_size=200,
            reward_num_layers=2,
            pred_discount_num_layers=3,
            gru_layer_norm=True,
            std_act="sigmoid2",
            depth=48,
        ),
        trainer_kwargs=dict(
            adam_eps=1e-5,
            discount=0.99,
            lam=0.95,
            forward_kl=False,
            free_nats=1.0,
            kl_loss_scale=0.0,
            transition_loss_scale=0.8,
            actor_lr=8e-5,
            vf_lr=8e-5,
            world_model_lr=3e-4,
            reward_loss_scale=1.0,
            imagination_horizon=15,
            use_pred_discount=False,
            policy_gradient_loss_scale=0.0,
            actor_entropy_loss_schedule="1e-4",
            target_update_period=100,
        ),
        num_eval_envs=1,
        expl_amount=0.0,
    )

    search_space = {
        "env_id": [args.env],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space,
        default_parameters=variant,
    )
    num_exps_launched = 0
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(args.num_seeds):
            seed = random.randint(0, 100000)
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
            num_exps_launched += 1
    print("Num exps launched: ", num_exps_launched)
