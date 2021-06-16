import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description="RL")
    parser.add_argument(
        "--gail",
        action="store_true",
        default=False,
        help="do imitation learning with gail",
    )
    parser.add_argument(
        "--gail-experts-dir",
        default="./gail_experts",
        help="directory that contains expert demonstrations for gail",
    )
    parser.add_argument(
        "--gail-batch-size",
        type=int,
        default=128,
        help="gail batch size (default: 128)",
    )
    parser.add_argument(
        "--gail-epoch", type=int, default=5, help="gail epochs (default: 5)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.99,
        help="RMSprop optimizer apha (default: 0.99)",
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument(
        "--cuda-deterministic",
        action="store_true",
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=16,
        help="how many training CPU processes to use (default: 16)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=5,
        help="number of forward steps in A2C (default: 5)",
    )
    parser.add_argument(
        "--env-name",
        default="PongNoFrameskip-v4",
        help="environment to train on (default: PongNoFrameskip-v4)",
    )
    parser.add_argument(
        "--log-dir",
        default="/tmp/gym/",
        help="directory to save agent logs (default: /tmp/gym)",
    )
    parser.add_argument(
        "--save-dir",
        default="",
        help="directory to save agent logs (default: ./trained_models/)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args
