# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch

from polymetis import RobotInterface
import numpy as np

if __name__ == "__main__":
    # Initialize robot interface
    robot = RobotInterface(
        ip_address="172.26.122.200",
    )

    # Reset
    # robot.go_home()

    ee_pos, ee_quat = robot.pose_ee()
    print(f"Current ee position: {ee_pos}")
    print(f"Current ee orientation: {ee_quat}  (xyzw)")
