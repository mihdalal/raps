""" Kitchen environment for long horizon manipulation """
#!/usr/bin/python
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import os

from gym.spaces.box import Box

try:
    from robosuite_vices.controllers.arm_controller import PositionController
except:
    pass

import cv2
import mujoco_py
import numpy as np
import quaternion
from dm_control.mujoco import engine
from gym import spaces

from d4rl.kitchen.adept_envs import robot_env

INIT_QPOS = np.array(
    [
        1.48388023e-01,
        -1.76848573e00,
        1.84390296e00,
        -2.47685760e00,
        2.60252026e-01,
        7.12533105e-01,
        1.59515394e00,
        4.79267505e-02,
        3.71350919e-02,
        -2.66279850e-04,
        -5.18043486e-05,
        3.12877220e-05,
        -4.51199853e-05,
        -3.90842156e-06,
        -4.22629655e-05,
        6.28065475e-05,
        4.04984708e-05,
        4.62730939e-04,
        -2.26906415e-04,
        -4.65501369e-04,
        -6.44129196e-03,
        -1.77048263e-03,
        1.08009684e-03,
        -2.69397440e-01,
        3.50383255e-01,
        1.61944683e00,
        1.00618764e00,
        4.06395120e-03,
        -6.62095997e-03,
        -2.68278933e-04,
    ]
)


class KitchenV0(robot_env.RobotEnv):

    CALIBRATION_PATHS = {
        "default": os.path.join(os.path.dirname(__file__), "robot/franka_config.xml")
    }
    # Converted to velocity actuation
    ROBOTS = {"robot": "d4rl.kitchen.adept_envs.franka.robot.franka_robot:Robot_VelAct"}
    EE_CTRL_MODEL = os.path.join(
        os.path.dirname(__file__), "../franka/assets/franka_kitchen_ee_ctrl.xml"
    )
    JOINT_POSITION_CTRL_MODEL = os.path.join(
        os.path.dirname(__file__),
        "../franka/assets/franka_kitchen_joint_position_ctrl.xml",
    )
    TORQUE_CTRL_MODEL = os.path.join(
        os.path.dirname(__file__),
        "../franka/assets/franka_kitchen_torque_ctrl.xml",
    )
    CTLR_MODES_DICT = dict(
        primitives=dict(
            model=EE_CTRL_MODEL,
            robot={
                "robot": "d4rl.kitchen.adept_envs.franka.robot.franka_robot:Robot_Unconstrained"
            },
        ),
        end_effector=dict(
            model=EE_CTRL_MODEL,
            robot={
                "robot": "d4rl.kitchen.adept_envs.franka.robot.franka_robot:Robot_Unconstrained"
            },
        ),
        torque=dict(
            model=TORQUE_CTRL_MODEL,
            robot={
                "robot": "d4rl.kitchen.adept_envs.franka.robot.franka_robot:Robot_Unconstrained"
            },
        ),
        joint_position=dict(
            model=JOINT_POSITION_CTRL_MODEL,
            robot={
                "robot": "d4rl.kitchen.adept_envs.franka.robot.franka_robot:Robot_PosAct"
            },
        ),
        joint_velocity=dict(
            model=JOINT_POSITION_CTRL_MODEL,
            robot={
                "robot": "d4rl.kitchen.adept_envs.franka.robot.franka_robot:Robot_VelAct"
            },
        ),
        vices=dict(
            model=JOINT_POSITION_CTRL_MODEL,
            robot={
                "robot": "d4rl.kitchen.adept_envs.franka.robot.franka_robot:Robot_Unconstrained"
            },
        ),
    )
    N_DOF_ROBOT = 9
    N_DOF_OBJECT = 21

    def __init__(
        self,
        robot_params={},
        frame_skip=40,
        image_obs=False,
        imwidth=64,
        imheight=64,
        action_scale=1,
        use_workspace_limits=True,
        control_mode="primitives",
        use_grasp_rewards=False,
    ):
        self.control_mode = control_mode
        self.MODEL = self.CTLR_MODES_DICT[self.control_mode]["model"]
        self.ROBOTS = self.CTLR_MODES_DICT[self.control_mode]["robot"]

        self.episodic_cumulative_reward = 0
        self.obs_dict = {}
        self.use_grasp_rewards = use_grasp_rewards
        self.robot_noise_ratio = 0.1  # 10% as per robot_config specs
        self.goal = np.zeros((30,))

        self.image_obs = image_obs
        self.imwidth = imwidth
        self.imheight = imheight
        self.action_scale = action_scale

        self.primitive_idx_to_name = {
            0: "angled_x_y_grasp",
            1: "move_delta_ee_pose",
            2: "rotate_about_y_axis",
            3: "lift",
            4: "drop",
            5: "move_left",
            6: "move_right",
            7: "move_forward",
            8: "move_backward",
            9: "open_gripper",
            10: "close_gripper",
            11: "rotate_about_x_axis",
        }
        self.primitive_name_to_func = dict(
            angled_x_y_grasp=self.angled_x_y_grasp,
            move_delta_ee_pose=self.move_delta_ee_pose,
            rotate_about_y_axis=self.rotate_about_y_axis,
            lift=self.lift,
            drop=self.drop,
            move_left=self.move_left,
            move_right=self.move_right,
            move_forward=self.move_forward,
            move_backward=self.move_backward,
            open_gripper=self.open_gripper,
            close_gripper=self.close_gripper,
            rotate_about_x_axis=self.rotate_about_x_axis,
        )
        self.primitive_name_to_action_idx = dict(
            angled_x_y_grasp=[0, 1, 2, 3],
            move_delta_ee_pose=[4, 5, 6],
            rotate_about_y_axis=7,
            lift=8,
            drop=9,
            move_left=10,
            move_right=11,
            move_forward=12,
            move_backward=13,
            rotate_about_x_axis=14,
            open_gripper=15,
            close_gripper=16,
        )
        self.max_arg_len = 17
        self.num_primitives = len(self.primitive_name_to_func)

        self.min_ee_pos = np.array([-0.9, 0, 1.5])
        self.max_ee_pos = np.array([0.7, 1.5, 3.25])
        self.use_workspace_limits = use_workspace_limits

        super().__init__(
            self.MODEL,
            robot=self.make_robot(
                n_jnt=self.N_DOF_ROBOT,  # root+robot_jnts
                n_obj=self.N_DOF_OBJECT,
                **robot_params
            ),
            frame_skip=frame_skip,
            camera_settings=dict(
                distance=2.2, lookat=[-0.2, 0.5, 2.0], azimuth=70, elevation=-35
            ),
        )

        if self.image_obs:
            self.imlength = imwidth * imheight
            self.imlength *= 3
            self.image_shape = (3, imheight, imwidth)

            self.observation_space = spaces.Box(
                0, 255, (self.imlength,), dtype=np.uint8
            )
        else:
            obs_upper = 8.0 * np.ones(self.obs_dim)
            obs_lower = -obs_upper
            self.observation_space = spaces.Box(obs_lower, obs_upper, dtype=np.float32)

        if self.control_mode in ["joint_position", "joint_velocity", "torque"]:
            self.act_mid = np.zeros(self.N_DOF_ROBOT)
            self.act_amp = 2.0 * np.ones(self.N_DOF_ROBOT)

            act_lower = -1 * np.ones((self.N_DOF_ROBOT,))
            act_upper = 1 * np.ones((self.N_DOF_ROBOT,))
            self.action_space = spaces.Box(act_lower, act_upper)
        elif self.control_mode == "end_effector":
            # 3 for xyz, 3 for rpy, 1 for gripper
            act_lower = -1 * np.ones((7,))
            act_upper = 1 * np.ones((7,))
            self.action_space = spaces.Box(act_lower, act_upper)
        elif self.control_mode == "vices":
            self.action_space = spaces.Box(-np.ones(10), np.ones(10))
            ctrl_ratio = 1.0
            control_range_pos = np.ones(3)
            kp_max = 10
            kp_max_abs_delta = 10
            kp_min = 0.1
            damping_max = 2
            damping_max_abs_delta = 1
            damping_min = 0.1
            use_delta_impedance = False
            initial_impedance_pos = 1
            initial_impedance_ori = 1
            initial_damping = 0.25
            control_freq = 1.0 * ctrl_ratio

            self.joint_index_vel = np.arange(7)
            self.controller = PositionController(
                control_range_pos,
                kp_max,
                kp_max_abs_delta,
                kp_min,
                damping_max,
                damping_max_abs_delta,
                damping_min,
                use_delta_impedance,
                initial_impedance_pos,
                initial_impedance_ori,
                initial_damping,
                control_freq=control_freq,
                interpolation="linear",
            )
            self.controller.update_model(
                self.sim,
                self.joint_index_vel,
                self.joint_index_vel,
                id_name="panda0_link7",
            )
        elif self.control_mode == "primitives":
            action_space_low = -self.action_scale * np.ones(self.max_arg_len)
            action_space_high = self.action_scale * np.ones(self.max_arg_len)
            act_lower_primitive = np.zeros(self.num_primitives)
            act_upper_primitive = np.ones(self.num_primitives)
            act_lower = np.concatenate((act_lower_primitive, action_space_low))
            act_upper = np.concatenate(
                (
                    act_upper_primitive,
                    action_space_high,
                )
            )
            self.action_space = Box(act_lower, act_upper, dtype=np.float32)

        if self.control_mode in ["primitives", "end_effector"]:
            self.reset_mocap_welds(self.sim)
            self.sim.forward()
            gripper_target = (
                np.array([-0.498, 0.005, -0.431 + 0.01]) + self.get_ee_pose()
            )
            gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
            self.set_mocap_pos("mocap", gripper_target)
            self.set_mocap_quat("mocap", gripper_rotation)
            for _ in range(10):
                self.sim.step()

        self.init_qpos = INIT_QPOS
        self.init_qvel = self.sim.model.key_qvel[0].copy()

    def get_ee_pose(self):
        return self.get_site_xpos("end_effector")

    def get_ee_quat(self):
        return self.sim.data.body_xquat[10]

    def rpy_to_quat(self, rpy):
        q = quaternion.from_euler_angles(rpy)
        return np.array([q.x, q.y, q.z, q.w])

    def quat_to_rpy(self, q):
        q = quaternion.quaternion(q[0], q[1], q[2], q[3])
        return quaternion.as_euler_angles(q)

    def convert_xyzw_to_wxyz(self, q):
        return np.array([q[3], q[0], q[1], q[2]])

    def get_site_xpos(self, name):
        id = self.sim.model.site_name2id(name)
        return self.sim.data.site_xpos[id]

    def get_mocap_pos(self, name):
        body_id = self.sim.model.body_name2id(name)
        mocap_id = self.sim.model.body_mocapid[body_id]
        return self.sim.data.mocap_pos[mocap_id]

    def set_mocap_pos(self, name, value):
        body_id = self.sim.model.body_name2id(name)
        mocap_id = self.sim.model.body_mocapid[body_id]
        self.sim.data.mocap_pos[mocap_id] = value

    def get_mocap_quat(self, name):
        body_id = self.sim.model.body_name2id(name)
        mocap_id = self.sim.model.body_mocapid[body_id]
        return self.sim.data.mocap_quat[mocap_id]

    def set_mocap_quat(self, name, value):
        body_id = self.sim.model.body_name2id(name)
        mocap_id = self.sim.model.body_mocapid[body_id]
        self.sim.data.mocap_quat[mocap_id] = value

    def get_idx_from_primitive_name(self, primitive_name):
        for idx, pn in self.primitive_idx_to_name.items():
            if pn == primitive_name:
                return idx

    def _get_reward_n_score(self, obs_dict):
        return 0

    def ctrl_set_action(self, action):
        self.data.ctrl[7] = action[-2]
        self.data.ctrl[8] = action[-1]

    def mocap_set_action(self, sim, action):
        if sim.model.nmocap > 0:
            action, _ = np.split(action, (sim.model.nmocap * 7,))
            action = action.reshape(sim.model.nmocap, 7)

            pos_delta = action[:, :3]
            quat_delta = action[:, 3:]
            self.reset_mocap2body_xpos(sim)
            sim.data.mocap_pos[:] = sim.data.mocap_pos + pos_delta
            sim.data.mocap_quat[:] = sim.data.mocap_quat + quat_delta

    def reset_mocap_welds(self, sim):
        if sim.model.nmocap > 0 and sim.model.eq_data is not None:
            for i in range(sim.model.eq_data.shape[0]):
                if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                    sim.model.eq_data[i, :] = np.array(
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
                    )
        sim.forward()

    def reset_mocap2body_xpos(self, sim):
        if (
            sim.model.eq_type is None
            or sim.model.eq_obj1id is None
            or sim.model.eq_obj2id is None
        ):
            return
        for eq_type, obj1_id, obj2_id in zip(
            sim.model.eq_type, sim.model.eq_obj1id, sim.model.eq_obj2id
        ):
            if eq_type != mujoco_py.const.EQ_WELD:
                continue

            mocap_id = sim.model.body_mocapid[obj1_id]
            if mocap_id != -1:
                body_idx = obj2_id
            else:
                mocap_id = sim.model.body_mocapid[obj2_id]
                body_idx = obj1_id

            assert mocap_id != -1
            sim.data.mocap_pos[mocap_id][:] = sim.data.body_xpos[body_idx]
            sim.data.mocap_quat[mocap_id][:] = sim.data.body_xquat[body_idx]

    def _set_action(self, action):
        assert action.shape == (9,)

        action = action.copy()
        pos_ctrl, rot_ctrl, gripper_ctrl = action[:3], action[3:7], action[7:9]

        if self.control_mode == "primitives":
            pos_ctrl *= 0.05
        elif self.control_mode == "end_effector":
            pos_ctrl *= 0.02
            rot_ctrl *= 0.05
        assert gripper_ctrl.shape == (2,)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        self.ctrl_set_action(action)
        self.mocap_set_action(self.sim, action)

    def call_render_every_step(self):
        if self.render_every_step:
            if self.render_mode == "rgb_array":
                self.img_array.append(
                    self.render(
                        self.render_mode,
                        self.render_im_shape[0],
                        self.render_im_shape[1],
                    )
                )
            else:
                self.render(
                    self.render_mode,
                    self.render_im_shape[0],
                    self.render_im_shape[1],
                )

    def close_gripper(self, d):
        d = np.abs(d) * 0.04
        for _ in range(200):
            self._set_action(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -d, -d]))
            self.sim.step()
            self.call_render_every_step()

    def open_gripper(
        self,
        d,
    ):
        d = np.abs(d) * 0.04
        for _ in range(200):
            self._set_action(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, d, d]))
            self.sim.step()
            self.call_render_every_step()

    def rotate_ee(self, rpy):
        gripper = self.sim.data.qpos[7:9]
        for _ in range(200):
            quat = self.rpy_to_quat(rpy)
            quat_delta = self.convert_xyzw_to_wxyz(quat) - self.get_ee_quat()
            self._set_action(
                np.array(
                    [
                        0.0,
                        0.0,
                        0.0,
                        quat_delta[0],
                        quat_delta[1],
                        quat_delta[2],
                        quat_delta[3],
                        gripper[0],
                        gripper[1],
                    ]
                )
            )
            self.sim.step()
            self.call_render_every_step()

    def goto_pose(self, pose):
        gripper = self.sim.data.qpos[7:9]
        for _ in range(300):
            if self.use_workspace_limits:
                pose = np.clip(pose, self.min_ee_pos, self.max_ee_pos)
            self.reset_mocap2body_xpos(self.sim)
            delta = pose - self.get_ee_pose()
            self._set_action(
                np.array(
                    [
                        delta[0],
                        delta[1],
                        delta[2],
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        gripper[0],
                        gripper[1],
                    ]
                )
            )
            self.sim.step()
            self.call_render_every_step()

    def rotate_about_x_axis(self, angle):
        rotation = self.quat_to_rpy(self.get_ee_quat()) - np.array([angle, 0, 0])
        self.rotate_ee(rotation)

    def angled_x_y_grasp(self, angle_and_xyd):
        angle, x_dist, y_dist, d_dist = angle_and_xyd
        angle = np.clip(angle, -np.pi, np.pi)
        rotation = self.quat_to_rpy(self.get_ee_quat()) - np.array([angle, 0, 0])
        self.rotate_ee(rotation)
        self.goto_pose(self.get_ee_pose() + np.array([x_dist, 0.0, 0]))
        self.goto_pose(self.get_ee_pose() + np.array([0.0, y_dist, 0]))
        self.close_gripper(d_dist)

    def move_delta_ee_pose(self, pose):
        self.goto_pose(self.get_ee_pose() + pose)

    def rotate_about_y_axis(self, angle):
        angle = np.clip(angle, -np.pi, np.pi)
        rotation = self.quat_to_rpy(self.get_ee_quat()) - np.array([0, 0, angle])
        self.rotate_ee(rotation)

    def lift(self, z_dist):
        z_dist = np.maximum(z_dist, 0.0)
        self.goto_pose(self.get_ee_pose() + np.array([0.0, 0.0, z_dist]))

    def drop(self, z_dist):
        z_dist = np.maximum(z_dist, 0.0)
        self.goto_pose(self.get_ee_pose() + np.array([0.0, 0.0, -z_dist]))

    def move_left(self, x_dist):
        x_dist = np.maximum(x_dist, 0.0)
        self.goto_pose(self.get_ee_pose() + np.array([-x_dist, 0.0, 0.0]))

    def move_right(self, x_dist):
        x_dist = np.maximum(x_dist, 0.0)
        self.goto_pose(self.get_ee_pose() + np.array([x_dist, 0.0, 0.0]))

    def move_forward(self, y_dist):
        y_dist = np.maximum(y_dist, 0.0)
        self.goto_pose(self.get_ee_pose() + np.array([0.0, y_dist, 0.0]))

    def move_backward(self, y_dist):
        y_dist = np.maximum(y_dist, 0.0)
        self.goto_pose(self.get_ee_pose() + np.array([0.0, -y_dist, 0.0]))

    def break_apart_action(self, a):
        broken_a = {}
        for k, v in self.primitive_name_to_action_idx.items():
            broken_a[k] = a[v]
        return broken_a

    def act(self, a):
        if not self.initializing:
            a = a * self.action_scale
            a = np.clip(a, self.action_space.low, self.action_space.high)
        primitive_idx, primitive_args = (
            np.argmax(a[: self.num_primitives]),
            a[self.num_primitives :],
        )
        primitive_name = self.primitive_idx_to_name[primitive_idx]
        if primitive_name != "no_op":
            primitive_name_to_action_dict = self.break_apart_action(primitive_args)
            primitive_action = primitive_name_to_action_dict[primitive_name]
            primitive = self.primitive_name_to_func[primitive_name]
            primitive(
                primitive_action,
            )

    def update(self):
        self.controller.update_mass_matrix(self.sim, self.joint_index_vel)
        self.controller.update_model(
            self.sim, self.joint_index_pos, self.joint_index_vel
        )

    def set_render_every_step(
        self,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        self.render_every_step = render_every_step
        self.render_mode = render_mode
        self.render_im_shape = render_im_shape

    def unset_render_every_step(self):
        self.render_every_step = False

    def step(
        self,
        a,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        self.set_render_every_step(render_every_step, render_mode, render_im_shape)
        if not self.initializing:
            if self.control_mode in [
                "joint_position",
                "joint_velocity",
                "torque",
                "end_effector",
                "vices",
            ]:
                a = np.clip(a, -1.0, 1.0)
                if self.control_mode == "end_effector":
                    rotation = self.quat_to_rpy(self.get_ee_quat()) - np.array(a[3:6])
                    target_pos = a[:3] + self.get_ee_pose()
                    target_pos = np.clip(target_pos, self.min_ee_pos, self.max_ee_pos)
                    a[:3] = target_pos - self.get_ee_pose()
                    for _ in range(32):
                        quat = self.rpy_to_quat(rotation)
                        quat_delta = (
                            self.convert_xyzw_to_wxyz(quat) - self.get_ee_quat()
                        )
                        self._set_action(
                            np.concatenate([a[:3], quat_delta, [a[-1], -a[-1]]])
                        )
                        self.sim.step()
                elif self.control_mode == "vices":
                    action = a
                    for i in range(int(self.controller.interpolation_steps)):
                        self.controller.update_model(
                            self.sim,
                            self.joint_index_vel,
                            self.joint_index_vel,
                            id_name="panda0_link7",
                        )
                        a = self.controller.action_to_torques(action[:-1], i == 0)
                        act = np.zeros(9)
                        act[-1] = -action[-1]
                        act[-2] = action[-1]
                        act[:7] = a
                        self.do_simulation(act, n_frames=1)
                else:
                    if self.control_mode == "joint_velocity":
                        a = self.act_mid + a * self.act_amp  # mean center and scale
                    self.robot.step(
                        self, a, step_duration=self.skip * self.model.opt.timestep
                    )
            else:
                if render_every_step and render_mode == "rgb_array":
                    self.img_array = []
                self.act(a)
        obs = self._get_obs()

        # rewards
        reward_dict, score = self._get_reward_n_score(self.obs_dict)

        # termination
        done = False

        # finalize step
        env_info = {
            "time": self.obs_dict["t"],
            "score": score,
        }
        self.unset_render_every_step()
        return obs, reward_dict["r_total"], done, env_info

    def _get_obs(self):
        t, qp, qv, obj_qp, obj_qv = self.robot.get_obs(
            self, robot_noise_ratio=self.robot_noise_ratio
        )

        self.obs_dict = {}
        self.obs_dict["t"] = t
        self.obs_dict["qp"] = qp
        self.obs_dict["qv"] = qv
        self.obs_dict["obj_qp"] = obj_qp
        self.obs_dict["obj_qv"] = obj_qv
        self.obs_dict["goal"] = self.goal
        if self.image_obs:
            img = self.render(mode="rgb_array")
            img = img.transpose(2, 0, 1).flatten()
            return img
        else:
            return np.concatenate(
                [self.obs_dict["qp"], self.obs_dict["obj_qp"], self.obs_dict["goal"]]
            )

    def reset_model(self):
        reset_pos = self.init_qpos[:].copy()
        reset_vel = self.init_qvel[:].copy()
        self.robot.reset(self, reset_pos, reset_vel)
        self.sim.forward()
        if self.control_mode in ["primitives", "end_effector"]:
            self.reset_mocap2body_xpos(self.sim)

        self.goal = self._get_task_goal()  # sample a new goal on reset

        if self.sim_robot._use_dm_backend:
            imwidth = self.imwidth
            imheight = self.imheight
            camera = engine.MovableCamera(self.sim, imwidth, imheight)
            camera.set_pose(
                distance=2.2, lookat=[-0.2, 0.5, 2.0], azimuth=70, elevation=-35
            )
            self.start_img = camera.render()
        else:
            self.start_img = self.sim_robot.renderer.render_offscreen(
                self.imwidth,
                self.imheight,
            )
        return self._get_obs()

    def evaluate_success(self, paths):
        # score
        mean_score_per_rollout = np.zeros(shape=len(paths))
        for idx, path in enumerate(paths):
            mean_score_per_rollout[idx] = np.mean(path["env_infos"]["score"])
        mean_score = np.mean(mean_score_per_rollout)

        # success percentage
        num_success = 0
        num_paths = len(paths)
        for path in paths:
            num_success += bool(path["env_infos"]["rewards"]["bonus"][-1])
        success_percentage = num_success * 100.0 / num_paths

        # fuse results
        return np.sign(mean_score) * (
            1e6 * round(success_percentage, 2) + abs(mean_score)
        )

    def close(self):
        self.robot.close()

    def set_goal(self, goal):
        self.goal = goal

    def _get_task_goal(self):
        return self.goal

    # Only include goal
    @property
    def goal_space(self):
        len_obs = self.observation_space.low.shape[0]
        env_lim = np.abs(self.observation_space.low[0])
        return spaces.Box(
            low=-env_lim, high=env_lim, shape=(len_obs // 2,), dtype=np.float32
        )

    def get_env_state(self):
        joint_state = self.sim.get_state()
        mocap_state = self.data.mocap_pos, self.data.mocap_quat
        state = joint_state, mocap_state
        return copy.deepcopy(state)

    def set_env_state(self, state):
        joint_state, mocap_state = state
        self.sim.set_state(joint_state)
        mocap_pos, mocap_quat = mocap_state
        self.set_mocap_pos("mocap", mocap_pos)
        self.set_mocap_quat("mocap", mocap_quat)
        self.sim.forward()


class KitchenTaskRelaxV1(KitchenV0):
    """Kitchen environment with proper camera and goal setup"""

    def __init__(self, **kwargs):
        super(KitchenTaskRelaxV1, self).__init__(**kwargs)

    def _get_reward_n_score(self, obs_dict):
        reward_dict = {}
        reward_dict["true_reward"] = 0.0
        reward_dict["bonus"] = 0.0
        reward_dict["r_total"] = 0.0
        score = 0.0
        return reward_dict, score

    def render(self, mode="human", imwidth=None, imheight=None):
        if not imwidth:
            imwidth = self.imwidth
        if not imheight:
            imheight = self.imheight
        if mode == "rgb_array":
            if self.sim_robot._use_dm_backend:
                camera = engine.MovableCamera(self.sim, imwidth, imheight)
                camera.set_pose(
                    distance=2.2, lookat=[-0.2, 0.5, 2.0], azimuth=70, elevation=-35
                )
                img = camera.render()
            else:
                img = self.sim_robot.renderer.render_offscreen(
                    imwidth,
                    imheight,
                )
            return img
        else:
            super(KitchenTaskRelaxV1, self).render(mode=mode)
