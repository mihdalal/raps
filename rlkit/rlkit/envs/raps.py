import time

import ipdb
import numpy as np
import rlkit.torch.pytorch_util as ptu
import torch
from gym.spaces import Box
from rlkit.torch.model_based.dreamer.actor_models import ActorModel
from rlkit.torch.model_based.dreamer.dreamer_policy import DreamerPolicy
from rlkit.torch.model_based.dreamer.world_models import WorldModel
from frankapy import FrankaConstants as FC
from frankapy import FrankaArm
from rlkit.envs.ros_image import RealsenseROSCamera
import rospy
import cv2
import gym


class FrankaEnv:
    """Image observations from camera.
    Take in an action that is a 6D xyzrollpitchyaw.
    Execute this on the robot.

    """

    def __init__(self):
        print(f"Connecting to arm")
        try:
            self.franka = FrankaArm()
        except:
            import ipdb

            ipdb.set_trace()
            self.franka = FrankaArm()

        print(f"Constructing cameras")
        self.obs_cam = RealsenseROSCamera(camera_id=1)

        self.reward_cam = RealsenseROSCamera(camera_id=2)

        self.action_space = Box(low=-1, high=1, dtype=np.float32, shape=(6,))
        self.observation_space = Box(
            low=0, high=255, dtype=np.uint8, shape=(64 * 64 * 3,)
        )
        self.image_shape = (3, 64, 64)
        self.wkspace_total_low = np.array([0.35, -0.2, 0.1])
        self.wkspace_total_high = np.array([0.65, 0.23, 0.2])
        self.reward_range = (0, 1)
        self.metadata = {}

    def get_image(self):
        img = self.obs_cam.get_image()[:, :, ::-1]
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        color_img = img.transpose(2, 0, 1).flatten()
        return color_img

    def step(
        self,
        action,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):

        """The action is a vector of size 3."""

        # scale action
        scale = 0.1
        action *= scale
        delta_xyz = action[:3]
        assert delta_xyz.shape == (3,)

        # Treat the action as a delta
        T_ee_world = self.franka.get_pose()
        T_ee_world.translation += delta_xyz
        try:
            self.franka.goto_pose(
                T_ee_world,
                duration=0.1,
                force_thresholds=[15, 15, 15, 100, 100, 100],
                torque_thresholds=np.ones(7).tolist(),
                block=True,
                ignore_virtual_walls=True,
                use_impedance=False,
            )
        except:
            import ipdb

            ipdb.set_trace()
            self.franka.goto_pose(
                T_ee_world,
                duration=0.1,
                force_thresholds=[15, 15, 15, 100, 100, 100],
                torque_thresholds=np.ones(7).tolist(),
                block=True,
                ignore_virtual_walls=True,
                use_impedance=False,
            )
        # Get the next observation
        obs = self.get_image()

        # Increment timestep, check if done
        self.action_timestep += 1
        done = self.action_timestep == self.ep_length

        reward = 0
        info = {}

        return obs, reward, done, info

    def reset(self):
        # print(f"Calling reset")
        self.action_timestep = 0

        # Move the arm back to a reasonable start location
        # print(f"Moving back to home")
        try:
            self.franka.goto_gripper(
                0.08,
                grasp=False,
                speed=0.1,
                force=0.0,
                epsilon_inner=0.08,
                epsilon_outer=0.08,
                block=False,
                ignore_errors=True,
                skill_desc="GoToGripper",
            )
            T_ee_world = self.franka.get_pose()
            T_ee_world.translation += [0, 0, 0.3]
            T_ee_world.translation = self.apply_workspace_limits(T_ee_world.translation)

            self.franka.goto_joints(
                [
                    0.79249497,
                    0.17514637,
                    -0.48099698,
                    -2.36368314,
                    0.18256195,
                    2.55480444,
                    0.96184119,
                ],
                duration=1.5,
                skill_desc="",
                block=True,
                ignore_errors=True,
            )
        except:
            import ipdb

            ipdb.set_trace()
            self.franka.goto_gripper(
                0.08,
                grasp=False,
                speed=0.1,
                force=0.0,
                epsilon_inner=0.08,
                epsilon_outer=0.08,
                block=False,
                ignore_errors=True,
                skill_desc="GoToGripper",
            )
            T_ee_world = self.franka.get_pose()
            T_ee_world.translation += [0, 0, 0.3]
            T_ee_world.translation = self.apply_workspace_limits(T_ee_world.translation)

            self.franka.goto_joints(
                [
                    0.79249497,
                    0.17514637,
                    -0.48099698,
                    -2.36368314,
                    0.18256195,
                    2.55480444,
                    0.96184119,
                ],
                duration=1.5,
                skill_desc="",
                block=True,
                ignore_errors=True,
            )

        obs = self.get_image()
        return obs

    def apply_workspace_limits(self, a):
        a = np.clip(a, self.wkspace_total_low, self.wkspace_total_high)
        return a


class FrankaPrimitivesEnv(FrankaEnv):
    def reset_action_space(
        self,
        control_mode="primitives",
        action_scale=1,
        max_path_length=5,
        hardcode_gripper_actions=True,
    ):
        self.max_path_length = max_path_length
        self.action_scale = action_scale
        self.hardcode_gripper_actions = hardcode_gripper_actions

        # primitives
        self.primitive_idx_to_name = {
            0: "move_delta_ee_pose",
            1: "top_grasp",
            2: "lift",
            3: "drop",
            4: "move_left",
            5: "move_right",
            6: "move_forward",
            7: "move_backward",
            8: "close_gripper",
            9: "open_gripper",
        }
        self.primitive_name_to_func = dict(
            move_delta_ee_pose=self.move_delta_ee_pose,
            top_grasp=self.top_grasp,
            lift=self.lift,
            drop=self.drop,
            move_left=self.move_left,
            move_right=self.move_right,
            move_forward=self.move_forward,
            move_backward=self.move_backward,
            open_gripper=self.open_gripper,
            close_gripper=self.close_gripper,
        )
        self.primitive_name_to_action_idx = dict(
            move_delta_ee_pose=[0, 1, 2],
            top_grasp=[3, 4],
            lift=5,
            drop=6,
            move_left=7,
            move_right=8,
            move_forward=9,
            move_backward=10,
            open_gripper=11,
            close_gripper=12,
        )
        self.max_arg_len = 13

        self.num_primitives = len(self.primitive_name_to_func)
        self.control_mode = control_mode

        if self.control_mode == "primitives":
            action_space_low = -1 * np.ones(self.max_arg_len)
            action_space_high = np.ones(self.max_arg_len)
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

    def reward(self):
        reward = 0
        return reward

    def step(
        self,
        action,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):

        if self.control_mode == "end_effector":
            return super().step(action)

        else:
            stats = self.act(action)

        reward = self.reward()
        obs = self.get_image()
        done = self.action_timestep == self.max_path_length
        self.action_timestep += 1
        info = {}

        return obs, reward, done, info

    @property
    def _eef_xpos(self):
        try:
            pose = self.franka.get_pose().translation
        except:
            import ipdb

            ipdb.set_trace()
            pose = self.franka.get_pose().translation
        return pose

    def goto_pose(self, pose, grasp=True):
        pose = self.apply_workspace_limits(pose)
        # print("pre action error: ", pose - self._eef_xpos)
        delta = pose - self._eef_xpos
        T_ee_world = self.franka.get_pose()
        try:
            if grasp:
                self.franka.goto_gripper(
                    0.0,
                    grasp=False,
                    speed=0.1,
                    force=0.0,
                    epsilon_inner=0.08,
                    epsilon_outer=0.08,
                    block=False,
                    ignore_errors=True,
                    skill_desc="GoToGripper",
                )
            else:
                self.franka.goto_gripper(
                    0.08,
                    grasp=False,
                    speed=0.1,
                    force=0.0,
                    epsilon_inner=0.08,
                    epsilon_outer=0.08,
                    block=False,
                    ignore_errors=True,
                    skill_desc="GoToGripper",
                )
            T_ee_world.translation += delta
            if np.linalg.norm(delta) > .15:
                duration = 1.5
                if np.linalg.norm(delta) > .3:
                    duration = 2
            else:
                duration = 1
            self.franka.goto_pose(
                T_ee_world,
                duration=duration,
                force_thresholds=[15, 15, 15, 100, 100, 100],
                torque_thresholds=np.ones(7).tolist(),
                block=True,
                ignore_virtual_walls=True,
                use_impedance=False,
            )
        except:
            import ipdb

            ipdb.set_trace()
            if grasp:
                self.franka.goto_gripper(
                    0.0,
                    grasp=False,
                    speed=0.1,
                    force=0.0,
                    epsilon_inner=0.08,
                    epsilon_outer=0.08,
                    block=False,
                    ignore_errors=True,
                    skill_desc="GoToGripper",
                )
            else:
                self.franka.goto_gripper(
                    0.08,
                    grasp=False,
                    speed=0.1,
                    force=0.0,
                    epsilon_inner=0.08,
                    epsilon_outer=0.08,
                    block=False,
                    ignore_errors=True,
                    skill_desc="GoToGripper",
                )
            T_ee_world.translation += delta
            if np.linalg.norm(delta) > .15:
                duration = 1.5
                if np.linalg.norm(delta) > .3:
                    duration = 2
            else:
                duration = 1
            self.franka.goto_pose(
                T_ee_world,
                duration=duration,
                force_thresholds=[15, 15, 15, 100, 100, 100],
                torque_thresholds=np.ones(7).tolist(),
                block=True,
                ignore_virtual_walls=True,
                use_impedance=False,
            )
        r = self.reward()

        # print("post action error: ", pose - self._eef_xpos)
        # print()

        return np.array((r, r))

    def close_gripper(self, d):
        if self.hardcode_gripper_actions:
            try:
                self.franka.goto_gripper(
                    .04,
                    grasp=False,
                    speed=0.1,
                    force=0.0,
                    epsilon_inner=0.08,
                    epsilon_outer=0.08,
                    block=True,
                    ignore_errors=True,
                    skill_desc="GoToGripper",
                )
            except:
                import ipdb

                ipdb.set_trace()
                self.franka.goto_gripper(
                    .04,
                    grasp=False,
                    speed=0.1,
                    force=0.0,
                    epsilon_inner=0.08,
                    epsilon_outer=0.08,
                    block=True,
                    ignore_errors=True,
                    skill_desc="GoToGripper",
                )
        else:
            d = d * 0.08
            d = np.abs(d)
            d = np.clip(d, 0, 0.08)
            current_gripper_position = self.franka.get_gripper_width()
            desired = max(current_gripper_position - d, 0)
            # print("pre action error: ", desired - current_gripper_position)
            try:
                self.franka.goto_gripper(
                    desired,
                    grasp=False,
                    speed=0.1,
                    force=0.0,
                    epsilon_inner=0.08,
                    epsilon_outer=0.08,
                    block=True,
                    ignore_errors=True,
                    skill_desc="GoToGripper",
                )
            except:
                import ipdb

                ipdb.set_trace()
                self.franka.goto_gripper(
                    desired,
                    grasp=False,
                    speed=0.1,
                    force=0.0,
                    epsilon_inner=0.08,
                    epsilon_outer=0.08,
                    block=True,
                    ignore_errors=True,
                    skill_desc="GoToGripper",
                )
        # print("post action error: ", desired - self.franka.get_gripper_width())
        return (self.reward(), self.reward())

    def open_gripper(self, d):
        if self.hardcode_gripper_actions:
            try:
                self.franka.goto_gripper(
                    .08,
                    grasp=False,
                    speed=0.1,
                    force=0.0,
                    epsilon_inner=0.08,
                    epsilon_outer=0.08,
                    block=True,
                    ignore_errors=True,
                    skill_desc="GoToGripper",
                )
            except:
                import ipdb

                ipdb.set_trace()
                self.franka.goto_gripper(
                    .08,
                    grasp=False,
                    speed=0.1,
                    force=0.0,
                    epsilon_inner=0.08,
                    epsilon_outer=0.08,
                    block=True,
                    ignore_errors=True,
                    skill_desc="GoToGripper",
                )
        else:
            d = d * 0.08
            d = np.abs(d)
            d = np.clip(d, 0, 0.08)
            current_gripper_position = self.franka.get_gripper_width()
            desired = min(current_gripper_position + d, 0.08)
            # print("pre action error: ", desired - current_gripper_position)
            try:
                self.franka.goto_gripper(
                    desired,
                    grasp=False,
                    speed=0.1,
                    force=0.0,
                    epsilon_inner=0.08,
                    epsilon_outer=0.08,
                    block=True,
                    ignore_errors=True,
                    skill_desc="GoToGripper",
                )
            except:
                import ipdb

                ipdb.set_trace()
                self.franka.goto_gripper(
                    desired,
                    grasp=False,
                    speed=0.1,
                    force=0.0,
                    epsilon_inner=0.08,
                    epsilon_outer=0.08,
                    block=True,
                    ignore_errors=True,
                    skill_desc="GoToGripper",
                )
        # print("post action error: ", desired - self.franka.get_gripper_width())
        return (self.reward(), self.reward())

    def top_grasp(self, zd):
        z_down, d = zd
        stats = self.goto_pose(
            self._eef_xpos + np.array([0, 0, -np.abs(z_down)]), grasp=False
        )
        self.close_gripper(d)
        return stats

    def move_delta_ee_pose(self, pose):
        stats = self.goto_pose(self._eef_xpos + pose, grasp=False)
        return stats

    def lift(self, z_dist):
        z_dist = np.abs(z_dist)
        stats = (0, 0)
        stats = self.goto_pose(
            self._eef_xpos + np.array([0.0, 0.0, z_dist]), grasp=False
        )
        return stats

    def drop(self, z_dist):
        z_dist = np.abs(z_dist)
        stats = (0, 0)
        stats = self.goto_pose(
            self._eef_xpos + np.array([0.0, 0.0, -z_dist]), grasp=False
        )
        return stats

    def move_left(self, x_dist):
        x_dist = np.abs(x_dist)
        stats = (0, 0)
        stats = self.goto_pose(
            self._eef_xpos + np.array([0, -x_dist, 0.0]), grasp=False
        )
        return stats

    def move_right(self, x_dist):
        x_dist = np.abs(x_dist)
        stats = (0, 0)
        stats = self.goto_pose(self._eef_xpos + np.array([0, x_dist, 0.0]), grasp=False)
        return stats

    def move_forward(self, y_dist):
        y_dist = np.abs(y_dist)
        stats = (0, 0)
        stats = self.goto_pose(self._eef_xpos + np.array([y_dist, 0, 0.0]), grasp=False)
        return stats

    def move_backward(self, y_dist):
        y_dist = np.abs(y_dist)
        stats = (0, 0)
        stats = self.goto_pose(
            self._eef_xpos + np.array([-y_dist, 0, 0.0]), grasp=False
        )
        return stats

    def break_apart_action(self, a):
        broken_a = {}
        for k, v in self.primitive_name_to_action_idx.items():
            broken_a[k] = a[v]
        return broken_a

    def act(self, a):
        a = np.clip(a, self.action_space.low, self.action_space.high)
        a = a * self.action_scale
        primitive_idx, primitive_args = (
            np.argmax(a[: self.num_primitives]),
            a[self.num_primitives :],
        )
        primitive_name = self.primitive_idx_to_name[primitive_idx]
        primitive_name_to_action_dict = self.break_apart_action(primitive_args)
        primitive_action = primitive_name_to_action_dict[primitive_name]
        primitive = self.primitive_name_to_func[primitive_name]
        # print(primitive_name, primitive_action)
        stats = primitive(primitive_action)
        return stats

    def get_idx_from_primitive_name(self, primitive_name):
        for idx, pn in self.primitive_idx_to_name.items():
            if pn == primitive_name:
                return idx


class DiceEnvWrapper(gym.Wrapper):
    def __init__(self, env, divider_xpos):
        gym.Wrapper.__init__(self, env)
        self.divider_xpos = divider_xpos

    def __getattr__(self, name):
        return getattr(self.env, name)

    def get_dice_center(self):
        img = self.reward_cam.get_image()[:, :, ::-1]
        purple = [168, 127, 214]
        higher = [255, 150, 255]
        lower = [80, 100, 100]
        lowerBound = np.array(lower)
        upperBound = np.array(higher)
        img = cv2.resize(img, (340, 220))
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(imgHSV, lowerBound, upperBound)
        kernelOpen = np.ones((5, 5))
        kernelClose = np.ones((20, 20))

        maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
        maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)

        maskFinal = maskClose
        conts, h = cv2.findContours(
            maskFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        largest_area = 0
        for i in range(len(conts)):
            x, y, w, h = cv2.boundingRect(conts[i])
            largest_area = max(largest_area, w * h)
        for i in range(len(conts)):
            x, y, w, h = cv2.boundingRect(conts[i])
            if w * h == largest_area:
                return x + w / 2
        return 0

    def reset(self):
        obs = self.env.reset()
        self.reference_dice_center = self.get_dice_center()
        return obs

    def step(
        self,
        action,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        o, r, d, i = self.env.step(action)
        i["reference dice center"] = self.reference_dice_center
        if d:
            old_dice_center = self.reference_dice_center
            self.reset()
            # check if dice center switched sides
            r = float(
                (old_dice_center > self.divider_xpos)
                != (self.reference_dice_center > self.divider_xpos)
            )
            print("SUCCESS", r > 0.0)
            import ipdb

            ipdb.set_trace()  # reset the dice

        dice_center = self.get_dice_center()
        i["dice center"] = dice_center
        # print("dice center: ", dice_center)
        return o, r, d, i


def make_policy(env, use_raw_actions=True):
    if use_raw_actions:
        discrete_continuous_dist = False
        continuous_action_dim = env.action_space.low.size
        discrete_action_dim = 0
        action_dim = continuous_action_dim
        actor_kwargs = dict(
            init_std=0.0,
            num_layers=4,
            min_std=0.1,
            dist="trunc_normal",
        )
    else:
        discrete_continuous_dist = True
        continuous_action_dim = env.max_arg_len
        discrete_action_dim = env.num_primitives
        action_dim = continuous_action_dim + discrete_action_dim
        actor_kwargs = dict(
            init_std=0.0,
            num_layers=4,
            min_std=0.1,
            dist="tanh_normal_dreamer_v1",
            discrete_continuous_dist=discrete_continuous_dist,
        )

    model_kwargs = dict(
        model_hidden_size=400,
        stochastic_state_size=50,
        deterministic_state_size=200,
        embedding_size=1024,
        rssm_hidden_size=200,
        reward_num_layers=2,
        pred_discount_num_layers=3,
        gru_layer_norm=True,
        std_act="sigmoid2",
    )

    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    world_model = WorldModel(
        action_dim,
        image_shape=(3, 64, 64),
        env=env,
        **model_kwargs,
    ).to(ptu.device)

    actor = ActorModel(
        model_kwargs["model_hidden_size"],
        world_model.feature_size,
        env=env,
        hidden_activation=torch.nn.functional.elu,
        discrete_action_dim=discrete_action_dim,
        continuous_action_dim=continuous_action_dim,
        **actor_kwargs,
    ).to(ptu.device)

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

    return eval_policy


def run_rollout(env, policy):
    obs = env.reset()
    policy.reset(obs)
    done = False

    start = time.time()
    while not done:
        action, _ = policy.get_action(obs[None, ...])
        action = action[0]
        obs, reward, done, info = env.step(action)
    end = time.time()
    duration = end - start
    return duration


def test_lift(env):
    env.reset()
    env.drop(100)  # moves max 10cm


def test_baseline(num_eps=10):

    env = FrankaEnv()
    ptu.device = torch.device("cuda:0")
    policy = make_policy(env, use_raw_actions=True)

    total_time = 0
    ep_times = []
    for ep in range(num_eps):
        print(f"Episode {ep}")
        ep_time = run_rollout(env, policy)
        total_time += ep_time
        ep_times.append(ep_time)

    time_per_ep = np.mean(ep_times[1:])

    print(f"baseline time_per_ep {time_per_ep}")
    ipdb.set_trace()


def test_raps(
    num_eps=10,
):

    env = FrankaPrimitivesEnv()
    env.reset_action_space(
        control_mode="primitives",
        action_scale=1,
        max_path_length=5,
        go_to_pose_iterations=10,
    )
    ptu.device = torch.device("cuda:0")
    policy = make_policy(env, use_raw_actions=False)

    total_time = 0
    ep_times = []
    for ep in range(num_eps):
        print(f"Episode {ep}")
        ep_time = run_rollout(env, policy)
        print(f"ep time {ep_time}")
        total_time += ep_time
        ep_times.append(ep_time)

    time_per_ep = np.mean(ep_times[1:])

    print(f"RAPS time_per_ep {time_per_ep}")


def test_dice_raps(num_eps=10):
    env = FrankaPrimitivesEnv()
    env.reset_action_space(
        control_mode="primitives",
        action_scale=1,
        max_path_length=5,
    )
    env = DiceEnvWrapper(env, divider_xpos=175)
    ptu.device = torch.device("cuda:0")
    policy = make_policy(env, use_raw_actions=False)

    total_time = 0
    ep_times = []
    for ep in range(num_eps):
        print(f"Episode {ep}")
        ep_time = run_rollout(env, policy)
        print(f"ep time {ep_time}")
        total_time += ep_time
        ep_times.append(ep_time)

    time_per_ep = np.mean(ep_times[1:])

    print(f"RAPS time_per_ep {time_per_ep}")


def main():
    test_dice_raps()
    exit()


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    np.random.seed(1)
    main()
