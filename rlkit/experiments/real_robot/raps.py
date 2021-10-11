import time

import ipdb
import numpy as np
import rlkit.torch.pytorch_util as ptu
import torch
from gym.spaces import Box
from rlkit.torch.model_based.dreamer.actor_models import ActorModel
from rlkit.torch.model_based.dreamer.dreamer_policy import DreamerPolicy
from rlkit.torch.model_based.dreamer.world_models import WorldModel
from frankapy import FrankaArm
from ros_image import RealsenseROSCamera
import rospy


class FrankaEnv:
    """Image observations from camera.
    Take in an action that is a 6D xyzrollpitchyaw.
    Execute this on the robot.

    """

    def __init__(self, ep_length, use_robot=True):

        self.ep_length = ep_length
        self.use_robot = use_robot

        # Construct franka api here
        if self.use_robot:
            print(f"Constructing camera")
            rospy.init_node("robot")
            self.obs_cam = RealsenseROSCamera(camera_id=1)

            self.reward_cam = RealsenseROSCamera(camera_id=2)

            self.franka = FrankaArm(init_node=False)
            print(f"connected to arm")
            # Reset
            self.reset()

            # Construct camera

        self.action_space = Box(low=-1, high=1, dtype=np.float32, shape=(6,))
        self.observation_space = Box(
            low=0, high=255, dtype=np.uint8, shape=(64 * 64 * 3,)
        )

        # self.wkspace_total_low = np.array([ 0.6400, -0.1732, -0.0312])
        self.wkspace_total_low = np.array([0.3746, -0.2700, 0.1759])
        self.wkspace_total_high = np.array([0.6407, 0.2972, 0.4281])

    def get_image(self):

        if self.use_robot:
            img = self.obs_cam.get_image()[:, :, ::-1]
            color_img = img.transpose(2, 0, 1).flatten()
            return color_img
        else:
            time.sleep(1 / 30)
            return np.random.randint(low=0, high=255, size=64 * 64 * 3).astype(np.uint8)

    def step(self, action):

        """The action is a vector of size 3."""

        if self.use_robot:
            # scale action
            scale = 0.1
            action *= scale
            delta_xyz = action[:3]
            assert delta_xyz.shape == (3,)

            # Treat the action as a delta
            T_ee_world = self.franka.get_pose()
            T_ee_world.translation += delta_xyz
            self.franka.goto_pose(T_ee_world)
            print(delta_xyz)
        # Get the next observation
        obs = self.get_image()

        # Increment timestep, check if done
        self.action_timestep += 1
        done = self.action_timestep == self.ep_length

        reward = 0
        info = {}

        return obs, reward, done, info

    def reset(self):
        print(f"Calling reset")
        self.action_timestep = 0

        # Move the arm back to a reasonable start location
        if self.use_robot:
            print(f"Moving back to home")
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
            T_ee_world.translation += [0, 0, 0.2]
            self.franka.goto_pose(T_ee_world)
            self.franka.reset_pose()

        obs = self.get_image()
        return obs

    def set_ee_pose(self, ee_pos_desired):
        T_ee_world = self.franka.get_pose()
        T_ee_world.translation = ee_pos_desired
        self.franka.goto_pose(T_ee_world)

    def apply_workspace_limits(self, a):
        a = np.clip(a, self.wkspace_total_low, self.wkspace_total_high)
        return a


class FrankaPrimitivesEnv(FrankaEnv):
    def reset_action_space(
        self,
        control_mode="primitives",
        action_scale=100,
        max_path_length=200,
        go_to_pose_iterations=100,
    ):
        self.max_path_length = max_path_length
        self.action_scale = action_scale
        self.go_to_pose_iterations = go_to_pose_iterations

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

    def reward(
        self,
    ):
        return 0

    def step(self, action):

        if self.control_mode == "end_effector":
            return super().step(action)

        else:
            stats = self.act(action)

        reward = 0
        obs = self.get_image()
        self.action_timestep += 1
        done = self.action_timestep == self.ep_length
        info = {}

        return obs, reward, done, info

    @property
    def _eef_xpos(self):
        # returns a 6 element list.
        # first 3 are xyz in mm
        # last 3 are rpy in radians
        if self.use_robot:
            pose = self.franka.get_pose().translation
            return pose
        else:
            return np.zeros(3)

    def apply_raw_action(self, desired_ee, desired_grasp):
        desired_grasp = np.clip(0, 0.08)
        self.franka.goto_gripper(
            desired_grasp,
            grasp=True,
            speed=0.1,
            force=0.0,
            epsilon_inner=0.08,
            epsilon_outer=0.08,
            block=False,
            ignore_errors=True,
            skill_desc="GoToGripper",
        )
        self.set_ee_pose(desired_ee)

    def goto_pose(self, pose, grasp=True):
        total_reward, total_success = 0, 0

        # pose = self.apply_workspace_limits(pose)
        print("pre action error: ", pose - self._eef_xpos)

        delta = pose - self._eef_xpos
        T_ee_world = self.franka.get_pose()
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
        self.franka.goto_pose(
            T_ee_world,
            duration=3,
            force_thresholds=[1, 1, 1, 1, 1, 1],
            torque_thresholds=[1, 1, 1, 1, 1, 1, 1],
        )
        r = self.reward()
        total_reward += r

        print("post action error: ", pose - self._eef_xpos)
        print()

        return np.array((total_reward, total_success))

    def close_gripper(self, d):
        d = d * 0.08
        d = np.clip(d, 0, 0.08)
        current_gripper_position = self.franka.get_gripper_width()
        desired = max(current_gripper_position - d, 0)
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
        return (0, 0)

    def open_gripper(self, d):
        d = d * 0.08
        d = np.clip(d, 0, 0.08)
        current_gripper_position = self.franka.get_gripper_width()
        desired = min(current_gripper_position + d, 0.08)
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
        return (0, 0)

    def top_grasp(self, zd):
        z_down, d = zd
        stats = self.goto_pose(
            self._eef_xpos + np.array([0, 0, -np.abs(z_down)]), grasp=False
        )
        self.close_gripper(d)
        return stats

    def move_delta_ee_pose(self, pose):
        stats = self.goto_pose(self._eef_xpos + pose, grasp=True)
        return stats

    def lift(self, z_dist):
        z_dist = np.maximum(z_dist, 0.0)
        stats = (0, 0)
        if z_dist > 0:
            stats = self.goto_pose(
                self._eef_xpos + np.array([0.0, 0.0, z_dist]), grasp=True
            )
        return stats

    def drop(self, z_dist):
        z_dist = np.maximum(z_dist, 0.0)
        stats = (0, 0)
        if z_dist > 0:
            stats = self.goto_pose(
                self._eef_xpos + np.array([0.0, 0.0, -z_dist]), grasp=True
            )
        return stats

    def move_left(self, x_dist):
        x_dist = np.maximum(x_dist, 0.0)
        stats = (0, 0)
        if x_dist > 0:
            stats = self.goto_pose(
                self._eef_xpos + np.array([0, -x_dist, 0.0]), grasp=True
            )
        return stats

    def move_right(self, x_dist):
        x_dist = np.maximum(x_dist, 0.0)
        stats = (0, 0)
        if x_dist > 0:
            stats = self.goto_pose(
                self._eef_xpos + np.array([0, x_dist, 0.0]), grasp=True
            )
        return stats

    def move_forward(self, y_dist):
        y_dist = np.maximum(y_dist, 0.0)
        stats = (0, 0)
        if y_dist > 0:
            stats = self.goto_pose(
                self._eef_xpos + np.array([y_dist, 0, 0.0]), grasp=True
            )
        return stats

    def move_backward(self, y_dist):
        y_dist = np.maximum(y_dist, 0.0)
        stats = (0, 0)
        if y_dist > 0:
            stats = self.goto_pose(
                self._eef_xpos + np.array([-y_dist, 0, 0.0]), grasp=True
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
        print(primitive_name, primitive_action)
        stats = primitive(primitive_action)
        return stats

    def get_idx_from_primitive_name(self, primitive_name):
        for idx, pn in self.primitive_idx_to_name.items():
            if pn == primitive_name:
                return idx


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


def test_baseline(num_eps=10, actions_per_ep=500):

    env = FrankaEnv(ep_length=actions_per_ep)
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


def test_raps(num_eps=10, actions_per_ep=5):

    env = FrankaPrimitivesEnv(ep_length=actions_per_ep, use_robot=True)
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


def main():

    # test_baseline()
    test_raps()
    exit()


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    np.random.seed(1)
    main()
