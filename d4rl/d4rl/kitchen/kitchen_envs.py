"""Environments using kitchen and Franka robot."""
import numpy as np
from d4rl.kitchen.adept_envs.franka.kitchen_multitask_v0 import KitchenTaskRelaxV1

OBS_ELEMENT_INDICES = {
    "bottom left burner": np.array([11, 12]),
    "top left burner": np.array([15, 16]),
    "light switch": np.array([17, 18]),
    "slide cabinet": np.array([19]),
    "hinge cabinet": np.array([20, 21]),
    "microwave": np.array([22]),
    "kettle": np.array([23, 24, 25, 26, 27, 28, 29]),
}
OBS_ELEMENT_GOALS = {
    "bottom left burner": np.array([-0.88, -0.01]),
    "top left burner": np.array([-0.92, -0.01]),
    "light switch": np.array([-0.69, -0.05]),
    "slide cabinet": np.array([0.37]),
    "hinge cabinet": np.array([0.0, 1.45]),
    "microwave": np.array([-0.75]),
    "kettle": np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
}
BONUS_THRESH = 0.3


class KitchenBase(KitchenTaskRelaxV1):
    # A string of element names. The robot's task is then to modify each of
    # these elements appropriately.
    TASK_ELEMENTS = []
    REMOVE_TASKS_WHEN_COMPLETE = False
    TERMINATE_ON_TASK_COMPLETE = False

    def __init__(self, dense=True, **kwargs):
        self.tasks_to_complete = set(self.TASK_ELEMENTS)
        self.dense = dense
        super(KitchenBase, self).__init__(**kwargs)

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

    def _get_task_goal(self):
        new_goal = np.zeros_like(self.goal)
        for element in self.TASK_ELEMENTS:
            element_idx = OBS_ELEMENT_INDICES[element]
            element_goal = OBS_ELEMENT_GOALS[element]
            new_goal[element_idx] = element_goal
        return new_goal

    def reset_model(self):
        self.tasks_to_complete = set(self.TASK_ELEMENTS)
        self.episodic_cumulative_reward = 0
        return super(KitchenBase, self).reset_model()

    def _get_reward_n_score(self, obs_dict):
        reward_dict, score = super(KitchenBase, self)._get_reward_n_score(obs_dict)
        next_q_obs = obs_dict["qp"]
        next_obj_obs = obs_dict["obj_qp"]
        idx_offset = len(next_q_obs)
        completions = []
        dense = 0
        for element in self.tasks_to_complete:
            element_idx = OBS_ELEMENT_INDICES[element]
            distance = np.linalg.norm(
                next_obj_obs[..., element_idx - idx_offset] - OBS_ELEMENT_GOALS[element]
            )
            dense += -1 * distance  # reward must be negative distance for RL
            is_grasped = True
            if not self.initializing and self.use_grasp_rewards:
                if element == "slide cabinet":
                    is_grasped = False
                    for i in range(1, 6):
                        obj_pos = self.get_site_xpos("schandle{}".format(i))
                        left_pad = self.get_site_xpos("leftpad")
                        right_pad = self.get_site_xpos("rightpad")
                        within_sphere_left = np.linalg.norm(obj_pos - left_pad) < 0.07
                        within_sphere_right = np.linalg.norm(obj_pos - right_pad) < 0.07
                        right = right_pad[0] < obj_pos[0]
                        left = obj_pos[0] < left_pad[0]
                        if (
                            right
                            and left
                            and within_sphere_right
                            and within_sphere_left
                        ):
                            is_grasped = True
                if element == "top left burner":
                    is_grasped = False
                    obj_pos = self.get_site_xpos("tlbhandle")
                    left_pad = self.get_site_xpos("leftpad")
                    right_pad = self.get_site_xpos("rightpad")
                    within_sphere_left = np.linalg.norm(obj_pos - left_pad) < 0.035
                    within_sphere_right = np.linalg.norm(obj_pos - right_pad) < 0.04
                    right = right_pad[0] < obj_pos[0]
                    left = obj_pos[0] < left_pad[0]
                    if within_sphere_right and within_sphere_left and right and left:
                        is_grasped = True
                if element == "microwave":
                    is_grasped = False
                    for i in range(1, 6):
                        obj_pos = self.get_site_xpos("mchandle{}".format(i))
                        left_pad = self.get_site_xpos("leftpad")
                        right_pad = self.get_site_xpos("rightpad")
                        within_sphere_left = np.linalg.norm(obj_pos - left_pad) < 0.05
                        within_sphere_right = np.linalg.norm(obj_pos - right_pad) < 0.05
                        if (
                            right_pad[0] < obj_pos[0]
                            and obj_pos[0] < left_pad[0]
                            and within_sphere_right
                            and within_sphere_left
                        ):
                            is_grasped = True
                if element == "hinge cabinet":
                    is_grasped = False
                    for i in range(1, 6):
                        obj_pos = self.get_site_xpos("hchandle{}".format(i))
                        left_pad = self.get_site_xpos("leftpad")
                        right_pad = self.get_site_xpos("rightpad")
                        within_sphere_left = np.linalg.norm(obj_pos - left_pad) < 0.06
                        within_sphere_right = np.linalg.norm(obj_pos - right_pad) < 0.06
                        if (
                            right_pad[0] < obj_pos[0]
                            and obj_pos[0] < left_pad[0]
                            and within_sphere_right
                        ):
                            is_grasped = True
                if element == "light switch":
                    is_grasped = False
                    for i in range(1, 4):
                        obj_pos = self.get_site_xpos("lshandle{}".format(i))
                        left_pad = self.get_site_xpos("leftpad")
                        right_pad = self.get_site_xpos("rightpad")
                        within_sphere_left = np.linalg.norm(obj_pos - left_pad) < 0.045
                        within_sphere_right = np.linalg.norm(obj_pos - right_pad) < 0.03
                        if within_sphere_right and within_sphere_left:
                            is_grasped = True
            complete = distance < BONUS_THRESH and is_grasped
            if complete:
                completions.append(element)
        if self.REMOVE_TASKS_WHEN_COMPLETE:
            [self.tasks_to_complete.remove(element) for element in completions]
        bonus = float(len(completions))
        reward_dict["bonus"] = bonus
        reward_dict["r_total"] = bonus
        if self.dense:
            reward_dict["r_total"] = dense
        score = bonus
        return reward_dict, score

    def step(
        self,
        a,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        obs, reward, done, env_info = super(KitchenBase, self).step(
            a,
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )
        self.episodic_cumulative_reward += reward

        if self.TERMINATE_ON_TASK_COMPLETE:
            done = not self.tasks_to_complete
        self.update_info(env_info)
        return obs, reward, done, env_info

    def update_info(self, info):
        next_q_obs = self.obs_dict["qp"]
        next_obj_obs = self.obs_dict["obj_qp"]
        idx_offset = len(next_q_obs)
        if self.initializing:
            self.per_task_cumulative_reward = {
                k: 0.0 for k in OBS_ELEMENT_INDICES.keys()
            }
        for element in OBS_ELEMENT_INDICES.keys():
            element_idx = OBS_ELEMENT_INDICES[element]
            distance = np.linalg.norm(
                next_obj_obs[..., element_idx - idx_offset] - OBS_ELEMENT_GOALS[element]
            )
            info[element + " distance to goal"] = distance
            info[element + " success"] = float(distance < BONUS_THRESH)
            success = float(distance < BONUS_THRESH)
            self.per_task_cumulative_reward[element] += success
            info[element + " cumulative reward"] = self.per_task_cumulative_reward[
                element
            ]
            info[element + " success"] = success
            if len(self.TASK_ELEMENTS) == 1 and self.TASK_ELEMENTS[0] == element:
                info["success"] = success
        info["episodic cumulative reward"] = self.episodic_cumulative_reward
        return info


class KitchenMicrowaveKettleLightTopLeftBurnerV0(KitchenBase):
    TASK_ELEMENTS = ["microwave", "kettle", "light switch", "top left burner"]
    REMOVE_TASKS_WHEN_COMPLETE = True


class KitchenHingeSlideBottomLeftBurnerLightV0(KitchenBase):
    TASK_ELEMENTS = [
        "hinge cabinet",
        "slide cabinet",
        "bottom left burner",
        "light switch",
    ]
    REMOVE_TASKS_WHEN_COMPLETE = True


class KitchenMicrowaveV0(KitchenBase):
    TASK_ELEMENTS = ["microwave"]


class KitchenKettleV0(KitchenBase):
    TASK_ELEMENTS = ["kettle"]


class KitchenBottomLeftBurnerV0(KitchenBase):
    TASK_ELEMENTS = ["bottom left burner"]


class KitchenTopLeftBurnerV0(KitchenBase):
    TASK_ELEMENTS = ["top left burner"]


class KitchenSlideCabinetV0(KitchenBase):
    TASK_ELEMENTS = ["slide cabinet"]


class KitchenHingeCabinetV0(KitchenBase):
    TASK_ELEMENTS = ["hinge cabinet"]


class KitchenLightSwitchV0(KitchenBase):
    TASK_ELEMENTS = ["light switch"]
