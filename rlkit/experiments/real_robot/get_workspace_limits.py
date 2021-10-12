from frankapy import FrankaArm


if __name__ == "__main__":
    franka = FrankaArm(init_node=True)
    T_ee_world = franka.get_pose()
    print(T_ee_world.translation)

# [ 0.48935749 -0.24988993  0.11593331]
# [0.19147676 0.30219993 0.42498111]

# low: [0.19147676, -0.24988993, 0.11593331]
# high: [0.48935749, 0.30219993, 0.42498111]
