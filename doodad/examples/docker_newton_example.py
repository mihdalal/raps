"""
Example script for using newton machines via docker + rllab
"""

import doodad as dd
import doodad.mount as mount
import doodad.ssh as ssh

MY_USERNAME = "justin"

# Use local mode to test code
mode_local = dd.mode.LocalDocker(image="justinfu/rl_base:0.1")

# Use docker mode to launch jobs on newton machine
mode_ssh = dd.mode.SSHDocker(
    image="justinfu/rl_base:0.1",
    credentials=ssh.SSHCredentials(
        hostname="newton2.banatao.berkeley.edu",
        username="rail",
        identity_file="path/to/identity",
    ),
)

# Set up code and output directories
OUTPUT_DIR = (
    "/mount/outputs"  # this is the directory visible to the target script inside docker
)
mounts = [
    mount.MountLocal(
        local_dir="~/install/rllab", pythonpath=True
    ),  # point to your rllab
    mount.MountLocal(
        local_dir="~/install/gym/.mujoco", mount_point="/root/.mujoco"
    ),  # point to your mujoco
    # this output directory will be visible on the remote machine
    # TODO: this directory will have root permissions. For now you need to scp your data inside your script.
    mount.MountLocal(
        local_dir="~/data/%s" % MY_USERNAME, mount_point=OUTPUT_DIR, output=True
    ),
]

pd.launch_python(
    target="path/to/script.py",  # point to a target script (absolute path).
    mode=mode_ssh,
    mount_points=mounts,
)
