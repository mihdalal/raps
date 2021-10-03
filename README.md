# Robot Action Primitives (RAPS)

This repository is the official implementation of Accelerating Robotic Reinforcement Learning via Parameterized Action Primitives (RAPS).

#### [[Project Website]](https://mihdalal.github.io/raps/)

[Murtaza Dalal](https://mihdalal.github.io/), [Deepak Pathak*](https://www.cs.cmu.edu/~dpathak/), [Ruslan Salakhutdinov*](https://www.cs.cmu.edu/~rsalakhu/)<br/>
(&#42; equal advising)

CMU

![alt text](readme_files/raps.png)

If you find this work useful in your research, please cite:
```
@inproceedings{dalal2021raps,
    Author = {Dalal, Murtaza and Pathak, Deepak and
              Salakhutdinov, Ruslan},
    Title = {Accelerating Robotic Reinforcement Learning via Parameterized Action Primitives},
    Booktitle = {NeurIPS},
    Year = {2021}
}
```
## Requirements
To install dependencies, please run the following commands:
```
sudo apt-get update
sudo apt-get install curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    unzip \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev
sudo apt-get install libglfw3-dev libgles2-mesa-dev patchelf
sudo mkdir /usr/lib/nvidia-000
```

Please add the following to your bashrc:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin
export MUJOCO_GL='egl'
export MKL_THREADING_LAYER=GNU
export D4RL_SUPPRESS_IMPORT_ERROR='1'
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000
```

To install python requirements:

```
./setup_python_env.sh <absolute path to raps>
```

## Training and Evaluation

### Kitchen

Prior to running any experiments, *make sure* to run `cd /path/to/raps/rlkit`

single task env names:
* microwave
* kettle
* slide_cabinet
* hinge_cabinet
* light_switch
* top_left_burner

multi task env names:
* microwave_kettle_light_top_left_burner //Sequential Multi Task 1
* hinge_slide_bottom_left_burner_light //Sequential Multi Task 2

To train RAPS with Dreamer on any single task kitchen environment, run:
```train
python experiments/kitchen/dreamer/dreamer_v2_single_task_primitives.py --mode here_no_doodad --exp_prefix <> --env <env name>
```

To train RAPS with Dreamer on the multi task kitchen environments, run:
```train
python experiments/kitchen/dreamer/dreamer_v2_multi_task_primitives.py --mode here_no_doodad --exp_prefix <> --env <env name>
```

To train Raw Actions with Dreamer on any kitchen environment
```train
python experiments/kitchen/dreamer/dreamer_v2_raw_actions.py --mode here_no_doodad --exp_prefix <> --env <env name>
```

To train RAPS with RAD on any single task kitchen environment
```train
python experiments/kitchen/rad/rad_single_task_primitives.py --mode here_no_doodad --exp_prefix <> --env <env name>
```

To train RAPS with RAD on any multi task kitchen environment
```train
python experiments/kitchen/rad/rad_multi_task_primitives.py --mode here_no_doodad --exp_prefix <> --env <env name>
```

To train Raw Actions with RAD on any kitchen environment
```train
python experiments/kitchen/rad/rad_raw_actions.py --mode here_no_doodad --exp_prefix <> --env <env name>
```

To train RAPS with PPO on any single task kitchen environment
```train
python experiments/kitchen/ppo/ppo_single_task_primitives.py --mode here_no_doodad --exp_prefix <> --env <env name>
```

To train RAPS with PPO on any multi task kitchen environment
```train
python experiments/kitchen/ppo/ppo_multi_task_primitives.py --mode here_no_doodad --exp_prefix <> --env <env name>
```

To train Raw Actions with PPO on any kitchen environment
```train
python experiments/kitchen/ppo/ppo_raw_actions.py --mode here_no_doodad --exp_prefix <> --env <env name>
```

### Metaworld
single task env names
* drawer-close-v2
* soccer-v2
* peg-unplug-side-v2
* sweep-into-v2
* assembly-v2
* disassemble-v2

To train RAPS with Dreamer on any metaworld environment
```train
python experiments/metaworld/dreamer/dreamer_v2_single_task_primitives.py --mode here_no_doodad --exp_prefix <> --env <env name>
```

To train Raw Actions with Dreamer on any metaworld environment
```train
python experiments/metaworld/dreamer/dreamer_v2_single_task_raw_actions.py --mode here_no_doodad --exp_prefix <> --env <env name>
```

### Robosuite
To train RAPS with Dreamer on an Robosuite Lift
```train
python experiments/robosuite/dreamer/dreamer_v2_single_task_primitives_lift.py --mode here_no_doodad --exp_prefix <>
```

To train Raw Actions with Dreamer on an Robosuite Lift
```train
python experiments/robosuite/dreamer/dreamer_v2_single_task_raw_actions_lift.py --mode here_no_doodad --exp_prefix <>
```

To train RAPS with Dreamer on an Robosuite Door
```train
python experiments/robosuite/dreamer/dreamer_v2_single_task_primitives_door.py --mode here_no_doodad --exp_prefix <>
```

To train Raw Actions with Dreamer on an Robosuite Door
```train
python experiments/robosuite/dreamer/dreamer_v2_single_task_raw_actions_door.py --mode here_no_doodad --exp_prefix <>
```

## Learning Curve visualization

```
cd /path/to/raps/rlkit
python ../viskit/viskit/frontend.py data/<exp_prefix> //open localhost:5000 to view
```
