# Easy Launch
Easy-launch is a wrapper around doodad that makes it easier to launch experiments across different platforms (EC2, BRC, etc.)
Usage:

```python
from doodad.easy_launch.python_function import run_experiment


def function(doodad_config, variant):
    print("The learning rate is", variant['learning_rate'])
    print("You are ", variant['parameter'])
    print("Save to ", doodad_config.base_log_dir)
    # save outputs (e.g. logs, parameter snapshots, etc.) to
    # doodad_config.base_log_dir

if __name__ == "__main__":
    variant = dict(
        learning_rate=1e-4,
        parameter='awesome',
    )
    run_experiment(
        function,
        exp_name='ec2-test-doodad-easy-launch',
        mode='ec2',
        variant=variant,
        use_gpue=False, # or True
    )
```

The function needs to take in two arguments:
 - doodad_config, which contains meta-data about the experiment
 - variant, which is a dictionary having variants (e.g. hyperparameters, logger config, etc.)

If you want, you can launch multiple jobs at once:
```
for variant in my_list_of_hyperparameters:
    run_experiment(
        function,
        exp_name='ec2-test-doodad-easy-launch',
        mode='ec2',
        variant=variant,
        use_gpue=False, # or True
    )
```

The supported modes are:
 - 'local': this will run the code locally on your machine.
 - 'here_no_doodad': this will run the code locally on your machine and bypass doodad completely (it just calls your function).
 - 'ec2': run your code on AWS EC2.
 - 'gcp': run your code on GCP.
 - 'htp': generate a taskfile and script for using BRC's high-throughput script
 - 'sss': generate a script to run on some slurm job using singularity

The last three are specific to slurm. If you're in the RAIL lab, you'll want to use `htp` and `sss` mode for BRC.

## Setup
To use it you well need to create a private version of the configuration file:
```bash
cp doodad/easy_launch/config.py doodad/easy_launch/config_private.py
```

There's a bunch of settings in the config file, but the main ones are `CODE_DIRS_TO_MOUNT` and `NON_CODE_DIRS_TO_MOUNT`. The first should be a list of directories containing python code.
 The later is a list of dictionaries, that describe how local directories containing data should be mounted on the final system that's running the code.

Here's an example of what mine are set to:
```
CODE_DIRS_TO_MOUNT = [
    '/home/vitchyr/git/railrl/',  # custom code repo
    '/home/vitchyr/git/multiworld/',  # custom environment repo
]
NON_CODE_DIRS_TO_MOUNT = [
    dict(
        local_dir='/home/vitchyr/.mujoco/',
        mount_point='/root/.mujoco',  # my Docker image expects for mujoco to be here.
    ),
]
```

After that, the settings are specific to the different mode you want to use

## BRC (sss and htp mode)
_Disclaimer: this workflow is not polished, but it works. Hopefully we'll update this BRC workflow to be more similar to [this doodad version](https://github.com/rail-berkeley/doodad), but that's a work in progress._

Setting up doodad + BRC takes a bit of extra leg-work, because you can't launched code from outside BRC using a API. Instead, you need to log in to the BRC node and launch code from inside of it.
The overall workflow is:
1. Run `run_experiment(...)` on your local machine, which will generate one or two scripts for you to SCP over, depending on the mode.
2. Copy the scripts **and your code** over to to BRC.
3. Run the script on BRC.
4. (optionally) Wait until your script starts running before launching any other job.

One warning with this workflow is that **you can't change your code too much before your job actually runs.**
Your jobs on BRC use a local copy of your code (step 2), and so then the job may fail if the code it depends on changed between when you queued up the job and when it started running.
In my experience, this doesn't happen too often since I hide new features behind boolean flags that default to the old behavior, but it's important to keep in mind. Don't repeat step 2 for another job if it'll cause the first job to fail!

Some more details on how to set up your config file and for each component is given below.

### Setting up config file
Here's an example of what your config file might look like:
```
# Some example configs that you can use.
SLURM_CONFIGS = dict(
    cpu=dict(
        account_name='co_rail',
        n_gpus=0,
        partition='savio',
        max_num_cores_per_node=20,
        time_in_mins=int(2.99*24*60),
        # see https://research-it.berkeley.edu/services/high-performance-computing/user-guide/savio-user-guide#Hardware
    ),
    gpu=dict(
        account_name='co_rail',
        partition='savio3_2080ti',
        # account_name='fc_rail',
        # partition='savio2_1080ti',
        n_gpus=1,
        max_num_cores_per_node=8,
        n_cpus_per_task=2,
        time_in_mins=int(2.99*24*60),
        extra_flags='--qos rail_2080ti3_normal --mail-user=vitchyr@berkeley.edu --mail-type=FAIL',
        # see https://research-it.berkeley.edu/services/high-performance-computing/user-guide/savio-user-guide#Hardware
    ),
)
# This is necessary for the GPU machines on BRC.
BRC_EXTRA_SINGULARITY_ARGS = '--writable -B /usr/lib64 -B /var/lib/dcv-gl'
TASKFILE_PATH_ON_BRC = '/global/scratch/vitchyr/logs/taskfile_from_doodad.sh'
SLURM_DIRECTORY_FOR_JOB_SCRIPT = '/global/scratch/vitchyr/doodad/launch'

# This is the same as `CODE_DIRS_TO_MOUNT` but the paths should be based on the BRC paths.
SSS_CODE_DIRS_TO_MOUNT = [
    '/global/home/users/vitchyr/git/railrl',
    '/global/home/users/vitchyr/git/multiworld',
]
# Ditto!
SSS_NON_CODE_DIRS_TO_MOUNT = [
    dict(
        local_dir='/global/home/users/vitchyr/.mujoco',
        mount_point='/root/.mujoco',
    ),
]
# where do you want doodad to output to?
SSS_LOG_DIR = '/global/scratch/user/...'
# point to your singularity image
SSS_GPU_IMAGE = SSS_CPU_IMAGE = '/global/scratch/user/...'
SSS_RUN_DOODAD_EXPERIMENT_SCRIPT_PATH = (
    '/global/home/users/.../doodad/doodad/easy_launch/run_experiment.py' )
# These commands are specific to my Docker + mujoco setup, but you'll probably need to at least include the LD_LIBRARY_PATH.
SSS_PRE_CMDS = [
    'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH'
    ':/global/home/users/vitchyr/.mujoco/mjpro150/bin'
    ':/global/home/users/vitchyr/.mujoco/mujoco200/bin',
    'export PATH=/global/home/users/vitchyr/railrl/bin:/opt/conda/envs/railrl/bin/:$PATH',
]
```

#### Additional slurm configs
You can also add extra keys to `SLURM_CONFIGS` and reference them by name when calling `run_experiment`, i.e.

In your settings
```python
SLURM_CONFIGS = dict(
    cpu=dict(...),
    gpu=dict(...),
    custom=dict(
        account_name='fc_rail',
        n_gpus=0,
        partition='savio',
        max_num_cores_per_node=20,
        time_in_mins=int(2.99*24*60),
        # see https://research-it.berkeley.edu/services/high-performance-computing/user-guide/savio-user-guide#Hardware
        extra_flags='--qos savio_normal',
    ),
```

In your launch script
```
run_experiment(my_function, slurm_config_name='custom', use_gpu=False)
```
If you don't specify a `slurm_config_name`, it will use the `SLURM_CONFIGS['gpu']` if `use_gpu=True` and `SLURM_CONFIGS['cpu']` if `use_gpu=False`.



### Script generation
The two modes for BRC are script slurm singularity (sss) and high throughput (HTP) mode.
If you're going to use GPUs, use `sss` mode.
The `sss` mode uses the standard slurm config, and it generates a script in `/tmp/script_to_scp_over.sh` where each line is a different sbatch command.

For CPU jobs that do not require 20 CPUs at once, use the `htp` mode!
Whenever you run a job on a BRC node with CPU instances, you always reserve all the CPUs on it (most of them have 20), so it's really wasteful to run one experiment on one node.
HTP mode allows you to run up to 20 jobs in parallel using a single node.
The `htp` mode uses a special [high throughput mode](https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/running-your-jobs/hthelper-script) that's provided by BRC.
You run one sbatch command which reads in a "taskfile." The taskfile is a list of commands. The reason this is useful is that it will automatically parallelize the jobs across the difference cores. See the [high throughput mode](https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/running-your-jobs/hthelper-script) page for more detail.

HTP mode will automatically generate both the script to run the sbatch command, as well as the task file. For example, if you run
```
run_experiment(function, exp_name='doodad-easy-launch', mode='htp')
```
it will generate two files `/tmp/script_to_scp_over.sh` and `/tmp/taskfile_from_doodad.sh`.
Copy these two files over to BRC and then run `script_to_scp_over.sh`.
Note that `script_to_scp_over.sh` will be a really short that looks something like
```
sbatch -A fc_rail -p savio -t 4305 -N 1 -n 5 --cpus-per-task=1 --qos savio_normal --wrap=$'module load gcc openmpi;ht_helper.sh -m "python/3.5" -t /global/scratch/vitchyr/logs/taskfile_from_doodad.sh'
```
That final path (`/global/scratch/vitchyr/logs/taskfile_from_doodad.sh`) should be where you copy the `/tmp/taskfile_from_doodad.sh` over to.
It's filled in automatically using the config's `TASKFILE_PATH_ON_BRC` variable.

### Copying files over
Here's the script I run on BRC to SCP code over:
```
rsync -rtv -e 'ssh' --exclude='*.simg' --exclude='*.img' --exclude='.git/' --exclude='.idea/' --exclude='*/__pycache__/' --exclude='*.pyc' --include='*/' --include='*.py' vitchyr@$COMPUTER_URL:/home/vitchyr    /git/doodad/ /global/home/users/vitchyr/git/doodad/
rsync -rtv -e 'ssh' --exclude='*.simg' --exclude='*.img' --exclude='.git/' --exclude='.idea/' --exclude='*/__pycache__/' --exclude='*.pyc' --include='*/' --include='*.py' vitchyr@$COMPUTER_URL:/home/vitchyr    /git/railrl/ /global/home/users/vitchyr/git/railrl/
rsync -rtv -e 'ssh' --exclude='*.simg' --exclude='*.img' --exclude='.git/' --exclude='.idea/' --exclude='*/__pycache__/' --exclude='*.pyc' --include='*/' --include='*.py' vitchyr@$COMPUTER_URL:/home/vitchyr    /git/multiworld/ /global/home/users/vitchyr/git/multiworld/

scp vitchyr@$COMPUTER_URL:/tmp/script_to_scp_over.sh /global/scratch/vitchyr/logs/script_to_scp_over.sh
scp vitchyr@$COMPUTER_URL:/tmp/taskfile_from_doodad.sh /global/scratch/vitchyr/logs/taskfile_from_doodad.sh
```
It's important that I copy the code over to the same location that's specified in `SSS_CODE_DIRS_TO_MOUNT`, and that I `scp` the task script to the same path as `TASKFILE_PATH_ON_BRC`.

### DoodadConfig object
As mentioned above, your function needs to take in two parameters: `doodad_config` and `variant`.
The second parameter is determined by you, but what about the first parameter?
The `doodad_config` object is a named tuple with the following information
```
DoodadConfig = NamedTuple(
    'DoodadConfig',
    [
        ('exp_name', str),
        ('base_log_dir', str),
        ('use_gpu', bool),
        ('gpu_id', Union[int, str]),
        ('git_infos', List[GitInfo]),
        ('script_name', str),
        ('extra_launch_info', dict),
    ],
)
GitInfo = NamedTuple(
    'GitInfo',
    [
        ('directory', str),
        ('code_diff', str),
        ('code_diff_staged', str),
        ('commit_hash', str),
        ('branch_name', str),
    ],
)
```
The most important parameter is `doodad_config.base_log_dir`, as this is where you must output data to. Under the hood, this directory will be re-directed via singularity to some local path that's determined by `SSS_LOG_DIR`. For best practice, just have your function read in this log directory and write to it.
Otherwise, you can read information like the GPU ID or the python script name used to run this experiment.


#### Git Information
_Feel free to ignore this extra feature, but it's nice for reproducibility._

If you have `GitPython` installed, then the meta-data will also save information like what the state of your git repositories are.
You might want to save this information to your directory as it makes your experiments more reproducible.
See [this repository](https://github.com/vitchyr/easy-logger/) for an [example of how to save this information](https://github.com/vitchyr/easy-logger/blob/37d8316ff5ca40dc7b23ede79f8b15496c08c9f9/easy_logger/logging.py#L315).


## Local mode
To set this up, just specify the `LOCAL_LOG_DIR` variable to point to where you want the output directory to be.
```
LOCAL_LOG_DIR = '/home/user/path/to/logs`
```

## EC2
TODO

## GCP
TODO
