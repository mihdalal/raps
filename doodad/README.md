# doodad


A library for launching python programs on different machines. Currently supports running locally and over EC2 and SSH (via Docker) with minimal (if any) modification to your existing program.

EC2 code is based on [rllab](https://github.com/rll/rllab/)'s code.


## Setup

- Add this repo to your pythonpath. 
```
export PYTHONPATH=$PYTHONPATH:/path/to/this/repo
```

- Install dependencies
```
pip install -r requirements.txt
```

- (Optional) Set up EC2
```
python scripts/ec2_setup.py
```

- (Optional) Set up [Docker](https://docs.docker.com/engine/installation/). This is required on the target machine if running in a Docker-enabled mode.

- (Optional) Set up GCP
  - https://cloud.google.com/sdk/docs/quickstart-debian-ubuntu
  - https://cloud.google.com/compute/docs/tutorials/python-guide
  - https://cloud.google.com/storage/docs/reference/libraries#client-libraries-install-python


## Example

See [ec2_launch_test.py](https://github.com/justinjfu/doodad/blob/master/examples/ec2_launch/ec2_launch_test.py) for an example on how to run scripts on EC2, over SSH, or locally.

## Tutorial

See the [wiki](https://github.com/justinjfu/doodad/wiki/Home)

## Changelog
17 March 2020 (v0.2.4)
 - Add `doodad.easy_launch` to make it very easy to run python functions across different modes supported by `doodad`.

04 March 2020 (v0.2.3)
 - SlurmConfig is only responsible for config.
 - Require user to explicitly specify whether or not they want to overwrite generated script in `ScriptSlurmSingularity` and `BrcHighThroughputMode`

26 February 2020 (v0.2.2)
 - Refactor slurm-based modes.
 - Add support for Berkeley Research Compute's high-throughput node.

## TODOs
- Add support for automatic experiment restarting (will require the user to write a save_state and restore_state function, or use something like CRIU)
- Fix output directories when using docker showing up as root permissions.

# `doodad.easy_launch`
The `doodad.easy_launch` module is intended to make it very easy to launch experiments. Usage:

```python
from doodad.easy_launch.python_function import run_experiment


def function(doodad_config, variant):
    print("The learning rate is", variant['learning_rate'])
    print("You are", variant['parameter'])
    print("Save to", doodad_config.base_log_dir)
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
    )

```

To use it you well need to create a private version of the configuration file:
```bash
cp doodad/easy_launch/config.py doodad/easy_launch/config_private.py
```

See the [README](doodad/easy_launch/README.md) for more details.
