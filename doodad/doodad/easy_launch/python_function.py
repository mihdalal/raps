import copy
import os
import os.path as osp
import sys
import time
from typing import List, NamedTuple, Union

import __main__ as main

import doodad
import doodad.mode
import doodad.mount as mount
import doodad.ssh
from doodad.easy_launch import config
from doodad.slurm.slurm_util import SlurmConfig
from doodad.utils import REPO_DIR

GitInfo = NamedTuple(
    "GitInfo",
    [
        ("directory", str),
        ("code_diff", str),
        ("code_diff_staged", str),
        ("commit_hash", str),
        ("branch_name", str),
    ],
)
DoodadConfig = NamedTuple(
    "DoodadConfig",
    [
        ("exp_name", str),
        ("base_log_dir", str),
        ("use_gpu", bool),
        ("gpu_id", Union[int, str]),
        ("git_infos", List[GitInfo]),
        ("script_name", str),
        ("extra_launch_info", dict),
    ],
)

CODE_MOUNTS = [
    mount.MountLocal(local_dir=REPO_DIR, pythonpath=True),
]
for code_dir in config.CODE_DIRS_TO_MOUNT:
    CODE_MOUNTS.append(mount.MountLocal(local_dir=code_dir, pythonpath=True))

NON_CODE_MOUNTS = [
    mount.MountLocal(**non_code_mapping)
    for non_code_mapping in config.NON_CODE_DIRS_TO_MOUNT
]
SSS_CODE_MOUNTS = [
    mount.MountLocal(**non_code_mapping)
    for non_code_mapping in config.SSS_NON_CODE_DIRS_TO_MOUNT
]
SSS_NON_CODE_MOUNTS = [
    mount.MountLocal(local_dir=code_dir, pythonpath=True)
    for code_dir in config.SSS_CODE_DIRS_TO_MOUNT
]


_global_target_mount = None
_global_is_first_launch = True
_global_n_tasks_total = 0


def run_experiment(
    method_call,
    mode="local",
    exp_name="default",
    variant=None,
    prepend_date_to_exp_name=True,
    use_gpu=False,
    gpu_id=0,
    base_log_dir=None,
    local_input_dir_to_mount_point_dict=None,  # TODO(vitchyr): test this
    # local settings
    skip_wait=False,
    # ec2 settings
    sync_interval=180,
    region="us-east-1",
    instance_type=None,
    spot_price=None,
    verbose=False,
    num_exps_per_instance=1,
    docker_image=None,
    # sss settings
    time_in_mins=None,
    slurm_config_name=None,
    # ssh settings
    ssh_host=None,
    # gcp
    gcp_kwargs=None,
    s3_log_prefix=None,
    s3_log_name="",
):
    """
    Usage:
    ```
    def foo(doodad_config, variant):
        x = variant['x']
        y = variant['y']
        print("sum", x+y)
        with open(doodad_config.base_log_dir, "w") as f:
          f.write('sum = %f' % x + y)

    variant = {
        'x': 4,
        'y': 3,
    }
    run_experiment(foo, variant, exp_name='my-experiment', mode='ec2')
    ```

    For outputs to be saved properly, make sure you write to the directory
    in `doodad_config.base_log_dir`. Do NOT output to
    `easy_launch.config.LOCAL_LOG_DIR` or any other directory in config.
    This ensures that when you run code on GCP or AWS, it'll save to the proper
    location and get synced accordingly.

    Within the corresponding output mount, the outputs are saved to
    `base_log_dir/<date>-my-experiment/<date>-my-experiment-<unique-id>`

    For local experiment, the base_log_dir is determined by
    `config.LOCAL_LOG_DIR/`. For GCP or AWS, base_log_dir will be synced to some
    bucket.

    :param method_call: a function that takes in a dictionary as argument
    :param mode: A string:
     - 'local'
     - 'local_docker'
     - 'ec2'
     - 'here_no_doodad': Run without doodad call
     - 'ssh'
     - 'gcp'
     - 'local_singularity': run locally with singularity
     - 'htp': generate a taskfile and script for using BRC's high-throughput script
     - 'slurm_singularity': submit a slurm job using singularity
     - 'sss': generate a script to run on some slurm job using singularity
    :param exp_name: name of experiment
    :param variant: Dictionary
    :param prepend_date_to_exp_name: If False, do not prepend the date to
    the experiment directory.
    :param use_gpu:
    :param sync_interval: How often to sync s3 data (in seconds).
    :param local_input_dir_to_mount_point_dict: Dictionary for doodad.
    :param ssh_host: the name of the host you want to ssh onto, should correspond to an entry in
    config.py of the following form:
    SSH_HOSTS=dict(
        ssh_host=dict(
            username='username',
            hostname='hostname/ip address',
        )
    )
    - if ssh_host is set to None, you will use ssh_host specified by
    config.SSH_DEFAULT_HOST
    :return:
    """
    global _global_target_mount
    global _global_is_first_launch
    global _global_n_tasks_total

    """
    Sanitize inputs as needed
    """
    variant = sanitize_variant(variant)
    base_log_dir = sanitize_base_log_dir(base_log_dir, mode)
    base_exp_name = exp_name
    if prepend_date_to_exp_name:
        exp_name = time.strftime("%y-%m-%d") + "-" + exp_name
    git_infos = generate_git_infos()

    doodad_config = DoodadConfig(
        exp_name=exp_name,
        base_log_dir=base_log_dir,
        use_gpu=use_gpu,
        gpu_id=gpu_id,
        git_infos=git_infos,
        script_name=" ".join(sys.argv),
        extra_launch_info=dict(
            base_exp_name=base_exp_name,
            instance_type=str(instance_type),
        ),
    )
    if mode == "here_no_doodad":
        return method_call(doodad_config, variant)

    """
    Safety Checks
    """

    if mode == "ec2" or mode == "gcp":
        if _global_is_first_launch and not query_yes_no(
            "{} costs money. Are you sure you want to run?".format(mode)
        ):
            sys.exit(1)
        if _global_is_first_launch and use_gpu:
            if not query_yes_no(
                "{} is more expensive with GPUs. Confirm?".format(mode)
            ):
                sys.exit(1)

    """
    GPU vs normal configs
    """
    if use_gpu:
        docker_image = docker_image or config.GPU_DOODAD_DOCKER_IMAGE
        if instance_type is None:
            instance_type = config.GPU_INSTANCE_TYPE
        else:
            assert instance_type[0] in {"g", "p"}
        if spot_price is None:
            spot_price = config.GPU_SPOT_PRICE
        doodad_config.extra_launch_info["docker_image"] = docker_image
    else:
        docker_image = docker_image or config.DOODAD_DOCKER_IMAGE
        if instance_type is None:
            instance_type = config.INSTANCE_TYPE
        if spot_price is None:
            spot_price = config.SPOT_PRICE
        doodad_config.extra_launch_info["docker_image"] = docker_image
    if mode in {"sss", "htp"}:
        if use_gpu:
            singularity_image = config.SSS_GPU_IMAGE
        else:
            singularity_image = config.SSS_CPU_IMAGE
        doodad_config.extra_launch_info["singularity_image"] = singularity_image
    elif mode in ["local_singularity", "slurm_singularity"]:
        singularity_image = config.SINGULARITY_IMAGE
        doodad_config.extra_launch_info["singularity_image"] = singularity_image
    else:
        singularity_image = None

    """
    Get the mode
    """
    mode_kwargs = {}
    if mode == "ec2":
        image_id = config.REGION_TO_GPU_AWS_IMAGE_ID[region]
        doodad_config.extra_launch_info["aws_ami_id"] = image_id
    if hasattr(config, "AWS_S3_PATH"):
        aws_s3_path = config.AWS_S3_PATH
    else:
        aws_s3_path = None

    """
    Create mode
    """
    _global_n_tasks_total += 1
    if mode == "local":
        dmode = doodad.mode.Local(skip_wait=skip_wait)
    elif mode == "local_docker":
        dmode = doodad.mode.LocalDocker(
            image=docker_image,
            gpu=use_gpu,
        )
    elif mode == "ssh":
        if ssh_host:
            ssh_dict = config.SSH_HOSTS[ssh_host]
        else:
            ssh_dict = config.SSH_HOSTS[config.SSH_DEFAULT_HOST]
        credentials = doodad.ssh.credentials.SSHCredentials(
            username=ssh_dict["username"],
            hostname=ssh_dict["hostname"],
            identity_file=config.SSH_PRIVATE_KEY,
        )
        dmode = doodad.mode.SSHDocker(
            credentials=credentials,
            image=docker_image,
            gpu=use_gpu,
            tmp_dir=config.SSH_TMP_DIR,
        )
    elif mode == "local_singularity":
        dmode = doodad.mode.LocalSingularity(
            image=singularity_image,
            gpu=use_gpu,
            pre_cmd=config.SINGULARITY_PRE_CMDS,
        )
    elif mode in {"slurm_singularity", "sss", "htp"}:
        if slurm_config_name is None:
            slurm_config_name = "gpu" if use_gpu else "cpu"
        slurm_config_kwargs = config.SLURM_CONFIGS[slurm_config_name]
        if use_gpu:
            assert slurm_config_kwargs["n_gpus"] > 0, slurm_config_name
        else:
            assert slurm_config_kwargs["n_gpus"] == 0, slurm_config_name
        if time_in_mins is not None:
            slurm_config_kwargs["time_in_mins"] = time_in_mins
        if slurm_config_kwargs["time_in_mins"] is None:
            raise ValueError("Must approximate/set time in minutes")
        slurm_config = SlurmConfig(**slurm_config_kwargs)
        if mode == "slurm_singularity":
            dmode = doodad.mode.SlurmSingularity(
                image=singularity_image,
                gpu=use_gpu,
                skip_wait=skip_wait,
                pre_cmd=config.SINGULARITY_PRE_CMDS,
                extra_args=config.BRC_EXTRA_SINGULARITY_ARGS,
                slurm_config=slurm_config,
            )
        elif mode == "htp":
            dmode = doodad.mode.BrcHighThroughputMode(
                image=singularity_image,
                gpu=use_gpu,
                pre_cmd=config.SSS_PRE_CMDS,
                extra_args=config.BRC_EXTRA_SINGULARITY_ARGS,
                slurm_config=slurm_config,
                taskfile_path_on_brc=config.TASKFILE_PATH_ON_BRC,
                overwrite_task_script=_global_is_first_launch,
                n_tasks_total=_global_n_tasks_total,
            )
        else:
            dmode = doodad.mode.ScriptSlurmSingularity(
                image=singularity_image,
                gpu=use_gpu,
                pre_cmd=config.SSS_PRE_CMDS,
                extra_args=config.BRC_EXTRA_SINGULARITY_ARGS,
                slurm_config=slurm_config,
                overwrite_script=_global_is_first_launch,
            )
    elif mode == "ec2":
        # Do this separately in case someone does not have EC2 configured
        if s3_log_prefix is None:
            s3_log_prefix = exp_name
        dmode = doodad.mode.EC2AutoconfigDocker(
            image=docker_image,
            image_id=image_id,
            region=region,
            instance_type=instance_type,
            spot_price=spot_price,
            s3_log_prefix=s3_log_prefix,
            # Make the sub-directories within launching code rather
            # than relying on doodad.
            s3_log_name=s3_log_name,
            gpu=use_gpu,
            aws_s3_path=aws_s3_path,
            num_exps=num_exps_per_instance,
            **mode_kwargs
        )
    elif mode == "gcp":
        image_name = config.GCP_IMAGE_NAME
        if use_gpu:
            image_name = config.GCP_GPU_IMAGE_NAME

        if gcp_kwargs is None:
            gcp_kwargs = {}
        config_kwargs = {
            **config.GCP_DEFAULT_KWARGS,
            **dict(image_name=image_name),
            **gcp_kwargs,
        }
        dmode = doodad.mode.GCPDocker(
            image=docker_image,
            gpu=use_gpu,
            gcp_bucket_name=config.GCP_BUCKET_NAME,
            gcp_log_prefix=exp_name,
            gcp_log_name="",
            num_exps=num_exps_per_instance,
            **config_kwargs
        )
        doodad_config.extra_launch_info["gcp_image"] = image_name
    else:
        raise NotImplementedError("Mode not supported: {}".format(mode))
    _global_is_first_launch = False

    """
    Get the mounts
    """
    mounts = create_mounts(
        base_log_dir=base_log_dir,
        mode=mode,
        sync_interval=sync_interval,
        local_input_dir_to_mount_point_dict=local_input_dir_to_mount_point_dict,
    )

    """
    Get the outputs
    """
    launch_locally = None
    # target = config.RUN_DOODAD_EXPERIMENT_SCRIPT_PATH
    target = osp.join(REPO_DIR, "doodad/easy_launch/run_experiment.py")
    snapshot_dir_for_script = None  # if not update, will be set automatically
    if mode == "ec2":
        # Ignored since I'm setting the snapshot dir directly
        base_log_dir_for_script = None
        # The snapshot dir needs to be specified for S3 because S3 will
        # automatically create the experiment director and sub-directory.
        snapshot_dir_for_script = config.OUTPUT_DIR_FOR_DOODAD_TARGET
    elif mode == "local":
        base_log_dir_for_script = base_log_dir
    elif mode == "local_docker":
        base_log_dir_for_script = config.OUTPUT_DIR_FOR_DOODAD_TARGET
    elif mode == "ssh":
        base_log_dir_for_script = config.OUTPUT_DIR_FOR_DOODAD_TARGET
    elif mode in {"local_singularity", "slurm_singularity", "sss", "htp"}:
        base_log_dir_for_script = base_log_dir
        launch_locally = True
        if mode in {"sss", "htp"}:
            target = config.SSS_RUN_DOODAD_EXPERIMENT_SCRIPT_PATH
    elif mode == "here_no_doodad":
        base_log_dir_for_script = base_log_dir
    elif mode == "gcp":
        # Ignored since I'm setting the snapshot dir directly
        base_log_dir_for_script = None
        snapshot_dir_for_script = config.OUTPUT_DIR_FOR_DOODAD_TARGET
    else:
        raise NotImplementedError("Mode not supported: {}".format(mode))
    doodad_config = doodad_config._replace(base_log_dir=base_log_dir_for_script)
    _global_target_mount = doodad.launch_python(
        target=target,
        mode=dmode,
        mount_points=mounts,
        args={
            "method_call": method_call,
            "output_dir": snapshot_dir_for_script,
            "doodad_config": doodad_config,
            "variant": variant,
            "mode": mode,
        },
        use_cloudpickle=True,
        target_mount=_global_target_mount,
        verbose=verbose,
        launch_locally=launch_locally,
    )


def sanitize_base_log_dir(base_log_dir, mode):
    if mode == "ssh" and base_log_dir is None:
        base_log_dir = config.SSH_LOG_DIR
    if base_log_dir is None:
        if mode in {"sss", "htp"}:
            base_log_dir = config.SSS_LOG_DIR
        else:
            base_log_dir = config.LOCAL_LOG_DIR
    return base_log_dir


def sanitize_variant(variant):
    variant = variant or {}
    if "doodad_config" in variant:
        raise ValueError("The key `doodad_config` is now allowed in variant.")
    for key, value in recursive_items(variant):
        # This check isn't really necessary, but it's to prevent myself from
        # forgetting to pass a variant through dot_map_dict_to_nested_dict.
        if "." in key:
            raise Exception(
                "Variants should not have periods in keys. Did you mean to "
                "convert {} into a nested dictionary?".format(key)
            )
    return copy.deepcopy(variant)


def generate_git_infos():
    try:
        import git

        doodad_path = osp.abspath(osp.join(osp.dirname(doodad.__file__), os.pardir))
        dirs = config.CODE_DIRS_TO_MOUNT + [doodad_path]

        git_infos = []
        for directory in dirs:
            # Idk how to query these things, so I'm just doing try-catch
            try:
                repo = git.Repo(directory)
                try:
                    branch_name = repo.active_branch.name
                except TypeError:
                    branch_name = "[DETACHED]"
                git_infos.append(
                    GitInfo(
                        directory=directory,
                        code_diff=repo.git.diff(None),
                        code_diff_staged=repo.git.diff("--staged"),
                        commit_hash=repo.head.commit.hexsha,
                        branch_name=branch_name,
                    )
                )
            except git.exc.InvalidGitRepositoryError:
                pass
    except (ImportError, UnboundLocalError, NameError):
        git_infos = None
    return git_infos


def create_mounts(
    mode,
    base_log_dir,
    sync_interval=180,
    local_input_dir_to_mount_point_dict=None,
):
    if mode in {"sss", "htp"}:
        code_mounts = SSS_CODE_MOUNTS
        non_code_mounts = SSS_NON_CODE_MOUNTS
    else:
        code_mounts = CODE_MOUNTS
        non_code_mounts = NON_CODE_MOUNTS

    if local_input_dir_to_mount_point_dict is None:
        local_input_dir_to_mount_point_dict = {}
    else:
        raise NotImplementedError("TODO(vitchyr): Implement this")

    mounts = [m for m in code_mounts]
    for dir, mount_point in local_input_dir_to_mount_point_dict.items():
        mounts.append(
            mount.MountLocal(
                local_dir=dir,
                mount_point=mount_point,
                pythonpath=False,
            )
        )

    if mode != "local":
        for m in non_code_mounts:
            mounts.append(m)

    if mode == "ec2":
        output_mount = mount.MountS3(
            s3_path="",
            mount_point=config.OUTPUT_DIR_FOR_DOODAD_TARGET,
            output=True,
            sync_interval=sync_interval,
            include_types=config.AWS_FILE_TYPES_TO_SAVE,
        )
    elif mode == "gcp":
        output_mount = mount.MountGCP(
            gcp_path="",
            mount_point=config.OUTPUT_DIR_FOR_DOODAD_TARGET,
            output=True,
            gcp_bucket_name=config.GCP_BUCKET_NAME,
            sync_interval=sync_interval,
            include_types=config.GCP_FILE_TYPES_TO_SAVE,
        )
    elif mode in {"local", "local_singularity", "slurm_singularity", "sss", "htp"}:
        # To save directly to local files, skip mounting
        output_mount = mount.MountLocal(
            local_dir=base_log_dir,
            mount_point=None,
            output=True,
        )
    elif mode == "local_docker":
        output_mount = mount.MountLocal(
            local_dir=base_log_dir,
            mount_point=config.OUTPUT_DIR_FOR_DOODAD_TARGET,
            output=True,
        )
    elif mode == "ssh":
        output_mount = mount.MountLocal(
            local_dir=base_log_dir,
            mount_point=config.OUTPUT_DIR_FOR_DOODAD_TARGET,
            output=True,
        )
    else:
        raise NotImplementedError("Mode not supported: {}".format(mode))
    mounts.append(output_mount)
    return mounts


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.
    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).
    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


def recursive_items(dictionary):
    """
    Get all (key, item) recursively in a potentially recursive dictionary.
    Usage:

    ```
    x = {
        'foo' : {
            'bar' : 5
        }
    }
    recursive_items(x)
    # output:
    # ('foo', {'bar' : 5})
    # ('bar', 5)
    ```
    :param dictionary:
    :return:
    """
    for key, value in dictionary.items():
        yield key, value
        if type(value) is dict:
            yield from recursive_items(value)
