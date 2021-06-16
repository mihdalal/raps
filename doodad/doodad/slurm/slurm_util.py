import math


class SlurmConfig(object):
    """
    Wrap a command around a call to sbatch

    Usage:
    ```
    generator = SBatchGenerator()
    cmd = "echo 'Hello, world!'"
    sbatch_cmd = generator.wrap(cmd)
    """

    def __init__(
        self,
        account_name,
        partition,
        time_in_mins,
        max_num_cores_per_node,
        n_gpus=0,
        n_cpus_per_task=1,
        n_nodes=None,
        extra_flags="",
    ):
        if n_gpus > 0 and n_cpus_per_task < 2:
            raise ValueError("Must have at least 2 cpus per GPU task")
        self.account_name = account_name
        self.partition = partition
        self.time_in_mins = time_in_mins
        self.n_nodes = n_nodes
        self.n_gpus = n_gpus
        self.n_cpus_per_task = n_cpus_per_task
        self.max_num_cores_per_node = max_num_cores_per_node
        self.extra_flags = extra_flags


class SlurmConfigMatrix(object):
    """
    Wrap a command around a call to sbatch

    Usage:
    ```
    generator = SBatchGenerator()
    cmd = "echo 'Hello, world!'"
    sbatch_cmd = generator.wrap(cmd)
    """

    def __init__(
        self,
        partition,
        time,
        mem,
        n_cpus_per_task=1,
        n_cpus_per_gpu=1,
        n_gpus=0,
        extra_flags="",
    ):
        if n_gpus > 0 and n_cpus_per_gpu < 2:
            raise ValueError("Must have at least 2 cpus per GPU task")
        self.partition = partition
        self.time = time
        self.n_cpus_per_gpu = n_cpus_per_gpu
        self.n_cpus_per_task = n_cpus_per_task
        self.n_gpus = n_gpus
        self.mem = mem
        self.extra_flags = extra_flags


def wrap_command_with_sbatch(
    cmd: str,
    config: SlurmConfig,
    n_tasks: int,
):
    """
    Wrap a command around a call to sbatch

    Usage:
    ```
    config = SlurmConfig()
    cmd = "echo 'Hello, world!'"
    sbatch_cmd = wrap_command_with_sbatch(cmd, config)
    """
    cmd = cmd.replace("'", "\\'")
    num_cpus = n_tasks * config.n_cpus_per_task
    n_nodes = config.n_nodes or math.ceil(num_cpus / config.max_num_cores_per_node)
    if config.n_gpus > 0:
        full_cmd = (
            "sbatch -A {account_name} -p {partition} -t {time}"
            " -N {nodes} -n {n_tasks} --cpus-per-task={cpus_per_task}"
            " --gres=gpu:{n_gpus} {extra_flags} --wrap=$'{cmd}'".format(
                account_name=config.account_name,
                partition=config.partition,
                time=config.time_in_mins,
                nodes=n_nodes,
                n_tasks=n_tasks,
                cpus_per_task=config.n_cpus_per_task,
                cmd=cmd,
                n_gpus=config.n_gpus,
                extra_flags=config.extra_flags,
            )
        )
    else:
        full_cmd = (
            "sbatch -A {account_name} -p {partition} -t {time}"
            " -N {nodes} -n {n_tasks} --cpus-per-task={cpus_per_task}"
            " {extra_flags} --wrap=$'{cmd}'".format(
                account_name=config.account_name,
                partition=config.partition,
                time=config.time_in_mins,
                nodes=n_nodes,
                n_tasks=n_tasks,
                cpus_per_task=config.n_cpus_per_task,
                cmd=cmd,
                extra_flags=config.extra_flags,
            )
        )
    return full_cmd


def wrap_command_with_sbatch_matrix(
    cmd: str,
    config: SlurmConfigMatrix,
    logdir: str,
):
    """
    Wrap a command around a call to sbatch

    Usage:
    ```
    config = SlurmConfig()
    cmd = "echo 'Hello, world!'"
    sbatch_cmd = wrap_command_with_sbatch(cmd, config)
    """
    cmd = cmd.replace("'", "\\'")
    logdir = logdir + "/slurm-%j.out"
    if config.n_gpus > 0:
        full_cmd = (
            "sbatch -p {partition} -t {time}"
            " --cpus-per-gpu {n_cpus_per_gpu} --cpus-per-task {n_cpus_per_task} --mem={mem} -o {logdir}"
            " --gres=gpu:{n_gpus} {extra_flags} --wrap=$'{cmd}'".format(
                partition=config.partition,
                time=config.time,
                n_cpus_per_gpu=config.n_cpus_per_gpu,
                n_cpus_per_task=config.n_cpus_per_task,
                mem=config.mem,
                logdir=logdir,
                cmd=cmd,
                n_gpus=config.n_gpus,
                extra_flags=config.extra_flags,
            )
        )
    else:
        full_cmd = (
            "sbatch -p {partition} -t {time}"
            " --cpus-per-task {n_cpus_per_task} --mem={mem}"
            " {extra_flags} --wrap=$'{cmd}'".format(
                partition=config.partition,
                time=config.time,
                n_cpus_per_task=config.n_cpus_per_task,
                mem=config.mem,
                cmd=cmd,
                extra_flags=config.extra_flags,
            )
        )
    return full_cmd
