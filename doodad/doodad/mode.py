import base64
import json
import os
import subprocess
import tempfile
import time
import uuid

from doodad.slurm.slurm_util import SlurmConfig, SlurmConfigMatrix

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from doodad import utils
from doodad.slurm import slurm_util

from .ec2.aws_util import s3_exists, s3_upload
from .gcp.gcp_util import (
    GCP_SHUTDOWN_SCRIPT_PATH,
    GCP_STARTUP_SCRIPT_PATH,
    get_gpu_type,
    get_machine_type,
    upload_file_to_gcp_storage,
)
from .mount import MountGCP, MountLocal, MountS3


class LaunchMode(object):
    def launch_command(self, cmd, mount_points=None, dry=False, verbose=False):
        raise NotImplementedError()


class Local(LaunchMode):
    def __init__(self, skip_wait=False):
        super(Local, self).__init__()
        self.env = {}
        self.skip_wait = skip_wait

    def launch_command(self, cmd, mount_points=None, dry=False, verbose=False):
        if dry:
            print(cmd)
            return

        commands = utils.CommandBuilder()
        # chdir to home dir
        commands.append("cd %s" % (os.path.expanduser("~")))

        # do mounting
        py_path = []
        cleanup_commands = utils.CommandBuilder()
        for mount in mount_points:
            print("mounting:", mount)
            if isinstance(mount, MountLocal):
                if not mount.no_remount:
                    mount.create_if_nonexistent()
                    commands.append(
                        "ln -s %s %s" % (mount.local_dir, mount.mount_point)
                    )
                    # subprocess.call(symlink_cmd, shell=True)
                    if mount.cleanup:
                        cleanup_commands.append('rm "%s"' % mount.mount_point)
                if mount.pythonpath:
                    py_path.append(mount.mount_point)
            else:
                raise NotImplementedError()

        # add pythonpath mounts
        if py_path:
            commands.append("export PYTHONPATH=$PYTHONPATH:%s" % (":".join(py_path)))

        # Add main command
        commands.append(cmd)

        # cleanup
        commands.extend(cleanup_commands)

        # Call everything
        commands.call_and_wait(verbose=verbose, dry=dry, skip_wait=self.skip_wait)


LOCAL = Local()


class DockerMode(LaunchMode):
    def __init__(self, image="ubuntu:16.04", gpu=False, gpu_id=0):
        super(DockerMode, self).__init__()
        self.docker_image = image
        self.docker_name = uuid.uuid4()
        self.gpu = gpu
        self.gpu_id = gpu_id

    def get_docker_cmd(
        self,
        main_cmd,
        extra_args="",
        use_tty=True,
        verbose=True,
        pythonpath=None,
        pre_cmd=None,
        post_cmd=None,
        checkpoint=False,
        no_root=False,
        use_docker_generated_name=False,
    ):
        cmd_list = utils.CommandBuilder()
        if pre_cmd:
            cmd_list.extend(pre_cmd)

        if verbose:
            if self.gpu:
                cmd_list.append('echo "Running in docker with gpu"')
            else:
                cmd_list.append('echo "Running in docker"')
        if pythonpath:
            cmd_list.append("export PYTHONPATH=$PYTHONPATH:%s" % (":".join(pythonpath)))
        if no_root:
            # This should work if you're running a script
            # cmd_list.append('useradd --uid $(id -u) --no-create-home --home-dir / doodaduser')
            # cmd_list.append('su - doodaduser /bin/bash {script}')

            # this is a temp workaround
            extra_args += " -u $(id -u)"

        cmd_list.append(main_cmd)
        if post_cmd:
            cmd_list.extend(post_cmd)

        docker_name = self.docker_name
        if docker_name and not use_docker_generated_name:
            extra_args += " --name %s " % docker_name

        if checkpoint:
            # set up checkpoint stuff
            use_tty = False
            extra_args += " -d "  # detach is optional

        if self.gpu:
            docker_run = "docker run --gpus device={}".format(self.gpu_id)
        else:
            docker_run = "docker run"
        if use_tty:
            docker_prefix = "%s %s -ti %s /bin/bash -c " % (
                docker_run,
                extra_args,
                self.docker_image,
            )
        else:
            docker_prefix = "%s %s %s /bin/bash -c " % (
                docker_run,
                extra_args,
                self.docker_image,
            )
        main_cmd = cmd_list.to_string()
        full_cmd = docker_prefix + ("'%s'" % main_cmd)
        return full_cmd


class LocalDocker(DockerMode):
    def __init__(self, checkpoints=None, skip_wait=False, **kwargs):
        super(LocalDocker, self).__init__(**kwargs)
        self.checkpoints = checkpoints
        self.skip_wait = skip_wait

    def launch_command(self, cmd, mount_points=None, dry=False, verbose=False):
        mnt_args = ""
        py_path = []
        for mount in mount_points:
            if isinstance(mount, MountLocal):
                # mount_pnt = os.path.expanduser(mount.mount_point)
                mount_pnt = mount.mount_dir()
                mnt_args += " -v %s:%s" % (mount.local_dir, mount_pnt)
                utils.call_and_wait("mkdir -p %s" % mount.local_dir)
                if mount.pythonpath:
                    py_path.append(mount_pnt)
            else:
                raise NotImplementedError(type(mount))

        full_cmd = self.get_docker_cmd(
            cmd, extra_args=mnt_args, pythonpath=py_path, checkpoint=self.checkpoints
        )
        utils.call_and_wait(
            full_cmd, verbose=verbose, dry=dry, skip_wait=self.skip_wait
        )


class SSHDocker(DockerMode):
    TMP_DIR = "~/.remote_tmp"

    def __init__(self, credentials=None, tmp_dir=None, **docker_args):
        if tmp_dir is None:
            tmp_dir = SSHDocker.TMP_DIR
        super(SSHDocker, self).__init__(**docker_args)
        self.credentials = credentials
        self.run_id = "run_%s" % uuid.uuid4()
        self.tmp_dir = os.path.join(tmp_dir, self.run_id)
        self.checkpoint = None

    def launch_command(self, main_cmd, mount_points=None, dry=False, verbose=False):
        py_path = []
        remote_cmds = utils.CommandBuilder()
        remote_cleanup_commands = utils.CommandBuilder()
        mnt_args = ""

        tmp_dir_cmd = "mkdir -p %s" % self.tmp_dir
        tmp_dir_cmd = self.credentials.get_ssh_bash_cmd(tmp_dir_cmd)
        utils.call_and_wait(tmp_dir_cmd, dry=dry, verbose=verbose)

        # SCP Code over
        for mount in mount_points:
            if isinstance(mount, MountLocal):
                if mount.read_only:
                    with mount.gzip() as gzip_file:
                        # scp
                        base_name = os.path.basename(gzip_file)
                        # file_hash = hash_file(gzip_path)  # TODO: store all code in a special "caches" folder
                        remote_mnt_dir = os.path.join(
                            self.tmp_dir, os.path.splitext(base_name)[0]
                        )
                        remote_tar = os.path.join(self.tmp_dir, base_name)
                        scp_cmd = self.credentials.get_scp_cmd(gzip_file, remote_tar)
                        utils.call_and_wait(scp_cmd, dry=dry, verbose=verbose)
                    remote_cmds.append("mkdir -p %s" % remote_mnt_dir)
                    unzip_cmd = "tar -xf %s -C %s" % (remote_tar, remote_mnt_dir)
                    remote_cmds.append(unzip_cmd)
                    mount_point = mount.mount_dir()
                    mnt_args += " -v %s:%s" % (
                        os.path.join(
                            remote_mnt_dir, os.path.basename(mount.mount_point)
                        ),
                        mount_point,
                    )
                else:
                    # remote_cmds.append('mkdir -p %s' % mount.mount_point)
                    remote_cmds.append("mkdir -p %s" % mount.local_dir_raw)
                    mnt_args += " -v %s:%s" % (mount.local_dir_raw, mount.mount_point)

                if mount.pythonpath:
                    py_path.append(mount_point)
            else:
                raise NotImplementedError()

        if self.checkpoint and self.checkpoint.restore:
            raise NotImplementedError()
        else:
            docker_cmd = self.get_docker_cmd(
                main_cmd, use_tty=False, extra_args=mnt_args, pythonpath=py_path
            )

        remote_cmds.append(docker_cmd)
        remote_cmds.extend(remote_cleanup_commands)

        with tempfile.NamedTemporaryFile("w+", suffix=".sh") as ntf:
            for cmd in remote_cmds:
                if verbose:
                    ntf.write('echo "%s$ %s"\n' % (self.credentials.user_host, cmd))
                ntf.write(cmd + "\n")
            ntf.seek(0)
            ssh_cmd = self.credentials.get_ssh_script_cmd(ntf.name)

            utils.call_and_wait(ssh_cmd, dry=dry, verbose=verbose)


def dedent(s):
    lines = [l.strip() for l in s.split("\n")]
    return "\n".join(lines)


class EC2SpotDocker(DockerMode):
    def __init__(
        self,
        credentials,
        region="us-west-1",
        s3_bucket_region="us-west-1",
        instance_type="m1.small",
        spot_price=0.0,
        s3_bucket=None,
        terminate=True,
        image_id=None,
        aws_key_name=None,
        iam_instance_profile_name="doodad",
        s3_log_prefix="experiment",
        s3_log_name=None,
        security_group_ids=None,
        security_groups=None,
        aws_s3_path=None,
        extra_ec2_instance_kwargs=None,
        num_exps=1,
        swap_size=4096,
        **kwargs
    ):
        super(EC2SpotDocker, self).__init__(**kwargs)
        if security_group_ids is None:
            security_group_ids = []
        if security_groups is None:
            security_groups = []
        self.credentials = credentials
        self.region = region
        self.s3_bucket_region = s3_bucket_region
        self.spot_price = str(float(spot_price))
        self.instance_type = instance_type
        self.terminate = terminate
        self.s3_bucket = s3_bucket
        self.image_id = image_id
        self.aws_key_name = aws_key_name
        self.s3_log_prefix = s3_log_prefix
        self.s3_log_name = s3_log_name
        self.security_group_ids = security_group_ids
        self.security_groups = security_groups
        self.iam_instance_profile_name = iam_instance_profile_name
        self.extra_ec2_instance_kwargs = extra_ec2_instance_kwargs
        self.num_exps = num_exps
        self.swap_size = swap_size
        self.checkpoint = None

        self.s3_mount_path = "s3://%s/doodad/mount" % self.s3_bucket
        self.aws_s3_path = aws_s3_path or "s3://%s/doodad/logs" % self.s3_bucket

    def upload_file_to_s3(self, script_content, dry=False):
        f = tempfile.NamedTemporaryFile(delete=False)
        f.write(script_content.encode())
        f.close()
        remote_path = os.path.join(
            self.s3_mount_path, "oversize_bash_scripts", str(uuid.uuid4())
        )
        subprocess.check_call(
            ["aws", "s3", "cp", f.name, remote_path, "--region", self.s3_bucket_region]
        )
        os.unlink(f.name)
        return remote_path

    def s3_upload(
        self, file_name, bucket, remote_filename=None, dry=False, check_exist=True
    ):
        if remote_filename is None:
            remote_filename = os.path.basename(file_name)
        remote_path = "doodad/mount/" + remote_filename
        if check_exist:
            if s3_exists(bucket, remote_path, region=self.s3_bucket_region):
                print("\t%s exists! " % os.path.join(bucket, remote_path))
                return "s3://" + os.path.join(bucket, remote_path)
        return s3_upload(
            file_name, bucket, remote_path, dry=dry, region=self.s3_bucket_region
        )

    def make_timekey(self):
        return "%d" % (int(time.time() * 1000))

    def launch_command(self, main_cmd, mount_points=None, dry=False, verbose=False):
        default_config = dict(
            image_id=self.image_id,
            instance_type=self.instance_type,
            key_name=self.aws_key_name,
            spot_price=self.spot_price,
            iam_instance_profile_name=self.iam_instance_profile_name,
            security_groups=self.security_groups,
            security_group_ids=self.security_group_ids,
            network_interfaces=[],
        )
        aws_config = dict(default_config)
        if self.s3_log_name is None:
            exp_name = "{}-{}".format(self.s3_log_prefix, self.make_timekey())
        else:
            exp_name = self.s3_log_name
        exp_prefix = self.s3_log_prefix
        s3_base_dir = os.path.join(
            self.aws_s3_path, exp_prefix.replace("_", "-"), exp_name
        )
        stdout_log_s3_path = os.path.join(s3_base_dir, "stdout_$EC2_INSTANCE_ID.log")

        sio = StringIO()
        sio.write("#!/bin/bash\n")
        sio.write("truncate -s 0 /home/ubuntu/user_data.log\n")
        sio.write("{\n")
        sio.write('die() { status=$1; shift; echo "FATAL: $*"; exit $status; }\n')
        sio.write(
            'EC2_INSTANCE_ID="`wget -q -O - http://169.254.169.254/latest/meta-data/instance-id`"\n'
        )
        sio.write(
            """
            aws ec2 create-tags --resources $EC2_INSTANCE_ID --tags Key=Name,Value={exp_name} --region {aws_region}
        """.format(
                exp_name=exp_name, aws_region=self.region
            )
        )
        sio.write(
            """
            aws ec2 create-tags --resources $EC2_INSTANCE_ID --tags Key=exp_prefix,Value={exp_prefix} --region {aws_region}
        """.format(
                exp_prefix=exp_prefix, aws_region=self.region
            )
        )

        # Add swap file
        if self.gpu:
            swap_location = "/mnt/swapfile"
        else:
            swap_location = "/var/swap.1"
        sio.write(
            "sudo dd if=/dev/zero of={swap_location} bs=1M count={swap_size}\n".format(
                swap_location=swap_location, swap_size=self.swap_size
            )
        )
        sio.write("sudo mkswap {swap_location}\n".format(swap_location=swap_location))
        sio.write(
            "sudo chmod 600 {swap_location}\n".format(swap_location=swap_location)
        )
        sio.write("sudo swapon {swap_location}\n".format(swap_location=swap_location))

        sio.write("service docker start\n")
        sio.write(
            "docker --config /home/ubuntu/.docker pull {docker_image}\n".format(
                docker_image=self.docker_image
            )
        )
        sio.write(
            "export AWS_DEFAULT_REGION={aws_region}\n".format(
                aws_region=self.s3_bucket_region
            )
        )
        sio.write(
            """
            curl "https://s3.amazonaws.com/aws-cli/awscli-bundle.zip" -o "awscli-bundle.zip"
            unzip awscli-bundle.zip
            sudo ./awscli-bundle/install -i /usr/local/aws -b /usr/local/bin/aws
        """
        )

        mnt_args = ""
        py_path = []
        local_output_dir_and_s3_path = []
        max_sync_interval = 0
        for mount in mount_points:
            print("Handling mount: ", mount)
            if isinstance(mount, MountLocal):  # TODO: these should be mount_s3 objects
                if mount.read_only:
                    if mount.path_on_remote is None:
                        with mount.gzip() as gzip_file:
                            gzip_path = os.path.realpath(gzip_file)
                            file_hash = utils.hash_file(gzip_path)
                            s3_path = self.s3_upload(
                                gzip_path,
                                self.s3_bucket,
                                remote_filename=file_hash + ".tar",
                            )
                        mount.path_on_remote = s3_path
                        mount.local_file_hash = gzip_path
                    else:
                        file_hash = mount.local_file_hash
                        s3_path = mount.path_on_remote
                    remote_tar_name = "/tmp/" + file_hash + ".tar"
                    remote_unpack_name = "/tmp/" + file_hash
                    sio.write(
                        "aws s3 cp {s3_path} {remote_tar_name}\n".format(
                            s3_path=s3_path, remote_tar_name=remote_tar_name
                        )
                    )
                    sio.write(
                        "mkdir -p {local_code_path}\n".format(
                            local_code_path=remote_unpack_name
                        )
                    )
                    sio.write(
                        "tar -xvf {remote_tar_name} -C {local_code_path}\n".format(
                            local_code_path=remote_unpack_name,
                            remote_tar_name=remote_tar_name,
                        )
                    )
                    mount_point = os.path.join(
                        "/mounts", mount.mount_point.replace("~/", "")
                    )
                    mnt_args += " -v %s:%s" % (
                        os.path.join(
                            remote_unpack_name, os.path.basename(mount.local_dir)
                        ),
                        mount_point,
                    )
                    if mount.pythonpath:
                        py_path.append(mount_point)
                else:
                    raise ValueError()
            elif isinstance(mount, MountS3):
                # In theory the ec2_local_dir could be some random directory,
                # but we make it the same as the mount directory for
                # convenience.
                #
                # ec2_local_dir: directory visible to ec2 spot instance
                # moint_point: directory visible to docker running inside ec2
                #               spot instance
                ec2_local_dir = mount.mount_point
                s3_path = os.path.join(s3_base_dir, mount.s3_path)
                if self.num_exps == 1:
                    stdout_log_s3_path = os.path.join(
                        s3_path, "stdout_$EC2_INSTANCE_ID.log"
                    )
                if not mount.output:
                    raise NotImplementedError()
                local_output_dir_and_s3_path.append((ec2_local_dir, s3_path))
                sio.write("mkdir -p {remote_dir}\n".format(remote_dir=ec2_local_dir))
                mnt_args += " -v %s:%s" % (ec2_local_dir, mount.mount_point)

                # Sync interval
                sio.write(
                    """
                while /bin/true; do
                    aws s3 sync --exclude '*' {include_string} {log_dir} {s3_path}
                    sleep {periodic_sync_interval}
                done & echo sync initiated
                """.format(
                        include_string=mount.include_string,
                        log_dir=ec2_local_dir,
                        s3_path=s3_path,
                        periodic_sync_interval=mount.sync_interval,
                    )
                )
                max_sync_interval = max(max_sync_interval, mount.sync_interval)

                # Sync on terminate. This catches the case where the spot
                # instance gets terminated before the user script ends.
                #
                # This is hoping that there's at least 3 seconds between when
                # the spot instance gets marked for  termination and when it
                # actually terminates.
                sio.write(
                    """
                    while /bin/true; do
                        if [ -z $(curl -Is http://169.254.169.254/latest/meta-data/spot/termination-time | head -1 | grep 404 | cut -d \  -f 2) ]
                        then
                            logger "Running shutdown hook."
                            aws s3 cp --recursive {log_dir} {s3_path}
                            aws s3 cp /home/ubuntu/user_data.log {stdout_log_s3_path}
                            break
                        else
                            # Spot instance not yet marked for termination.
                            # This is hoping that there's at least 3 seconds
                            # between when the spot instance gets marked for
                            # termination and when it actually terminates.
                            sleep 3
                        fi
                    done & echo log sync initiated
                """.format(
                        log_dir=ec2_local_dir,
                        s3_path=s3_path,
                        stdout_log_s3_path=stdout_log_s3_path,
                    )
                )
            else:
                raise NotImplementedError()

        sio.write(
            """
        while /bin/true; do
            aws s3 cp /home/ubuntu/user_data.log {stdout_log_s3_path}
            sleep {periodic_sync_interval}
        done & echo sync initiated
        """.format(
                stdout_log_s3_path=stdout_log_s3_path,
                periodic_sync_interval=max_sync_interval,
            )
        )

        if self.gpu:
            sio.write("echo 'Testing nvidia-smi'\n")
            sio.write("nvidia-smi\n")
            sio.write("echo 'Testing nvidia-smi inside docker'\n")
            sio.write(
                "docker run --gpus all --rm {docker_image} nvidia-smi\n".format(
                    docker_image=self.docker_image
                )
            )

        if self.checkpoint and self.checkpoint.restore:
            raise NotImplementedError()
        else:
            docker_cmd = self.get_docker_cmd(
                main_cmd,
                use_tty=False,
                extra_args=mnt_args,
                pythonpath=py_path,
                use_docker_generated_name=True,
            )
        assert self.num_exps > 0
        for _ in range(self.num_exps - 1):
            sio.write(docker_cmd + " &\n")
        sio.write(docker_cmd + "\n")

        # Sync all output mounts to s3 after running the user script
        # Ideally the earlier while loop would be sufficient, but it might be
        # the case that the earlier while loop isn't fast enough to catch a
        # termination. So, we explicitly sync on termination.
        for (local_output_dir, s3_dir_path) in local_output_dir_and_s3_path:
            sio.write(
                "aws s3 cp --recursive {local_dir} {s3_dir}\n".format(
                    local_dir=local_output_dir, s3_dir=s3_dir_path
                )
            )
        sio.write(
            "aws s3 cp /home/ubuntu/user_data.log {}\n".format(
                stdout_log_s3_path,
            )
        )

        # Wait for last sync
        if max_sync_interval > 0:
            sio.write("sleep {}\n".format(max_sync_interval + 5))

        if self.terminate:
            sio.write(
                """
                EC2_INSTANCE_ID="`wget -q -O - http://169.254.169.254/latest/meta-data/instance-id || die \"wget instance-id has failed: $?\"`"
                aws ec2 terminate-instances --instance-ids $EC2_INSTANCE_ID --region {aws_region}
            """.format(
                    aws_region=self.region
                )
            )
        sio.write("} >> /home/ubuntu/user_data.log 2>&1\n")

        full_script = dedent(sio.getvalue())
        import boto3
        import botocore

        ec2 = boto3.client(
            "ec2",
            region_name=self.region,
            aws_access_key_id=self.credentials.aws_key,
            aws_secret_access_key=self.credentials.aws_secret_key,
        )

        if (
            len(full_script) > 10000
            or len(base64.b64encode(full_script.encode()).decode("utf-8")) > 10000
        ):
            s3_path = self.upload_file_to_s3(full_script, dry=dry)
            sio = StringIO()
            sio.write("#!/bin/bash\n")
            sio.write(
                """
            aws s3 cp {s3_path} /home/ubuntu/remote_script.sh --region {aws_region} && \\
            chmod +x /home/ubuntu/remote_script.sh && \\
            bash /home/ubuntu/remote_script.sh
            """.format(
                    s3_path=s3_path, aws_region=self.s3_bucket_region
                )
            )
            user_data = dedent(sio.getvalue())
        else:
            user_data = full_script

        if verbose:
            print(full_script)
            with open("/tmp/full_ec2_script", "w") as f:
                f.write(full_script)

        instance_args = dict(
            ImageId=aws_config["image_id"],
            KeyName=aws_config["key_name"],
            UserData=user_data,
            InstanceType=aws_config["instance_type"],
            EbsOptimized=False,
            SecurityGroups=aws_config["security_groups"],
            SecurityGroupIds=aws_config["security_group_ids"],
            NetworkInterfaces=aws_config["network_interfaces"],
            IamInstanceProfile=dict(
                Name=aws_config["iam_instance_profile_name"],
            ),
            # **config.AWS_EXTRA_CONFIGS,
        )
        if self.extra_ec2_instance_kwargs is not None:
            instance_args.update(self.extra_ec2_instance_kwargs)

        if verbose:
            print("************************************************************")
            print("UserData:", instance_args["UserData"])
            print("************************************************************")
        instance_args["UserData"] = base64.b64encode(
            instance_args["UserData"].encode()
        ).decode("utf-8")
        spot_args = dict(
            DryRun=dry,
            InstanceCount=1,
            LaunchSpecification=instance_args,
            SpotPrice=aws_config["spot_price"],
            # ClientToken=params_list[0]["exp_name"],
        )

        import pprint

        if verbose:
            pprint.pprint(spot_args)
        if not dry:
            response = ec2.request_spot_instances(**spot_args)
            print("Launched EC2 job - Server response:")
            pprint.pprint(response)
            print("*****" * 5)
            spot_request_id = response["SpotInstanceRequests"][0][
                "SpotInstanceRequestId"
            ]
            for _ in range(10):
                try:
                    ec2.create_tags(
                        Resources=[spot_request_id],
                        Tags=[{"Key": "Name", "Value": exp_name}],
                    )
                    break
                except botocore.exceptions.ClientError:
                    continue


class EC2AutoconfigDocker(EC2SpotDocker):
    def __init__(
        self,
        region="us-west-1",
        s3_bucket=None,
        image_id=None,
        aws_key_name=None,
        iam_profile=None,
        **kwargs
    ):
        # find config file
        from doodad.ec2.autoconfig import AUTOCONFIG
        from doodad.ec2.credentials import AWSCredentials

        s3_bucket = AUTOCONFIG.s3_bucket() if s3_bucket is None else s3_bucket
        image_id = AUTOCONFIG.aws_image_id(region) if image_id is None else image_id
        aws_key_name = (
            AUTOCONFIG.aws_key_name(region) if aws_key_name is None else aws_key_name
        )
        iam_profile = (
            AUTOCONFIG.iam_profile_name() if iam_profile is None else iam_profile
        )
        credentials = AWSCredentials(
            aws_key=AUTOCONFIG.aws_access_key(),
            aws_secret=AUTOCONFIG.aws_access_secret(),
        )
        security_group_ids = AUTOCONFIG.aws_security_group_ids()[region]
        security_groups = AUTOCONFIG.aws_security_groups()

        super(EC2AutoconfigDocker, self).__init__(
            s3_bucket=s3_bucket,
            image_id=image_id,
            aws_key_name=aws_key_name,
            iam_instance_profile_name=iam_profile,
            credentials=credentials,
            region=region,
            security_groups=security_groups,
            security_group_ids=security_group_ids,
            **kwargs
        )


class GCPDocker(DockerMode):
    def __init__(
        self,
        zone="us-east4-a",
        gcp_bucket_name=None,
        instance_type="n1-standard-4",
        image_name=None,
        image_project=None,
        disk_size=64,  # Gb
        num_exps=1,
        terminate=True,
        preemptible=True,
        gcp_log_prefix="experiment",
        gcp_log_name=None,
        gcp_log_path=None,
        gpu_kwargs=None,
        **kwargs
    ):
        super(GCPDocker, self).__init__(**kwargs)
        assert "CLOUDSDK_CORE_PROJECT" in os.environ.keys()
        self.project = os.environ["CLOUDSDK_CORE_PROJECT"]
        self.zone = zone
        self.gcp_bucket_name = gcp_bucket_name
        self.instance_type = instance_type
        self.terminate = terminate
        self.disk_size = disk_size
        self.image_project = image_project
        self.image_name = image_name
        self.preemptible = preemptible
        self.num_exps = num_exps

        self.gcp_log_prefix = gcp_log_prefix
        self.gcp_log_name = gcp_log_name
        self.gcp_log_path = gcp_log_path or "doodad/logs"
        if self.gpu:
            self.num_gpu = gpu_kwargs["num_gpu"]
            self.gpu_model = gpu_kwargs["gpu_model"]
            self.gpu_type = get_gpu_type(self.project, self.zone, self.gpu_model)

        import googleapiclient.discovery

        self.compute = googleapiclient.discovery.build("compute", "v1")

    def launch_command(self, main_cmd, mount_points=None, dry=False, verbose=False):
        if self.gcp_log_name is None:
            exp_name = "{}-{}".format(
                self.gcp_log_prefix, EC2SpotDocker.make_timekey(self)
            )
        else:
            exp_name = self.gcp_log_name
        exp_prefix = self.gcp_log_prefix
        gcp_base_dir = os.path.join(
            self.gcp_log_path, exp_prefix.replace("_", "-"), exp_name
        )

        mnt_args = ""
        py_path = []
        gcp_mount_info = []
        local_mounts = []
        for mount in mount_points:
            print("Handling mount: ", mount)
            if isinstance(mount, MountLocal):  # TODO: these should be mount_s3 objects
                if mount.read_only:
                    if mount.path_on_remote is None:
                        with mount.gzip() as gzip_file:
                            gzip_path = os.path.realpath(gzip_file)
                            file_hash = utils.hash_file(gzip_path)
                            gcp_path = upload_file_to_gcp_storage(
                                bucket_name=self.gcp_bucket_name,
                                file_name=gzip_path,
                                remote_filename=file_hash + ".tar",
                            )
                        mount.path_on_remote = gcp_path
                        mount.local_file_hash = file_hash
                    else:
                        file_hash = mount.local_file_hash
                        gcp_path = mount.path_on_remote
                    remote_unpack_name = "/tmp/" + file_hash
                    mount_point = os.path.join(
                        "/mounts", mount.mount_point.replace("~/", "")
                    )
                    mnt_args += " -v %s:%s" % (
                        os.path.join(
                            remote_unpack_name, os.path.basename(mount.local_dir)
                        ),
                        mount_point,
                    )
                    if mount.pythonpath:
                        py_path.append(mount_point)
                    local_mounts.append(file_hash)
                else:
                    raise ValueError()
            elif isinstance(mount, MountGCP):
                gcp_local_dir = mount.mount_point
                gcp_path = os.path.join(gcp_base_dir, mount.gcp_path)
                if not mount.output:
                    raise NotImplementedError()
                gcp_mount_info.append(
                    (gcp_local_dir, gcp_path, mount.include_string, mount.sync_interval)
                )
                mnt_args += " -v %s:%s" % (gcp_local_dir, mount.mount_point)
            else:
                raise NotImplementedError()

        docker_cmd = self.get_docker_cmd(
            main_cmd,
            use_tty=False,
            extra_args=mnt_args,
            pythonpath=py_path,
            use_docker_generated_name=True,
        )

        metadata = {
            "bucket_name": self.gcp_bucket_name,
            "docker_cmd": docker_cmd,
            "docker_image": self.docker_image,
            "local_mounts": json.dumps(local_mounts),
            "gcp_mounts": json.dumps(gcp_mount_info),
            "use_gpu": json.dumps(self.gpu),
            "num_exps": self.num_exps,
            "terminate": json.dumps(self.terminate),
            "startup-script": open(GCP_STARTUP_SCRIPT_PATH, "r").read(),
            "shutdown-script": open(GCP_SHUTDOWN_SCRIPT_PATH, "r").read(),
        }
        # instance name must match regex'(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?)'">
        unique_name = "doodad" + str(uuid.uuid4()).replace("-", "")
        self.create_instance(metadata, unique_name, exp_name, exp_prefix)
        if verbose:
            print(unique_name)
            print(metadata)

    def create_instance(self, metadata, name, exp_name="", exp_prefix=""):
        image_response = (
            self.compute.images()
            .get(
                project=self.image_project,
                image=self.image_name,
            )
            .execute()
        )
        source_disk_image = image_response["selfLink"]
        config = {
            "name": name,
            "machineType": get_machine_type(self.zone, self.instance_type),
            "disks": [
                {
                    "boot": True,
                    "autoDelete": True,
                    "initializeParams": {
                        "sourceImage": source_disk_image,
                        "diskSizeGb": self.disk_size,
                    },
                }
            ],
            "networkInterfaces": [
                {
                    "network": "global/networks/default",
                    "accessConfigs": [
                        {"type": "ONE_TO_ONE_NAT", "name": "External NAT"}
                    ],
                }
            ],
            "serviceAccounts": [
                {
                    "email": "default",
                    "scopes": ["https://www.googleapis.com/auth/cloud-platform"],
                }
            ],
            "metadata": {
                "items": [
                    {"key": key, "value": value} for key, value in metadata.items()
                ]
            },
            "scheduling": {
                "onHostMaintenance": "terminate",
                "automaticRestart": False,
                "preemptible": self.preemptible,
            },
            "labels": {
                "exp_name": exp_name,
                "exp_prefix": exp_prefix,
            },
        }
        if self.gpu:
            config["guestAccelerators"] = [
                {
                    "acceleratorType": self.gpu_type,
                    "acceleratorCount": self.num_gpu,
                }
            ]
        return (
            self.compute.instances()
            .insert(project=self.project, zone=self.zone, body=config)
            .execute()
        )


class CodalabDocker(DockerMode):
    def __init__(self):
        super(CodalabDocker, self).__init__()
        raise NotImplementedError()


class SingularityMode(LaunchMode):
    def __init__(
        self,
        image,
        gpu=False,
        pre_cmd=None,
        post_cmd=None,
        extra_args="",
        verbose_cmd=False,
    ):
        super(SingularityMode, self).__init__()
        self.singularity_image = image
        self.gpu = gpu
        self.pre_cmd = pre_cmd
        self.post_cmd = post_cmd
        self._extra_args = extra_args
        self._verbose_cmd = verbose_cmd

    def create_singularity_cmd(
        self,
        main_cmd,
        mount_points=None,
    ):
        extra_args = self._extra_args
        cmd_list = utils.CommandBuilder()
        if self.pre_cmd:
            cmd_list.extend(self.pre_cmd)

        if self._verbose_cmd:
            if self.gpu:
                cmd_list.append('echo "Running in singularity (gpu)"')
            else:
                cmd_list.append('echo "Running in singularity"')

        py_paths = []
        for mount in mount_points:
            if isinstance(mount, MountLocal):
                if mount.pythonpath:
                    py_paths.append(mount.local_dir)
            else:
                raise NotImplementedError(type(mount))
        if py_paths:
            cmd_list.append("export PYTHONPATH=$PYTHONPATH:%s" % (":".join(py_paths)))

        cmd_list.append(main_cmd)
        if self.post_cmd:
            cmd_list.extend(self.post_cmd)

        if self.gpu:
            extra_args += " --nv "
        singularity_prefix = "/opt/singularity/bin/singularity exec %s %s /bin/bash -c " % (
            extra_args,
            self.singularity_image,
        )
        main_cmd = cmd_list.to_string()
        full_cmd = singularity_prefix + ("'%s'" % main_cmd)
        return full_cmd


class LocalSingularity(SingularityMode):
    def __init__(self, *args, skip_wait=False, **kwargs):
        super(LocalSingularity, self).__init__(*args, **kwargs)
        self.skip_wait = skip_wait

    def launch_command(self, cmd, mount_points=None, dry=False, verbose=False):
        full_cmd = self.create_singularity_cmd(cmd, mount_points=mount_points)
        utils.call_and_wait(
            full_cmd, verbose=verbose, dry=dry, skip_wait=self.skip_wait
        )


class BrcHighThroughputMode(SingularityMode):
    """
    Create or add to a script to run a bunch of slurm jobs.
    """

    TASK_FILE = "/tmp/taskfile_from_doodad.sh"
    SBATCH_FILE = "/tmp/script_to_scp_over.sh"

    def __init__(
        self,
        slurm_config,
        taskfile_path_on_brc,
        n_tasks_total,
        overwrite_task_script=False,
        *args,
        **kwargs
    ):
        super(BrcHighThroughputMode, self).__init__(*args, **kwargs)
        self._overwrite_task_script = overwrite_task_script
        self._taskfile_path_on_brc = taskfile_path_on_brc
        self._slurm_config = slurm_config
        self._n_tasks_total = n_tasks_total

    def launch_command(
        self,
        cmd,
        dry=False,
        mount_points=None,
        verbose=False,
    ):
        full_cmd = self.create_singularity_cmd(cmd, mount_points=mount_points)
        utils.add_to_script(
            full_cmd,
            path=self.TASK_FILE,
            verbose=True,
            overwrite=self._overwrite_task_script,
        )

        cmd_list = utils.CommandBuilder()
        cmd_list.append("module load gcc openmpi")
        cmd_list.append(
            'ht_helper.sh -m "python/3.5" -t {}'.format(self._taskfile_path_on_brc)
        )
        sbatch_cmd = slurm_util.wrap_command_with_sbatch(
            cmd_list.to_string(),
            self._slurm_config,
            self._n_tasks_total,
        )
        utils.add_to_script(
            sbatch_cmd,
            path=self.SBATCH_FILE,
            verbose=True,
            overwrite=True,
        )


class SlurmSingularity(SingularityMode):
    # TODO: set up an auto-config
    def __init__(self, image, slurm_config: SlurmConfig, skip_wait=False, **kwargs):
        super(SlurmSingularity, self).__init__(image, **kwargs)
        self._slurm_config = slurm_config
        self.skip_wait = skip_wait

    def launch_command(self, cmd, mount_points=None, dry=False, verbose=False):
        full_cmd = self.create_slurm_command(
            cmd,
            mount_points=mount_points,
        )
        utils.call_and_wait(
            full_cmd, verbose=verbose, dry=dry, skip_wait=self.skip_wait
        )

    def create_slurm_command(self, cmd, mount_points=None):
        singularity_cmd = self.create_singularity_cmd(
            cmd,
            mount_points=mount_points,
        )
        full_cmd = slurm_util.wrap_command_with_sbatch(
            singularity_cmd,
            self._slurm_config,
            n_tasks=1,
        )
        return full_cmd


class SlurmSingularityMatrix(SingularityMode):
    def __init__(
        self, image, slurm_config: SlurmConfigMatrix, logdir, skip_wait=False, **kwargs
    ):
        super(SlurmSingularityMatrix, self).__init__(image, **kwargs)
        self._slurm_config = slurm_config
        self.skip_wait = skip_wait
        self.logdir = logdir

    def launch_command(self, cmd, mount_points=None, dry=False, verbose=False):
        full_cmd = self.create_slurm_command(
            cmd,
            mount_points=mount_points,
        )
        utils.call_and_wait(
            full_cmd, verbose=verbose, dry=dry, skip_wait=self.skip_wait
        )

    def create_slurm_command(self, cmd, mount_points=None):
        singularity_cmd = self.create_singularity_cmd(
            cmd,
            mount_points=mount_points,
        )
        full_cmd = slurm_util.wrap_command_with_sbatch_matrix(
            singularity_cmd,
            self._slurm_config,
            logdir=self.logdir,
        )
        return full_cmd


class ScriptSlurmSingularity(SlurmSingularity):
    """
    Create or add to a script to run a bunch of slurm jobs.
    """

    TMP_FILE = "/tmp/script_to_scp_over.sh"

    def __init__(self, image, slurm_config, overwrite_script=False, **kwargs):
        super().__init__(image, slurm_config, **kwargs)
        self._overwrite_script = overwrite_script

    def launch_command(
        self,
        cmd,
        dry=False,
        mount_points=None,
        verbose=True,
    ):
        full_cmd = self.create_slurm_command(cmd, mount_points=mount_points)
        # full_cmd = self.create_singularity_cmd(cmd, mount_points=mount_points)
        utils.add_to_script(
            full_cmd,
            path=self.TMP_FILE,
            verbose=True,
            overwrite=self._overwrite_script,
        )
