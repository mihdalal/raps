import doodad as dd

if __name__ == "__main__":
    import matplotlib

    matplotlib.use("agg")

    args_dict = dd.get_args()
    method_call = args_dict["method_call"]
    doodad_config = args_dict["doodad_config"]
    variant = args_dict["variant"]
    output_dir = args_dict["output_dir"]
    run_mode = args_dict.get("mode", None)
    if run_mode and run_mode in ["slurm_singularity", "sss", "htp"]:
        import os

        doodad_config.extra_launch_info["slurm-job-id"] = os.environ.get(
            "SLURM_JOB_ID", None
        )
    if run_mode and (run_mode == "ec2" or run_mode == "gcp"):
        if run_mode == "ec2":
            try:
                import urllib.request

                instance_id = (
                    urllib.request.urlopen(
                        "http://169.254.169.254/latest/meta-data/instance-id"
                    )
                    .read()
                    .decode()
                )
                doodad_config.extra_launch_info["EC2_instance_id"] = instance_id
            except Exception as e:
                print("Could not get AWS instance ID. Error was...")
                print(e)
        if run_mode == "gcp":
            try:
                import urllib.request

                request = urllib.request.Request(
                    "http://metadata/computeMetadata/v1/instance/name",
                )
                # See this URL for why we need this header:
                # https://cloud.google.com/compute/docs/storing-retrieving-metadata
                request.add_header("Metadata-Flavor", "Google")
                instance_name = urllib.request.urlopen(request).read().decode()
                doodad_config.extra_launch_info["GCP_instance_name"] = instance_name
            except Exception as e:
                print("Could not get GCP instance name. Error was...")
                print(e)
        # Do this in case base_log_dir was already set
        doodad_config = doodad_config._replace(
            base_log_dir=output_dir,
        )

    method_call(doodad_config, variant)
