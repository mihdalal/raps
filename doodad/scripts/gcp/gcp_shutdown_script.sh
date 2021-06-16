#!/bin/bash
query_metadata() {
    attribute_name=$1
    curl http://metadata/computeMetadata/v1/instance/attributes/$attribute_name -H "Metadata-Flavor: Google"
}

bucket_name=$(query_metadata bucket_name)
gcp_mounts=$(query_metadata gcp_mounts)
instance_name=$(curl http://metadata/computeMetadata/v1/instance/name -H "Metadata-Flavor: Google")

num_gcp_mounts=$(jq length <<< $gcp_mounts)
for ((i=0;i<$num_gcp_mounts;i++)); do
    gcp_mount_info=$(jq .[$i] <<< $gcp_mounts)
    # assume gcp_mount_info is a (local_path, bucket_path, include_string, periodic_sync_interval) tuple
    local_path=$(jq .[0] <<< $gcp_mount_info | tr -d '"')
    gcp_bucket_path=$(jq .[1] <<< $gcp_mount_info | tr -d '"')
    gsutil -m rsync -r $local_path gs://$bucket_name/$gcp_bucket_path
done

gsutil cp /home/ubuntu/user_data.log gs://$bucket_name/$gcp_bucket_path/${instance_name}_stdout.log
