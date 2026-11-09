#!/usr/bin/env bash
# Start, stop and check one EC2 GPU instance used as the sddc-remote server.
#
#   SDDC_AWS_INSTANCE=i-... scripts/cluster/aws.sh status|start|stop
#
# start: boots the instance, rewrites the HostName of the ssh alias ($SDDC_REMOTE_HOST,
#        default aws-gpu) in ~/.ssh/config to the new public IP, and restarts the Ray head
#        in tmux. stop: stops the instance (compute billing ends; EBS disks are kept).
# Only the instance named by SDDC_AWS_INSTANCE is ever touched.
set -euo pipefail

id=${SDDC_AWS_INSTANCE:?Set SDDC_AWS_INSTANCE to your instance id}
host=${SDDC_REMOTE_HOST:-aws-gpu}
root=${SDDC_REMOTE_STORAGE_ROOT:-/data/sddc}
gpus=${GPUS:-0}
export AWS_REGION=${AWS_REGION:-us-east-1}

state() {
  aws ec2 describe-instances --instance-ids "$id" \
    --query 'Reservations[0].Instances[0].[State.Name,InstanceType,PublicIpAddress]' --output text
}

case ${1:-status} in
  status)
    state
    ;;
  stop)
    aws ec2 stop-instances --instance-ids "$id" --query 'StoppingInstances[0].CurrentState.Name' --output text
    aws ec2 wait instance-stopped --instance-ids "$id"
    state
    ;;
  start)
    aws ec2 start-instances --instance-ids "$id" --query 'StartingInstances[0].CurrentState.Name' --output text
    aws ec2 wait instance-running --instance-ids "$id"
    ip=$(aws ec2 describe-instances --instance-ids "$id" \
      --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
    # Rewrite HostName inside the "Host $host" block only.
    awk -v h="$host" -v ip="$ip" '
      $1 == "Host" { inblock = ($2 == h) }
      inblock && $1 == "HostName" { sub(/HostName.*/, "HostName " ip) }
      { print }' ~/.ssh/config > ~/.ssh/config.tmp && mv ~/.ssh/config.tmp ~/.ssh/config
    for _ in $(seq 1 20); do ssh -o ConnectTimeout=5 "$host" true 2>/dev/null && break; sleep 6; done
    ssh "$host" "tmux has-session -t sddc-head 2>/dev/null || (cd $root/sddc-head && \
      tmux new -d -s sddc-head 'SDDC_STORAGE_ROOT=$root GPUS=$gpus CPUS=\$(nproc) scripts/cluster/start_head.sh 2>&1 | tee $root/head.log')"
    echo "running at $ip; Ray head starting in tmux session sddc-head (log: $root/head.log)"
    ;;
  *)
    echo "usage: $0 status|start|stop" >&2; exit 2
    ;;
esac
