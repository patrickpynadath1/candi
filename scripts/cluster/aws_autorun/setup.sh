#!/usr/bin/env bash
# Laptop side of the self-healing spot runner: upload a run's inputs and (re)create its launch
# template and Auto Scaling group (spot only, any us-east-1 zone, 0..1 machines).
#
#   scripts/cluster/aws_autorun/setup.sh RUN SETS_FILE "TYPE[,TYPE...]"   # creates, desired 0
#   aws autoscaling set-desired-capacity --auto-scaling-group-name PREFIX-RUN --desired-capacity 1
#
# SETS_FILE lines: `<plan name> <sweep> [prepare overrides...]`. The group replaces an
# interrupted machine automatically; the machine resumes from S3 and sets the group to 0 when
# every case is done. Code is the working tree at setup time (tracked and new files, local edits).
#
# Several machines can share one run as slices: SDDC_AWS_SLICE=NAME SDDC_AWS_CASES=FILE (lines
# `PLAN/CASE`) adds a group PREFIX-RUN-NAME whose machine runs, restores and syncs only
# those cases; the run's code, run.env and sets.txt are reused, not replaced.
# SDDC_AWS_ON_DEMAND=1 makes the group on-demand instead of spot.
set -euo pipefail
run=${1:?RUN name} sets=${2:?SETS_FILE} types=${3:?instance types}
export AWS_REGION=${AWS_REGION:-us-east-1}
# Account-specific settings come from the environment (keep them in an untracked file):
#   SDDC_AWS_BUCKET   S3 bucket holding data_cache/, hf/, plans/, runs/, logs/, autorun/
#   SDDC_AWS_PROFILE  EC2 instance profile with access to that bucket (and, to let a
#                     machine turn its own group off, autoscaling:SetDesiredCapacity
#                     on PREFIX-* groups)
#   SDDC_AWS_SG       security group (SSH only from your address)
#   SDDC_AWS_KEY      EC2 key pair name
#   SDDC_AWS_PREFIX   name prefix for groups and launch templates (default sddc)
#   SDDC_AWS_OWNER    value of the Owner tag on everything created (default $USER)
bucket=${SDDC_AWS_BUCKET:?set SDDC_AWS_BUCKET}
profile=${SDDC_AWS_PROFILE:?set SDDC_AWS_PROFILE}
sg=${SDDC_AWS_SG:?set SDDC_AWS_SG}
key=${SDDC_AWS_KEY:?set SDDC_AWS_KEY}
prefix=${SDDC_AWS_PREFIX:-sddc}
owner=${SDDC_AWS_OWNER:-$USER}
ami=${SDDC_AWS_AMI:-$(aws ssm get-parameter --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id --query Parameter.Value --output text)}
slice=${SDDC_AWS_SLICE:-}
asg=$prefix-$run${slice:+-$slice}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../../.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# Inputs. A run's code is fixed once machines have used it: refuse to replace it.
if [ -n "$slice" ]; then
  aws s3 ls "s3://$bucket/autorun/$run/code.tar.gz" >/dev/null || { echo "slices need an existing run" >&2; exit 1; }
  printf 'ASG=%s\n' "$asg" > "$tmp/slice.env"
  aws s3 cp --only-show-errors "$tmp/slice.env" "s3://$bucket/autorun/$run/slices/$slice/run.env"
  aws s3 cp --only-show-errors "${SDDC_AWS_CASES:?SDDC_AWS_CASES for a slice}" "s3://$bucket/autorun/$run/slices/$slice/cases.txt"
elif aws s3 ls "s3://$bucket/autorun/$run/code.tar.gz" >/dev/null 2>&1 && [ -z "${SDDC_AWS_REPLACE_CODE:-}" ]; then
  echo "s3://$bucket/autorun/$run/code.tar.gz exists; keeping it (SDDC_AWS_REPLACE_CODE=1 to replace)"
else
  # Tracked and new (not ignored) files; never local notes, archives or other large untracked output.
  (cd "$repo" && git ls-files -co --exclude-standard \
    | grep -vE '^CLAUDE\.local\.md$|\.tar\.gz$|^jev-out/|^scripts/analysis/jev_ladder\.py$' \
    | tar czf "$tmp/code.tar.gz" -T -)
  aws s3 cp --only-show-errors "$tmp/code.tar.gz" "s3://$bucket/autorun/$run/code.tar.gz"
fi
if [ -z "$slice" ]; then
  dirty=$([ -n "$(cd "$repo" && git status --porcelain --untracked-files=no)" ] && echo true || echo false)
  printf 'ASG=%s\nSOURCE_REV=%s\nSOURCE_DIRTY=%s\n' "$asg" "$(cd "$repo" && git rev-parse --short=12 HEAD)" "$dirty" > "$tmp/run.env"
  aws s3 cp --only-show-errors "$tmp/run.env" "s3://$bucket/autorun/$run/run.env"
  aws s3 cp --only-show-errors "$sets" "s3://$bucket/autorun/$run/sets.txt"
fi
aws s3 cp --only-show-errors "$here/run.sh" "s3://$bucket/autorun/$run/run.sh"
aws s3 cp --only-show-errors "$here/restore_mtimes.py" "s3://$bucket/autorun/$run/restore_mtimes.py"

# Boot script: fetch run.sh and start it as ubuntu, logging to /var/log/sddc-autorun/run.log.
cat > "$tmp/user-data" <<EOF
#!/bin/bash
mkdir -p /var/log/sddc-autorun && chown ubuntu:ubuntu /var/log/sddc-autorun
aws s3 cp --region $AWS_REGION s3://$bucket/autorun/$run/run.sh /home/ubuntu/run.sh
chmod +x /home/ubuntu/run.sh && chown ubuntu:ubuntu /home/ubuntu/run.sh
systemd-run --unit sddc-autorun --uid ubuntu --setenv=HOME=/home/ubuntu --setenv=AWS_REGION=$AWS_REGION \
  /bin/bash -c '/home/ubuntu/run.sh $bucket $run $slice >> /var/log/sddc-autorun/run.log 2>&1'
EOF

# Launch template (a new version on every setup).
python3 - "$tmp" "$ami" "$profile" "$sg" "$key" "$asg" "$owner" <<'EOF'
import base64, json, sys
tmp, ami, profile, sg, key, name, owner = sys.argv[1:]
tags = [{"Key": "Owner", "Value": owner}, {"Key": "Name", "Value": name}]
data = {
    "ImageId": ami, "KeyName": key, "SecurityGroupIds": [sg],
    "IamInstanceProfile": {"Name": profile},
    "MetadataOptions": {"HttpTokens": "required", "HttpEndpoint": "enabled"},
    "BlockDeviceMappings": [{"DeviceName": "/dev/sda1",
                             "Ebs": {"VolumeSize": 200, "VolumeType": "gp3", "DeleteOnTermination": True}}],
    "UserData": base64.b64encode(open(f"{tmp}/user-data", "rb").read()).decode(),
    "TagSpecifications": [{"ResourceType": t, "Tags": tags} for t in ("instance", "volume")],
}
json.dump(data, open(f"{tmp}/template.json", "w"))
EOF
lt=$asg
if aws ec2 describe-launch-templates --launch-template-names "$lt" >/dev/null 2>&1; then
  version=$(aws ec2 create-launch-template-version --launch-template-name "$lt" \
    --launch-template-data "file://$tmp/template.json" --query LaunchTemplateVersion.VersionNumber --output text)
  aws ec2 modify-launch-template --launch-template-name "$lt" --default-version "$version" >/dev/null
else
  aws ec2 create-launch-template --launch-template-name "$lt" --launch-template-data "file://$tmp/template.json" \
    --tag-specifications "ResourceType=launch-template,Tags=[{Key=Owner,Value=$owner}]" >/dev/null
fi

# Auto Scaling group: spot only, every default subnet, the given types in priority-free capacity order.
subnets=$(aws ec2 describe-subnets --filters Name=default-for-az,Values=true --query 'Subnets[].SubnetId' --output text | tr '\t' ',')
overrides=$(python3 -c "import json,sys; print(json.dumps([{'InstanceType': t} for t in sys.argv[1].split(',')]))" "$types")
policy=$(python3 - "$lt" "$overrides" "${SDDC_AWS_ON_DEMAND:-}" <<'EOF'
import json, sys
on_demand = 100 if sys.argv[3] else 0
print(json.dumps({
    "LaunchTemplate": {"LaunchTemplateSpecification": {"LaunchTemplateName": sys.argv[1], "Version": "$Default"},
                       "Overrides": json.loads(sys.argv[2])},
    "InstancesDistribution": {"OnDemandBaseCapacity": 0, "OnDemandPercentageAboveBaseCapacity": on_demand,
                              "SpotAllocationStrategy": "price-capacity-optimized"},
}))
EOF
)
if aws autoscaling describe-auto-scaling-groups --auto-scaling-group-names "$asg" --query 'AutoScalingGroups[0].AutoScalingGroupName' --output text | grep -q "$asg"; then
  aws autoscaling update-auto-scaling-group --auto-scaling-group-name "$asg" --mixed-instances-policy "$policy" \
    --vpc-zone-identifier "$subnets" --min-size 0 --max-size 1
else
  aws autoscaling create-auto-scaling-group --auto-scaling-group-name "$asg" --mixed-instances-policy "$policy" \
    --vpc-zone-identifier "$subnets" --min-size 0 --max-size 1 --desired-capacity 0 \
    --tags "Key=Owner,Value=$owner,PropagateAtLaunch=false"
fi
echo "ready: $asg (desired 0). Start: aws autoscaling set-desired-capacity --auto-scaling-group-name $asg --desired-capacity 1"
echo "logs:  aws s3 ls s3://$bucket/logs/$run/   results: s3://$bucket/runs/$run/"
