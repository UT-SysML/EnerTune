#!/bin/bash

git_dir=$(git rev-parse --show-toplevel)
cd ${git_dir}

freq=1410
device=3

# Model-BatchSize-MIGSlice-NumReqs-LoadStart-LoadEnd-LoadSteps-Rps
mixes=(
# "whisper-1-4-708-0.25-1.0-0.25-3.9362878866995845 resnet50-1-3-250380-0.25-1.0-0.25-1391.97"
"resnet50-1-4-250380-0.25-1.25-0.25-1391.97 whisper-16-3-708-0.25-1.25-0.25-3.9362878866995845"
)

for mix in "${mixes[@]}"; do
    echo "System: FGD | Running mix: $mix | Device: $device"
    ./run.sh \
      --device-type a100 \
      --device-id ${device} \
      --modes custom-mig \
      --distribution poisson \
      --min-freq ${freq} \
      --max-freq ${freq} \
      ${mix}
    sleep 10
    echo "Completed ${mix} on GPU ${device} with ${freq} MHz for system FGD."
    echo""
done
