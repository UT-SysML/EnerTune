#!/bin/bash

git_dir=$(git rev-parse --show-toplevel)
cd ${git_dir}

freq=1410
device=2

# Model-BatchSize-MIGSlice-NumReqs-LoadStart-LoadEnd-LoadSteps-Rps
mixes=(
"densenet161-1-17-42154-0.5-0.5-0.25-234.19 resnet50-2-22-250380-0.5-0.5-0.25-1391.97 resnet152-2-20-50104-0.5-0.5-0.25-278.36 vgg11-1-40-134037-0.5-0.5-0.25-744.65" # 50% Job Mix Plan
# "diffusion_1024_1024-1-51-29-0.5-0.5-0.25-0.0923146137935218 resnet50-8-47-250380-0.5-0.5-0.25-1391.97" # 25% Job Mix Plan
)

for mix in "${mixes[@]}"; do
    echo "System: Usher | Running mix: $mix | Device: $device"
    ./run.sh \
      --device-type a100 \
      --device-id ${device} \
      --modes mps-manual-cap \
      --distribution poisson \
      --min-freq ${freq} \
      --max-freq ${freq} \
      ${mix}
    sleep 10
    echo "Completed ${mix} on GPU ${device} with ${freq} MHz for system Usher."
    echo""
done
