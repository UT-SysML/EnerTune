#!/bin/bash

git_dir=$(git rev-parse --show-toplevel)
cd ${git_dir}

freq=1410
device=2

# Model-BatchSize-MIGSlice-NumReqs-LoadStart-LoadEnd-LoadSteps-Rps
mixes=(
"diffusion_1024_1024-1-51-29-0.25-0.25-0.25-0.0923146137935218 resnet50-8-47-250380-0.25-0.25-0.25-1391.97" # 25% Job Mix Plan
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
