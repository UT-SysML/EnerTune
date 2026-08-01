#!/bin/bash

git_dir=$(git rev-parse --show-toplevel)
cd ${git_dir}

freq=1410
device=0

# Model-BatchSize-MIGSlice-NumReqs-LoadStart-LoadEnd-LoadSteps-Rps
mixes=(
"densenet161-4-3-42154-0.25-0.25-0.25-234.19"
"diffusion_256_256-1-4-83-0.25-0.25-0.25-0.46497842657583255 diffusion_1024_1024-1-3-29-0.25-0.25-0.25-0.16060091049904315"
# "diffusion_1024_1024-1-4-29-0.25-1.0-0.25-0.16060091049904315 diffusion_256_256-1-3-83-0.25-1.0-0.25-0.46497842657583255"
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
