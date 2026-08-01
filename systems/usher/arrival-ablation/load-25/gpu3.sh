#!/bin/bash

git_dir=$(git rev-parse --show-toplevel)
cd ${git_dir}

freq=1410
device=3

# Model-BatchSize-MIGSlice-NumReqs-LoadStart-LoadEnd-LoadSteps-Rps
mixes=(
# "vgg11-1-40-134037-0.25-0.25-0.25-744.65 resnet50-2-40-250380-0.25-0.25-0.25-1391.97" # 25% Job Mix Plan (GPU 3)
"resnet152-8-45-50104-0.25-0.25-0.25-278.36 densenet161-8-40-42154-0.25-0.25-0.25-234.19"
"gpt-1-45-40-0.25-0.25-0.25-0.16192860916956147 mobilenet_v3_large-8-35-176686-0.25-0.25-0.25-981.59"
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
