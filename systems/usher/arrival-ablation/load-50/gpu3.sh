#!/bin/bash

git_dir=$(git rev-parse --show-toplevel)
cd ${git_dir}

freq=1410
device=3

# Model-BatchSize-MIGSlice-NumReqs-LoadStart-LoadEnd-LoadSteps-Rps
mixes=(
# "diffusion_1024_1024-1-51-29-0.5-0.5-0.25-0.0923146137935218 vgg19-1-41-30517-0.5-0.5-0.25-169.54" # 50% Job Mix Plan (GPU 3)
"resnet152-8-45-50104-0.5-0.5-0.25-278.36 densenet161-8-40-42154-0.5-0.5-0.25-234.19"
"gpt-1-45-40-0.5-0.5-0.25-0.16192860916956147 mobilenet_v3_large-8-35-176686-0.5-0.5-0.25-981.59"
)

for mix in "${mixes[@]}"; do
    echo "System: Usher | Running mix: $mix | Device: $device"
    ./run.sh \
      --device-type a100 \
      --device-id ${device} \
      --modes mps-manual-cap \
      --distribution point \
      --min-freq ${freq} \
      --max-freq ${freq} \
      ${mix}
    sleep 10
    echo "Completed ${mix} on GPU ${device} with ${freq} MHz for system Usher."
    echo""
done
