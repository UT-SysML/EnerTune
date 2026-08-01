#!/bin/bash

git_dir=$(git rev-parse --show-toplevel)
cd ${git_dir}

freq=1410
device=3

# Model-BatchSize-MIGSlice-NumReqs-LoadStart-LoadEnd-LoadSteps-Rps
mixes=(
# "diffusion_1024_1024-1-51-29-1.25-1.25-0.25-0.0923146137935218 vgg19-1-41-30517-1.25-1.25-0.25-169.54" # 75% and 100% Job Mix Plan (GPU 3)
# "vgg11-1-40-134037-1.25-1.25-0.25-744.65" # 75% and 100% Job Mix Plan (GPU 4)
"resnet152-8-45-50104-1.25-1.25-0.25-278.36 densenet161-8-40-42154-1.25-1.25-0.25-234.19"
"gpt-1-45-40-1.25-1.25-0.25-0.16192860916956147 mobilenet_v3_large-8-35-176686-1.25-1.25-0.25-981.59"
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
