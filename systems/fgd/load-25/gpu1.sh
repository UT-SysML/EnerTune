#!/bin/bash

git_dir=$(git rev-parse --show-toplevel)
cd ${git_dir}

freq=1410
device=1

# Model-BatchSize-MIGSlice-NumReqs-LoadStart-LoadEnd-LoadSteps-Rps
mixes=(
"vgg19-2-2-30517-0.25-0.25-0.25-169.54 efficientnet_b5-2-2-37353-0.25-0.25-0.25-207.52 inception_v3-2-1-34065-0.25-0.25-0.25-189.25 mobilenet_v3_large-16-1-176686-0.25-0.25-0.25-981.59"
"resnet152-4-4-50104-0.25-0.25-0.25-278.36 densenet169-4-3-45372-0.25-0.25-0.25-254.07"
)

for mix in "${mixes[@]}"; do
    echo "System: FGD | Running mix: $mix | Device: $device"
    ./run.sh \
      --device-type a100 \
      --device-id ${device} \
      --modes custom-mig \
      --distribution point \
      --min-freq ${freq} \
      --max-freq ${freq} \
      ${mix}
    sleep 10
    echo "Completed ${mix} on GPU ${device} with ${freq} MHz for system FGD."
    echo""
done
