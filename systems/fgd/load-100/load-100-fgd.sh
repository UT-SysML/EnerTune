#!/bin/bash

git_dir=$(git rev-parse --show-toplevel)
cd ${git_dir}

freq=1410
device=0

# Model-BatchSize-MIGSlice-NumReqs-LoadStart-LoadEnd-LoadSteps-Rps
mixes=(
"densenet161-4-3-42154-1.0-1.0-0.25-234.19"
"diffusion_256_256-1-4-83-1.0-1.0-0.25-0.46497842657583255 diffusion_1024_1024-1-3-29-1.0-1.0-0.25-0.16060091049904315"
"vgg19-2-2-30517-1.0-1.0-0.25-169.54 efficientnet_b5-2-2-37353-1.0-1.0-0.25-207.52 inception_v3-2-1-34065-1.0-1.0-0.25-189.25 mobilenet_v3_large-16-1-176686-1.0-1.0-0.25-981.59"
"resnet152-4-4-50104-1.0-1.0-0.25-278.36 densenet169-4-3-45372-1.0-1.0-0.25-254.07"
"gpt-1-4-40-1.0-1.0-0.25-0.22192860916956147 vgg11-4-3-134037-1.0-1.0-0.25-744.65"
"resnet50-1-4-250380-1.0-1.0-0.25-1391.97 whisper-16-3-708-1.0-1.0-0.25-3.9362878866995845"
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
