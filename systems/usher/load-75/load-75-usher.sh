#!/bin/bash

git_dir=$(git rev-parse --show-toplevel)
cd ${git_dir}

freq=1410
device=0

# Model-BatchSize-MIGSlice-NumReqs-LoadStart-LoadEnd-LoadSteps-Rps
mixes=(
"diffusion_256_256-1-43-83-0.75-0.75-0.25-0.41497842657583255 gpt-1-34-40-0.75-0.75-0.25-0.16192860916956147 mobilenet_v3_large-2-17-176686-0.75-0.75-0.25-981.59"
"mobilenet_v3_large-2-17-176686-0.75-0.75-0.25-981.59 whisper-1-20-708-0.75-0.75-0.25-3.29843852958 resnet50-4-26-250380-0.75-0.75-0.25-1391.97 resnet152-4-24-50104-0.75-0.75-0.25-278.36"
"whisper-1-20-708-0.75-0.75-0.25-3.29843852958 resnet50-4-26-250380-0.75-0.75-0.25-1391.97 densenet161-4-26-42154-0.75-0.75-0.25-234.19 densenet169-8-25-45732-0.75-0.75-0.25-254.07"
"diffusion_1024_1024-1-51-29-0.75-0.75-0.25-0.0923146137935218 vgg19-1-41-30517-0.75-0.75-0.25-169.54"
"vgg11-1-40-134037-0.75-0.75-0.25-744.65"
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
