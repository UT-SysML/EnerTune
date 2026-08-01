#!/bin/bash

git_dir=$(git rev-parse --show-toplevel)
cd ${git_dir}

freq=1410
device=1

# Model-BatchSize-MIGSlice-NumReqs-LoadStart-LoadEnd-LoadSteps-Rps
mixes=(
# "mobilenet_v3_large-2-17-176686-1.25-1.25-0.25-981.59 whisper-1-20-708-1.25-1.25-0.25-3.29843852958 resnet50-4-26-250380-1.25-1.25-0.25-1391.97 resnet152-4-24-50104-1.25-1.25-0.25-278.36" # 75% and 100% Job Mix Plan
"vgg19-8-60-30517-1.25-1.25-0.25-169.54 densenet169-8-40-45732-1.25-1.25-0.25-254.07 whisper-1-20-708-1.25-1.25-0.25-3.29843852958"
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
