#!/bin/bash

git_dir=$(git rev-parse --show-toplevel)
cd ${git_dir}

freq=1410
device=1

# Model-BatchSize-MIGSlice-NumReqs-LoadStart-LoadEnd-LoadSteps-Rps
mixes=(
# "mobilenet_v3_large-1-15-176686-0.5-0.5-0.25-981.59 whisper-1-20-708-0.5-0.5-0.25-3.29843852958 densenet169-2-19-45732-0.5-0.5-0.25-254.07 densenet161-1-17-42154-0.5-0.5-0.25-234.19 resnet50-2-22-250380-0.5-0.5-0.25-1391.97" # 50% Job Mix Plan
"vgg19-8-60-30517-0.5-0.5-0.25-169.54 densenet169-8-40-45732-0.5-0.5-0.25-254.07"
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
