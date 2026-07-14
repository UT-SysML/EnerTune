#!/bin/bash

git_dir=$(git rev-parse --show-toplevel)
cd ${git_dir}

freq=1410
device=1

# Model-BatchSize-MIGSlice-NumReqs-LoadStart-LoadEnd-LoadSteps-Rps
mixes=(
# "whisper-1-20-708-0.25-0.25-0.25-3.29843852958 resnet152-1-30-50104-0.25-0.25-0.25-278.36 densenet169-2-30-45732-0.25-0.25-0.25-254.07 densenet161-1-20-42154-0.25-0.25-0.25-234.19" # 25% Job Mix Plan
"vgg19-8-60-30517-0.25-0.25-0.25-169.54 densenet169-8-40-45732-0.25-0.25-0.25-254.07"
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
