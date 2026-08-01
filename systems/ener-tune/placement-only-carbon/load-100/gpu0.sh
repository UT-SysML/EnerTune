#!/bin/bash

git_dir=$(git rev-parse --show-toplevel)
cd ${git_dir}

freq=1410
device=0
system="EnerTune"

# Model-BatchSize-MIGSlice-NumReqs-LoadStart-LoadEnd-LoadSteps-Rps
mixes=(
"densenet169-4-3-45732-1.0-1.0-0.25-254.07 resnet50-16-3-250380-1.0-1.0-0.25-1391.97"
)

for mix in "${mixes[@]}"; do
    echo "System: ${system} | Running mix: $mix | Device: $device"
    ./run.sh \
      --device-type a100 \
      --device-id ${device} \
      --modes custom-mig \
      --distribution point \
      --min-freq ${freq} \
      --max-freq ${freq} \
      ${mix}
    sleep 10
    echo "Completed ${mix} on GPU ${device} with ${freq} MHz for system ${system}."
    echo""
done
