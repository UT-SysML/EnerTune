#!/bin/bash

git_dir=$(git rev-parse --show-toplevel)
cd ${git_dir}

freq=780
device=0

# Model-BatchSize-MIGSlice-NumReqs-LoadStart-LoadEnd-LoadSteps-Rps
mixes=(
# "resnet50-32-4-250380-1.0-1.0-0.25-1391.97" # requires 1050 MHz
"densenet169-4-4-45732-1.25-1.25-0.25-254.07 vgg19-4-2-30517-1.25-1.25-0.25-169.54" # requires 780 MHz
)

for mix in "${mixes[@]}"; do
    echo "System: EnerTune | Running mix: $mix | Device: $device"
    ./run.sh \
      --device-type a100 \
      --device-id ${device} \
      --modes custom-mig \
      --distribution point \
      --min-freq ${freq} \
      --max-freq ${freq} \
      ${mix}
    sleep 10
    echo "Completed ${mix} on GPU ${device} with ${freq} MHz for system EnerTune."
    echo""
done
