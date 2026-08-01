#!/bin/bash

git_dir=$(git rev-parse --show-toplevel)
cd ${git_dir}

freq=1050
device=0
system="EnerTune"

# Model-BatchSize-MIGSlice-NumReqs-LoadStart-LoadEnd-LoadSteps-Rps
mixes=(
"gpt-1-4-40-1.0-1.0-0.25-0.16192860916956147 resnet152-4-3-50104-1.25-1.25-0.25-278.36"
)

for mix in "${mixes[@]}"; do
    echo "System: ${system} | Running mix: $mix | Device: $device"
    ./run.sh \
      --device-type a100 \
      --device-id ${device} \
      --modes custom-mig \
      --distribution poisson \
      --min-freq ${freq} \
      --max-freq ${freq} \
      ${mix}
    sleep 10
    echo "Completed ${mix} on GPU ${device} with ${freq} MHz for system ${system}."
    echo""
done
