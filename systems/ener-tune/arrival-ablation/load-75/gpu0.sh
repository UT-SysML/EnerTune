#!/bin/bash

git_dir=$(git rev-parse --show-toplevel)
cd ${git_dir}

freq=1050
device=0
system="EnerTune"

# Model-BatchSize-MIGSlice-NumReqs-LoadStart-LoadEnd-LoadSteps-Rps
mixes=(
# "vgg11-1-4-134037-0.75-0.75-0.25-744.65  gpt-1-3-40-0.75-0.75-0.25-0.16192860916956147"
"gpt-1-4-40-0.75-0.75-0.25-0.16192860916956147 vgg11-1-3-134037-0.75-0.75-0.25-744.65"
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
