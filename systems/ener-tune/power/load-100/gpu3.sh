#!/bin/bash

git_dir=$(git rev-parse --show-toplevel)
cd ${git_dir}

freq=960
device=3

# Model-BatchSize-MIGSlice-NumReqs-LoadStart-LoadEnd-LoadSteps-Rps
mixes=(
"gpt-1-3-40-1.0-1.0-0.25-0.16192860916956147 diffusion_256_256-1-3-83-1.0-1.0-0.25-0.41497842657583255" # requires frequency change to 960
# "vgg11-8-3-134037-1.0-1.0-0.25-744.65 densenet169-4-3-45732-1.0-1.0-0.25-254.07" # requires frequency change to 780 
)

for mix in "${mixes[@]}"; do
    echo "System: Usher | Running mix: $mix | Device: $device"
    ./run.sh \
      --device-type a100 \
      --device-id ${device} \
      --modes custom-mig \
      --distribution point \
      --min-freq ${freq} \
      --max-freq ${freq} \
      ${mix}
    sleep 10
    echo "Completed ${mix} on GPU ${device} with ${freq} MHz for system Usher."
    echo""
done
