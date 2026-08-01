#!/bin/bash

git_dir=$(git rev-parse --show-toplevel)
cd ${git_dir}

freq=690
device=0
system="EnerTune"

# Model-BatchSize-MIGSlice-NumReqs-LoadStart-LoadEnd-LoadSteps-Rps
mixes=(
"diffusion_1024_1024-1-4-29-0.25-0.25-0.25-0.0923146137935218 diffusion_256_256-1-3-83-0.25-0.25-0.25-0.41497842657583255"
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
