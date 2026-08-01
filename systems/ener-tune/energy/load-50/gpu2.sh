#!/bin/bash

git_dir=$(git rev-parse --show-toplevel)
cd ${git_dir}

freq=870
device=0

# Model-BatchSize-MIGSlice-NumReqs-LoadStart-LoadEnd-LoadSteps-Rps
mixes=(
"diffusion_256_256-1-4-83-0.5-0.5-0.25-0.41497842657583255 densenet161-2-3-42154-0.5-0.5-0.25-234.19"
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
