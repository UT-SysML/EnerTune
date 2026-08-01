#!/bin/bash

git_dir=$(git rev-parse --show-toplevel)
cd ${git_dir}

freq=690
device=0

# Model-BatchSize-MIGSlice-NumReqs-LoadStart-LoadEnd-LoadSteps-Rps
mixes=(
"resnet152-8-4-50104-1.25-1.25-0.25-278.36 densenet161-16-3-42154-1.25-1.25-0.25-234.19" # requires 690 MHz
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

freq=960

# Model-BatchSize-MIGSlice-NumReqs-LoadStart-LoadEnd-LoadSteps-Rps
mixes=(
"diffusion_256_256-1-3-83-1.0-1.0-0.25-0.41497842657583255 gpt-1-3-40-1.0-1.0-0.25-0.16192860916956147" # requires 960 MHz
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
