#!/bin/bash

git_dir=$(git rev-parse --show-toplevel)
cd ${git_dir}

freq=1050
device=3

# Model-BatchSize-MIGSlice-NumReqs-LoadStart-LoadEnd-LoadSteps-Rps
mixes=(
# "resnet50-4-4-250380-0.75-0.75-0.25-1391.97 vgg19-2-3-30517-0.75-0.75-0.25-169.54" # requires 1320 MHz
"diffusion_1024_1024-1-4-29-0.75-0.75-0.25-0.0923146137935218 densenet161-4-3-42154-0.75-0.75-0.25-234.19" # requires 1050 MHz
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
