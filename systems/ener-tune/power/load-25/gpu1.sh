#!/bin/bash

git_dir=$(git rev-parse --show-toplevel)
cd ${git_dir}

freq=240
device=0

# Model-BatchSize-MIGSlice-NumReqs-LoadStart-LoadEnd-LoadSteps-Rps
mixes=(
"vgg19-2-4-30517-0.25-0.25-0.25-169.54 whisper-2-3-708-0.25-0.25-0.25-3.29843852958"
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
