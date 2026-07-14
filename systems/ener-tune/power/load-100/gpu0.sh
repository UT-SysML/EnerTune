#!/bin/bash

git_dir=$(git rev-parse --show-toplevel)
cd ${git_dir}

freq=780
device=0
system="EnerTune"

# Model-BatchSize-MIGSlice-NumReqs-LoadStart-LoadEnd-LoadSteps-Rps
mixes=(
# "densenet161-4-4-42154-1.0-1.0-0.25-234.19 whisper-16-3-708-1.0-1.0-0.25-3.29843852958"
"whisper-16-4-708-1.0-1.0-0.25-3.29843852958 densenet161-4-3-42154-1.0-1.0-0.25-234.19"
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
