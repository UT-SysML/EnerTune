#!/bin/bash

git_dir=$(git rev-parse --show-toplevel)
cd ${git_dir}

freq=1140
device=0

# Model-BatchSize-MIGSlice-NumReqs-LoadStart-LoadEnd-LoadSteps-Rps
mixes=(
# "whisper-2-4-708-1.25-1.25-0.25-3.29843852958 mobilenet_v3_large-4-3-176686-1.25-1.25-0.25-981.59" # requires 1320 MHz
"diffusion_1024_1024-1-4-29-1.0-1.0-0.25-0.0923146137935218" # requires 1140 MHz
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

freq=1320

# Model-BatchSize-MIGSlice-NumReqs-LoadStart-LoadEnd-LoadSteps-Rps
mixes=(
"whisper-2-4-708-1.25-1.25-0.25-3.29843852958 mobilenet_v3_large-4-3-176686-1.25-1.25-0.25-981.59" # requires 1320 MHz
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
