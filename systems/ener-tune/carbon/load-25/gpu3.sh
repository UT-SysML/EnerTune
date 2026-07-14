#!/bin/bash

git_dir=$(git rev-parse --show-toplevel)
cd ${git_dir}

freq=870
device=3

# Model-BatchSize-MIGSlice-NumReqs-LoadStart-LoadEnd-LoadSteps-Rps
mixes=(
# "vgg19-2-4-30517-0.25-0.25-0.25-169.54 gpt-1-3-40-0.25-0.25-0.25-0.16192860916956147" # requires 1050 MHz
"diffusion_256_256-1-4-83-0.25-0.25-0.25-0.41497842657583255 whisper-2-2-708-0.25-0.25-0.25-3.29843852958" # requires 870 MHz
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
