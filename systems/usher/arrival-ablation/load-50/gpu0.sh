#!/bin/bash

git_dir=$(git rev-parse --show-toplevel)
cd ${git_dir}

freq=1410
device=0

# Model-BatchSize-MIGSlice-NumReqs-LoadStart-LoadEnd-LoadSteps-Rps
mixes=(
# "diffusion_256_256-1-43-83-0.5-0.5-0.25-0.41497842657583255 gpt-1-34-40-0.5-0.5-0.25-0.16192860916956147 mobilenet_v3_large-1-15-176686-0.5-0.5-0.25-981.59" # 50% Job Mix Plan
"diffusion_256_256-1-40-83-0.5-0.5-0.25-0.41497842657583255 vgg11-8-40-30517-0.5-0.5-0.25-169.54 whisper-1-20-708-0.5-0.5-0.25-3.29843852958"
)

for mix in "${mixes[@]}"; do
    echo "System: Usher | Running mix: $mix | Device: $device"
    ./run.sh \
      --device-type a100 \
      --device-id ${device} \
      --modes mps-manual-cap \
      --distribution poisson \
      --min-freq ${freq} \
      --max-freq ${freq} \
      ${mix}
    sleep 10
    echo "Completed ${mix} on GPU ${device} with ${freq} MHz for system Usher."
    echo""
done
