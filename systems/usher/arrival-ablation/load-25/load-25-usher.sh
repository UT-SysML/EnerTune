#!/bin/bash

git_dir=$(git rev-parse --show-toplevel)
cd ${git_dir}

freq=1410
device=0

# Model-BatchSize-MIGSlice-NumReqs-LoadStart-LoadEnd-LoadSteps-Rps
mixes=(
"diffusion_256_256-1-40-83-0.25-0.25-0.25-0.41497842657583255 vgg11-8-40-30517-0.25-0.25-0.25-169.54 whisper-1-20-708-0.25-0.25-0.25-3.29843852958"
# "whisper-1-20-708-0.25-0.25-0.25-3.29843852958 resnet152-1-30-50104-0.25-0.25-0.25-278.36 densenet169-2-30-45732-0.25-0.25-0.25-254.07 densenet161-1-20-42154-0.25-0.25-0.25-234.19"
# "vgg19-8-60-30517-0.25-0.25-0.25-169.54 densenet169-8-40-45732-0.25-0.25-0.25-254.07"
# "diffusion_1024_1024-1-51-29-0.25-0.25-0.25-0.0923146137935218 resnet50-8-47-250380-0.25-0.25-0.25-1391.97"
# "vgg11-1-40-134037-0.25-0.25-0.25-744.65 resnet50-2-40-250380-0.25-0.25-0.25-1391.97"
# "resnet152-8-45-50104-0.25-0.25-0.25-278.36 densenet161-8-40-42154-0.25-0.25-0.25-234.19"
# "gpt-1-45-40-0.25-0.25-0.25-0.16192860916956147 mobilenet_v3_large-8-35-176686-0.25-0.25-0.25-981.59"
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

mkdir -p ${git_dir}/results/usher-ablation-arrival-results/load-25
cp -r ${git_dir}/results/a100/* ${git_dir}/results/usher-ablation-arrival-results/load-25/
rm -rf ${git_dir}/results/a100/*
