#!/bin/bash

git_dir=$(git rev-parse --show-toplevel)
cd ${git_dir}

freq=1410
device=0

# Model-BatchSize-MIGSlice-NumReqs-LoadStart-LoadEnd-LoadSteps-Rps
mixes=(
"diffusion_1024_1024-1-57-29-0.75-0.75-0.25-0.0923146137935218 resnet152-8-43-50104-0.75-0.75-0.25-278.36"
"mobilenet_v3_large-8-43-176686-0.75-0.75-0.25-981.59 whisper-1-43-708-0.75-0.75-0.25-3.29843852958"
"vgg11-16-100-134037-0.75-0.75-0.25-744.65"
"diffusion_1024_1024-1-57-29-0.75-0.75-0.25-0.0923146137935218"
"diffusion_256_256-1-43-83-0.75-0.75-0.25-0.41497842657583255 vgg19-16-57-30517-0.75-0.75-0.25-169.54"
"gpt-1-43-40-0.75-0.75-0.25-0.16192860916956147 densenet169-4-29-45732-0.75-0.75-0.25-254.07"
"resnet50-32-57-250380-0.75-0.75-0.25-1391.97 densenet161-4-29-42154-0.75-0.75-0.25-234.19"

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

mkdir -p ${git_dir}/results/gpulets-ablation-arrival-results/load-75
cp -r ${git_dir}/results/a100/* ${git_dir}/results/gpulets-ablation-arrival-results/load-75/
rm -rf ${git_dir}/results/a100/*
