#!/bin/bash

git_dir=$(git rev-parse --show-toplevel)
cd ${git_dir}

freq=1410
device=0

# Model-BatchSize-MIGSlice-NumReqs-LoadStart-LoadEnd-LoadSteps-Rps
mixes=(
"gpt-1-4-40-0.5-0.5-0.25-0.16192860916956147 diffusion_1024_1024-1-3-29-0.5-0.5-0.25-0.0923146137935218"
"diffusion_256_256-1-3-83-0.5-0.5-0.25-0.41497842657583255 resnet152-16-2-50104-0.5-0.5-0.25-278.36 mobilenet_v3_large-8-1-176686-0.5-0.5-0.25-981.59"
"resnet50-32-3-250380-0.5-0.5-0.25-1391.97 densenet169-16-2-45732-0.5-0.5-0.25-254.07 densenet161-8-2-42154-0.5-0.5-0.25-234.19"
"whisper-1-4-708-0.25-1.0-0.25-3.9362878866995845 resnet50-1-3-250380-0.25-1.0-0.25-1391.97"
"vgg19-2-2-30517-0.5-0.5-0.25-169.54 whisper-8-2-708-0.5-0.5-0.25-3.9362878866995845"
)

for mix in "${mixes[@]}"; do
    echo "System: FGD | Running mix: $mix | Device: $device"
    ./run.sh \
      --device-type a100 \
      --device-id ${device} \
      --modes custom-mig \
      --distribution poisson \
      --min-freq ${freq} \
      --max-freq ${freq} \
      ${mix}
    sleep 10
    echo "Completed ${mix} on GPU ${device} with ${freq} MHz for system FGD."
    echo""
done

mkdir -p ${git_dir}/results/parva-ablation-arrival-results/load-50
cp -r ${git_dir}/results/a100/* ${git_dir}/results/parva-ablation-arrival-results/load-50/
rm -rf ${git_dir}/results/a100/*
