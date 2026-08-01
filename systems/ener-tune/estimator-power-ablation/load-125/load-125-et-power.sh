#!/bin/bash

git_dir=$(git rev-parse --show-toplevel)
cd ${git_dir}/systems/ener-tune/estimator-ablation/load-125
./gpu0.sh
./gpu1.sh
./gpu2.sh
./gpu3.sh
mkdir -p ${git_dir}/results/et-pwr-estimator-ablation-results/load-125
cp -r ${git_dir}/results/a100/* ${git_dir}/results/et-pwr-estimator-ablation-results/load-125/
rm -rf ${git_dir}/results/a100/*
