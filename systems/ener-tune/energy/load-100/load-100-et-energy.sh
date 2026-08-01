#!/bin/bash

git_dir=$(git rev-parse --show-toplevel)
cd ${git_dir}/systems/ener-tune/energy/load-100
./gpu0.sh
./gpu1.sh
./gpu2.sh
./gpu3.sh
mkdir -p ${git_dir}/results/et-energy-results/load-100
cp -r ${git_dir}/results/a100/* ${git_dir}/results/et-energy-results/load-100/
rm -rf ${git_dir}/results/a100/*
