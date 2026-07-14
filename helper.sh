#!/bin/bash

DOCKER="docker"
ZOOKEEPER_PORT="22181"
KAFKA_PORT="29092"

# orion details
ORION_CTR_PREFIX="orion"
ORION_IMG="fotstrt/orion-ae:v1"
ORION_FORK="orion-fork"

if [[ $USE_SUDO == 1 ]]; then
    SUDO="sudo"
    DOCKER="sudo -E docker"
fi

function helper_setup() {
    if [[ -z ${VENV} ]]; then
        echo "Set VENV env variables"
        return 1
    fi

    if [[ ! -z ${VENV} ]]; then
        source ${VENV}/bin/activate
    fi
}

function is_mig_feature_available() {
    echo $(nvidia-smi --query-gpu=name --format=csv,noheader | egrep -i "a100|h100" | wc -l)
}

function assert_mig_status()
{
    mode=$1
    device_id=$2
    if [[ ${mode} == "mig" || ${mode} == "mig-profile-3" || ${mode} == "mig-profile-2" || ${mode} == "mig-profile-1" || ${mode} == "custom-mig" ]]; then
        if [[ $(${SUDO} nvidia-smi -i ${device_id} --query-gpu=mig.mode.current --format=csv | grep "Enabled" | wc -l) -eq 0 ]]; then
            echo "MIG mode not enabled"
            return 1
        fi
    else
	mig_gpu=$(is_mig_feature_available)
        if [[ $mig_gpu -ne 0 && $(${SUDO} nvidia-smi -i ${device_id} --query-gpu=mig.mode.current --format=csv | grep "Disabled" | wc -l) -eq 0 ]]; then
            echo "MIG mode is not disabled"
            return 1
        fi
    fi
}

function enable_mps_if_needed()
{
    local mode=$1
    local device_id=$2
    if [[ ${mode} == mps-* ]]; then
        echo "Enabling MPS"
        export CUDA_VISIBLE_DEVICES=${device_id}
        export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_${device_id}
        export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_${device_id}
        ${SUDO} nvidia-smi -i ${device_id} -c EXCLUSIVE_PROCESS
        nvidia-cuda-mps-control -d
        unset CUDA_VISIBLE_DEVICES CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
    fi
}

function disable_mps_if_needed()
{
    local mode=$1
    local device_id=$2
    if [[ ${mode} == mps-* ]]; then
        export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_${device_id}
        echo quit | nvidia-cuda-mps-control
        ${SUDO} nvidia-smi -i ${device_id} -c DEFAULT
        ${SUDO} rm -fr /tmp/mps_${device_id} || :
        ${SUDO} rm -fr /tmp/mps_log_$i || :
    fi
}

function setup_mig_if_needed()
{
    local mode=$1
    local device_id=$2
    local num_procs=$3
    local mig_slice=$4
    local -n mig_slices="$5"

    if [[ ${mode} != "mig" && "$mode" != "mig-profile-1" && "$mode" != "mig-profile-2" && "$mode" != "mig-profile-3" && "$mode" != "custom-mig" ]]; then
        return
    fi

    # Create GPU instances
    if [[ ${mode} == "mig" ]]; then
        if [[ $# -eq 3 && $3 == "chunks" ]]; then
            gi=("" "19" "19,19" "19,19,19" "19,19,19,19" "19,19,19,19,19" "19,19,19,19,19,19" "19,19,19,19,19,19,19")
            ${SUDO} nvidia-smi mig -i ${device_id} -cgi "${gi[${num_procs}]}"
        elif [[ ${mig_slice} -gt 0 ]]; then # Used for profiling purposes, single model right now
            gi=("" "19" "14" "9" "5" "" "" "0")
            ${SUDO} nvidia-smi mig -i ${device_id} -cgi "${gi[${mig_slice}]}"
        else
            gi=("" "0" "9,5" "9,14,14" "14,14,14,19" "14,14,19,19,19" "14,19,19,19,19,19" "19,19,19,19,19,19,19")
            ${SUDO} nvidia-smi mig -i ${device_id} -cgi "${gi[${num_procs}]}"
        fi
    elif [[ ${mode} == "mig-profile-3" ]]; then
        gi=("" "9" "9,9")
        ${SUDO} nvidia-smi mig -i ${device_id} -cgi "${gi[${num_procs}]}"
    elif [[ ${mode} == "mig-profile-2" ]]; then
        gi=("" "14" "14,14")
        ${SUDO} nvidia-smi mig -i ${device_id} -cgi "${gi[${num_procs}]}"
    elif [[ ${mode} == "mig-profile-1" ]]; then
        gi=("" "19" "19,19" "19,19,19" "19,19,19,19")
        ${SUDO} nvidia-smi mig -i ${device_id} -cgi "${gi[${num_procs}]}"
    elif [[ ${mode} == "custom-mig" ]]; then
        local mapped=()
        for val in "${mig_slices[@]}"; do
            case "$val" in
                2) mapped+=(14) ;;
                1) mapped+=(19) ;;
                4) mapped+=(5)  ;;
                3) mapped+=(9)  ;;
		7) mapped+=(0)  ;;
                *) echo "Unknown MIG slice: $val" >&2; return 1 ;;
            esac
        done

        # Join with commas
        local IFS=,
        mapped_string="${mapped[*]}"
        ${SUDO} nvidia-smi mig -i ${device_id} -cgi "$mapped_string"
    fi

    # Get GPU Instance ID
    # gi_id_arr=($(${SUDO} nvidia-smi mig -i ${device_id} -lgi | awk '{print $6}' | grep -P '^\d+$'))
    mapfile -t gi_id_arr < <(
        ${SUDO} nvidia-smi mig -i ${device_id} -lgi \
            | awk '{print $6}' \
            | tr ',' '\n' \
            | grep -P '^\d+$'
    )

    # Create Compute Instance
    iteration_num=0
    for gi_id in "${gi_id_arr[@]}"
    do
        iteration_num=$((iteration_num+1))
        ci=$(${SUDO} nvidia-smi mig -i ${device_id} -gi ${gi_id} -lcip | grep "\*" | awk '{print $6}' | tr -d "*")
        ${SUDO} nvidia-smi mig -i ${device_id} -gi ${gi_id} -cci ${ci}
    done
}

function cleanup_mig_if_needed()
{
    mode=$1
    device_id=$2
    if [[ ${mode} != "mig" && ${mode} != "mig-profile-3" && ${mode} != "mig-profile-2" && ${mode} != "mig-profile-1" && ${mode} != "custom-mig" ]]; then
        return
    fi

	# Delete the GPU profile
	while :;
	do
        # Cleanup Compute Instances
        ci_arr=($(${SUDO} nvidia-smi mig -i ${device_id} -lci | awk '{print $7}' | grep -P '^\d+$'))
        gi_arr=($(${SUDO} nvidia-smi mig -i ${device_id} -lci | awk '{print $3}' | grep -P '^\d+$'))
        local i=0
        while [[ $i -lt ${#gi_arr[*]} ]];
        do
            ci=${ci_arr[$i]}
            gi=${gi_arr[$i]}
            ${SUDO} nvidia-smi mig -i ${device_id} -dci -ci ${ci} -gi ${gi}
            i=$(( $i + 1))
        done

        # Cleanup GPU Instances
        original_e=$(set +o | grep errexit)
        set +e
        ${SUDO} nvidia-smi mig -i ${device_id} -dgi
        local mig_exit_code=$?
        eval "$original_e"

        if [[ ${mig_exit_code} -eq 0 || ${mig_exit_code} -eq 6 ]]; then
            break
        fi
        echo "${SUDO} nvidia-smi mig -i ${device_id} -dgi failed. Trying in 1s"
        sleep 1
    done
}

function lock_gpu() {
    local device_id=$1
    echo "Attempting to acquire exclusive lock for GPU: ${device_id}"
    exec 9>/tmp/gpu_${device_id}.lock
    flock -x 9
    echo "Acquired exclusive lock for GPU: ${device_id}"
}

function unlock_gpu() {
    local device_id=$1
    exec 9>&-
}

function set_gpu_freq() {
    local device_id=$1
    local min_freq=$2
    local max_freq=$3
    echo "Attempting to set GPU frequency for GPU: ${device_id}"
    for i in {1..5}; do
        ${SUDO} nvidia-smi -i ${device_id} -lgc ${min_freq},${max_freq}

        # Parse the output of nvidia-smi to check the second line
        local current_freq=$(${SUDO} nvidia-smi -i ${device_id} --query-gpu=clocks.sm --format=csv | sed -n '2p' | awk -F', ' '{print $1}' | tr -d '[:space:]' | tr -d 'MHz')
        if [[ "${current_freq}" != "${max_freq}" ]]; then
            echo "Warning: Current SM clock frequency (${current_freq}) does not match the expected max frequency (${max_freq})."
        else
            echo "Max frequency set successfully to ${max_freq} on GPU: ${device_id}"
            break
        fi
    done
    # ${SUDO} nvidia-smi -i ${device_id} -lgc ${min_freq},${max_freq}

    # # Parse the output of nvidia-smi to check the second line
    # local current_freq=$(${SUDO} nvidia-smi -i ${device_id} --query-gpu=clocks.sm --format=csv | sed -n '2p' | awk -F', ' '{print $1}' | tr -d '[:space:]' | tr -d 'MHz')
    # if [[ "${current_freq}" != "${max_freq}" ]]; then
    #     echo "Warning: Current SM clock frequency (${current_freq}) does not match the expected max frequency (${max_freq})."
    # else
    #     echo "Max frequency set successfully to ${max_freq} on GPU: ${device_id}"
    # fi
}

function reset_gpu_freq() {
    local device_id=$1
    echo "Resetting GPU frequency for GPU: ${device_id}"
    ${SUDO} nvidia-smi -i ${device_id} -rgc

}

function cleanup()
{
    mode=$1
    device_id=$2
    disable_mps_if_needed ${mode} ${device_id}
    cleanup_mig_if_needed ${mode} ${device_id}
    reset_gpu_freq ${device_id}
    unlock_gpu ${device_id}
}

function calc_mean_sd() {
    arr=("$@")
    mean=$(echo ${arr[@]} | awk '{for(i=1;i<=NF;i++){sum+=$i};print sum/NF}')
    sd=$(echo ${arr[@]} | awk -vM=$mean '{for(i=1;i<=NF;i++){sum+=($i-M)*($i-M)};print sqrt(sum/NF)}')
    echo "$mean, $sd"
}

function calc_min_max_mean {
    arr=("$@")
    max=$(echo ${arr[@]} | awk -vmax=${arr[0]} '{for(i=1;i<=NF;i++){(( $i > max )) && max=$i};print max}')
    min=$(echo ${arr[@]} | awk -vmin=${arr[0]} '{for(i=1;i<=NF;i++){(( $i < min )) && min=$i};print min}')
    mean=$(echo ${arr[@]} | awk '{for(i=1;i<=NF;i++){sum+=$i};print sum/NF}')
    echo "$min, $max, $mean"
}

function wait_till_one_process_exits {
    pids=("$@")
    cpu=0
    done=0
    while true; do
        for pid in ${pids[@]}; do
            if ! taskset -c ${cpu} kill -0 $pid >/dev/null 2>&1; then
                done=1
                break
            fi
        done

        if [[ ${done} -eq 1 ]]; then
            echo "Sending SIGUSR2 to ${pids[@]}"
            kill -SIGUSR2 ${pids[@]} >/dev/null 2>&1
            break
        fi
	done

    for pid in ${pids[@]}; do
        echo "Waiting for ${pid} to finish"
        while taskset -c 0 kill -0 ${pid} >/dev/null 2>&1; do sleep 1; done
    done
}

function setup_orion_container {
    local devices_arg=$1
    local uuid_arg=$2
    local WS=$(git rev-parse --show-toplevel)
    local DOCKER_WS=/root/$(basename ${WS})

    if [[ -z ${uuid_arg} ]]; then
        uuid_arg=$(uuidgen)
    fi

    local gpu_filter="--gpus '\"device=${devices_arg}\"'"
    if [[ -z ${devices_arg} ]]; then
        unset gpu_filter
    fi

    orion_ctr=${ORION_CTR_PREFIX}"-"${uuid_arg}
    cmd="${DOCKER} run -v ${WS}:${DOCKER_WS} -it -d \
        -w ${DOCKER_WS} \
        --name ${orion_ctr} \
        --net=host \
        --ipc=host --pid=host \
        ${gpu_filter} \
        ${ORION_IMG} bash > /dev/null"
    echo "Running cmd: '${cmd}'"
    eval ${cmd}

    # Install necessary package
    NSIGHT_COMPUTE_TAR=nsight-compute.tar
    if [[ ! -f ${NSIGHT_COMPUTE_TAR} ]]; then
        # pip install gdown
        cmd="gdown --id 1_HY1FOIS6KP7dLTKRZ30Wliqc9P_N7hu"

        original_e=$(set +o | grep errexit)
        set +e
        eval ${cmd}
        local orion_setup_exit_code=$?
        eval "$original_e"

        if [[ ${orion_setup_exit_code} -ne 0 ]]; then
            echo "gdown package not found!"
            echo "Install using: 'pip install gdown'"
            echo "Or try running command: '${cmd}', fix it and re-call the script"
            return 1
        fi
    fi
    ${DOCKER} cp ${NSIGHT_COMPUTE_TAR} ${orion_ctr}:/usr/local/ > /dev/null 2>&1
    ${DOCKER} exec ${orion_ctr} bash -c "tar -xf /usr/local/nsight-compute.tar -C /usr/local/ > /dev/null 2>&1"
    ${DOCKER} exec ${orion_ctr} bash -c "wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_1/nsightsystems-linux-cli-public-2024.1.1.59-3380207.deb > /dev/null 2>&1"
    ${DOCKER} exec ${orion_ctr} bash -c "dpkg -i nsightsystems-linux-cli-public-2024.1.1.59-3380207.deb > /dev/null 2>&1 && rm -f nsightsystems-linux-cli-public-2024.1.1.59-3380207.deb"
    ${DOCKER} exec ${orion_ctr} bash -c "pip install transformers grpcio grpcio-tools protobuf kafka-python > /dev/null 2>&1"

}

function cleanup_orion_containers {
    local grep_filter=$1
    if [[ -z ${grep_filter} ]]; then
        return
    fi
    ${DOCKER} ps -a | grep ${ORION_CTR_PREFIX} | grep "${grep_filter}" |
        awk '{print $NF}' | \
        xargs -I{} ${DOCKER} rm -f {} >/dev/null 2>&1 || :
}

function multiply_and_round() {
    local result=$(echo "$1 * $2" | bc)
    if [[ $# -eq 3 ]]; then
        printf "%.$3f\n" "$result"
    else
        printf "%.2f\n" "$result"
    fi
}

function divide_and_round() {
    local result=$(echo "scale=2; $1 / $2" | bc)
    printf "%.2f\n" "$result"
}

function get_result_dir() {
    declare -a models_arr=("${!1}")
    declare -a batch_sizes_arr=("${!2}")
    declare -a distribution_types_arr=("${!3}")
    local device_type_arg=$4

    local result_base=$(IFS=- ; echo "${models_arr[*]}")
    for i in "${!models_arr[@]}"; do
        concatenated_string="${concatenated_string}${models_arr[$i]}-${batch_sizes_arr[$i]}-${distribution_types_arr[$i]}_"
    done
    local result_id=${concatenated_string%_}
    result_dir=results/${device_type_arg}/${result_base}/${result_id}
}

function parse_model_parameters()
{
    model_run_params=()
    models=()
    batch_sizes=()
    distribution_types=()
    mig_slices=()
    num_reqs=()

    while IFS= read -r element; do
        args=()

        # Extract key-value pairs from the JSON object
        keys=($(echo "$element" | jq -r 'keys[]'))
        for key in "${keys[@]}"; do
            value=$(echo "$element" | jq -r '.["'"$key"'"]')
            args+=( "--$key" "$value" )
        done

        # Join the arguments array
        formatted_string="${args[*]}"

        # Append the formatted string to the array
        model_run_params+=( "$formatted_string" )
        model=$(echo "$element" | jq -r '.["'"model"'"]')
        bs=$(echo "$element" | jq -r '.["'"batch-size"'"]')
        dt=$(echo "$element" | jq -r '.["'"distribution-type"'"]')
        slice=$(echo "$element" | jq -r '.["'"mig-slice"'"]')
        reqs=$(echo "$element" | jq -r '.["'"num-reqs"'"]')

        if [[ ${model} == null || ${bs} == null || ${dt} == null || ${slice} == null || ${reqs} == null ]]; then
            echo "Required key missing."
            echo "One of 'model', 'batch-size'"
            echo "Check: '${element}'"
            return 1
        fi

        models+=(${model})
        batch_sizes+=(${bs})
        distribution_types+=(${dt})
        mig_slices+=(${slice})
        num_reqs+=(${reqs})
    echo "${mig_slices[@]}"
    done < <(echo "$@" | jq -c '.[]')

    model_run_params_raw="$@"
}

function safe_clean_gpu() {
    declare -a procs_arg=("${!1}")
    local mode_arg=$2
    local device_id_arg=$3

    if [[ ${#procs_arg[@]} -eq 0 ]]; then
        return
    fi

    # Check if processes are alive before sending signal - could be dead because model reached inference request limit
    for pid in "${procs_arg[@]}"; do
        if taskset -c 0 kill -0 ${pid} 2>/dev/null; then
            kill -SIGUSR2 ${procs_arg[@]}
            break
        fi
    done

    # kill -SIGUSR2 ${procs_arg[@]}

    # Wait for the process to clean up
    for pid in "${procs_arg[@]}"
    do
        while taskset -c 0 kill -0 ${pid} >/dev/null 2>&1; do sleep 1; done
    done

    # Clean up the GPU for future use
    disable_mps_if_needed ${mode_arg} ${device_id_arg}
    cleanup_mig_if_needed ${mode_arg} ${device_id_arg}
    reset_gpu_freq ${device_id_arg}
    unlock_gpu ${device_id_arg}
}

check_kafka_up() {
    kafka_server_arg=$1
    local ip=$(echo "${kafka_server_arg}" | awk -F':' '{print $1}')
    local port=$(echo "${kafka_server_arg}" | awk -F':' '{print $2}')
    nc -zv ${ip} ${port} > /dev/null
}

mps_mig_percentages=("" "100" "57,43" "42,29,29" "29,29,28,14" "29,29,14,14,14" "29,15,14,14,14,14" "15,15,14,14,14,14,14")
mps_equi_percentages=("" "100" "50,50" "34,33,33" "25,25,25,25" "20,20,20,20,20" "17,17,17,17,16,16" "15,15,14,14,14,14,14")
# Command to profile
# nsys profile --stats=true --force-overwrite true --wait=all -o trial
