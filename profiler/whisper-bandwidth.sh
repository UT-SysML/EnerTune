#!/bin/bash

# ==== CONFIGURATION ====

GPU_TYPE="a100"
DEVICE_ID=1

# Models to iterate over
models=(
"whisper"
"diffusion_256_256"
"diffusion_1024_1024"
)

# Batch sizes to test
batch_sizes=(1 2 4 8 16 32)

# CSV output file
csv_file="profiling_times.csv"

# Initialize CSV (write header if not exists)
if [ ! -f "$csv_file" ]; then
    echo "model,batch_size,time_sec" > "$csv_file"
fi

# ==== EXECUTION LOOP ====

echo "🚀 Starting profiling sweep..."
echo "Results will be logged to: ${csv_file}"
echo "--------------------------------------------"

for model in "${models[@]}"; do
    echo "========== MODEL: ${model} =========="

    for batch_size in "${batch_sizes[@]}"; do

        # Skip large batch sizes for diffusion and GPT
        if [[ ("$model" == "gpt" || "$model" == "diffusion_256_256" || "$model" == "diffusion_1024_1024") && ( "$batch_size" -gt 8 ) ]]; then
            echo "⏩ Skipping ${model} with batch_size=${batch_size} (too large)"
            continue
        fi

        echo "▶️  Running ./nsys.sh ${GPU_TYPE} ${DEVICE_ID} ${model} ${batch_size}"

        # Record start time (nanoseconds)
        start_time_ns=$(date +%s%N)

        # Execute profiling run
        ./nsys.sh "${GPU_TYPE}" "${DEVICE_ID}" "${model}" "${batch_size}"
        status=$?

        # Record end time
        end_time_ns=$(date +%s%N)
        duration_ns=$((end_time_ns - start_time_ns))
        duration_sec=$(awk "BEGIN {printf \"%.3f\", ${duration_ns}/1000000000}")

        if [ $status -ne 0 ]; then
            echo "❌ Error: ${model} (batch_size=${batch_size}) failed after ${duration_sec}s"
        else
            echo "✅ Completed ${model} (batch_size=${batch_size}) in ${duration_sec}s"
        fi

        # Append timing info to CSV
        echo "${model},${batch_size},${duration_sec}" >> "$csv_file"

        echo "--------------------------------------"
    done
done

echo "🏁 All profiling runs complete!"
echo "CSV summary written to: ${csv_file}"
