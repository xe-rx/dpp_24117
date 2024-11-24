#!/bin/bash

# Define workload sizes
workloads=(1000 10000 100000 1000000 10000000)

# Output files for results
individual_file="cuda_individual_trials.csv"
averages_file="cuda_averages.csv"

# Write headers only if the files don't already exist
if [ ! -f $individual_file ]; then
    echo "N,Trial,Average Time (ms),Total Time (ms),Count" > $individual_file
fi
if [ ! -f $averages_file ]; then
    echo "N,Average of Avg Time (ms),Average of Total Time (ms),Count (Trials)" > $averages_file
fi

# Loop over each workload size
for N in "${workloads[@]}"; do
    avg_time_sum=0
    total_time_sum=0
    trial_count=3  # Number of trials per workload

    for trial in {1..3}; do
        # Execute the CUDA program using prun
        result=$(prun -v -np 1 -native "-C TitanX --gres=gpu:1" ./assign2_1 $N 1000 512)
        
        # Extract relevant timing information
        avg_time=$(echo "$result" | grep "avg =" | awk -F 'avg = ' '{print $2}' | awk -F ' ms,' '{print $1}')
        total_time=$(echo "$result" | grep "total =" | awk -F 'total = ' '{print $2}' | awk -F ' ms,' '{print $1}')
        count=$(echo "$result" | grep "count =" | awk -F 'count = ' '{print $2}' | tr -d ' ')
        
        # Validate extracted values
        if [[ "$avg_time" =~ ^[0-9.]+$ && "$total_time" =~ ^[0-9.]+$ ]]; then
            # Append the individual trial result to the file
            echo "$N,Trial $trial,$avg_time,$total_time,$count" >> $individual_file
            
            # Accumulate values for averaging
            avg_time_sum=$(echo "$avg_time_sum + $avg_time" | bc)
            total_time_sum=$(echo "$total_time_sum + $total_time" | bc)
        else
            echo "Error: Non-numeric value encountered in trial $trial for N=$N."
            exit 1
        fi
    done

    # Calculate averages
    avg_avg_time=$(echo "$avg_time_sum / $trial_count" | bc -l)
    avg_total_time=$(echo "$total_time_sum / $trial_count" | bc -l)

    # Append the averages to the averages file
    echo "$N,$avg_avg_time,$avg_total_time,$trial_count" >> $averages_file
done

# Notify the user of completion
echo "CUDA timing tests complete."
echo "Individual trial results saved in $individual_file."
echo "Averages saved in $averages_file."

