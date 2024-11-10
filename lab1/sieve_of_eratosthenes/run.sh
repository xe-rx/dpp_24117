#!/bin/bash

# Define core lists for each core count
declare -A core_counts
core_counts=( [1]="0" [4]="0,1,2,3" [8]="0-7" [16]="0-15" )

# Create or clear the log file
log_file="execution_times.log"
> "$log_file"

# Loop through each core configuration and log the execution time
for cores in "${!core_counts[@]}"; do
    core_list="${core_counts[$cores]}"

    # Log the configuration
    echo "Running ./eratosthenes on $cores core(s) (taskset -c $core_list)" | tee -a "$log_file"

    # Start timer, run the command, and stop timer
    start_time=$(date +%s.%N)
    taskset -c "$core_list" ./eratosthenes
    end_time=$(date +%s.%N)

    # Calculate elapsed time
    elapsed_time=$(echo "$end_time - $start_time" | bc)

    # Log the execution time
    echo "Execution time on $cores core(s): $elapsed_time seconds" | tee -a "$log_file"
    echo "---------------------------------------------" | tee -a "$log_file"
done
