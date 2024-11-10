
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# Define parameters for the C program
imax = 100000
tmax = 100000

# Define the range of thread counts you want to test
thread_counts = list(range(1,33))
execution_times = []

# Debug message to indicate the start of the process
print("Starting execution of ./assign_1_1_framework/assign1_1 with varying thread counts.")

# Run the C program with each thread count and capture the execution time
for threads in thread_counts:
    print(f"Running ./assign_1_1_framework/assign1_1 with {threads} threads...")

    try:
        # Execute the C program
        result = subprocess.run(
            ["./assign_1_1_framework/assign1_1", str(imax), str(tmax), str(threads)],
            capture_output=True, text=True, timeout=60  # Setting a timeout for each run
        )

        # Debug output to verify result
        print(f"Output for {threads} threads:\n{result.stdout}\n")

        # Parse the output to get the execution time in seconds
        output = result.stdout.splitlines()
        time_taken = None
        for line in output:
            if "Took" in line:
                time_taken = float(line.split()[1])  # Extract the time value
                execution_times.append(time_taken)
                print(f"Execution time for {threads} threads: {time_taken} seconds")  # Debug message
                break

        # Check if execution time was captured
        if time_taken is None:
            print(f"Warning: No valid execution time found in the output for {threads} threads.")

    except subprocess.TimeoutExpired:
        print(f"Execution for {threads} threads timed out.")
        execution_times.append(float('inf'))  # Append infinity if timed out to indicate failure

# Verify if execution times were recorded correctly
print(f"Collected execution times: {execution_times}")

# Calculate speedup relative to single-threaded execution if available
if execution_times[0] != float('inf'):
    baseline_time = execution_times[0]
    speedup = [baseline_time / time if time != float('inf') else 0 for time in execution_times]
else:
    print("Error: Baseline time for single-thread execution is invalid.")
    speedup = [0] * len(thread_counts)

# Convert thread counts and speedup to numpy arrays for interpolation
thread_counts_np = np.array(thread_counts)
speedup_np = np.array(speedup)

# Define a range of points for smoother plotting
thread_counts_smooth = np.linspace(thread_counts_np.min(), thread_counts_np.max(), 300)

# Perform cubic spline interpolation
spline = make_interp_spline(thread_counts_np, speedup_np, k=3)
speedup_smooth = spline(thread_counts_smooth)

# Plot the smooth speedup curve
plt.figure(figsize=(10, 6))
plt.plot(thread_counts_smooth, speedup_smooth, marker='o', label='Interpolated Speedup')
plt.scatter(thread_counts, speedup, color="red", label="Original Data Points")  # Show original points
plt.xlabel("Number of Threads")
plt.ylabel("Speedup")
plt.title("Speedup vs. Number of Threads")
plt.legend()
plt.grid(True)

# Save the plot as a PNG file
plt.savefig("speedup_plot.png", format="png", dpi=300)  # Save with high resolution

# Display the plot
plt.show()

