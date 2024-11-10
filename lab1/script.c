
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    int imax = 100000;
    int tmax = 100000;
    int max_threads = 6;

    // Open the file to store results
    FILE *file = fopen("results.csv", "w");
    if (file == NULL) {
        perror("Error opening results file");
        return 1;
    }
    fprintf(file, "Threads,ExecutionTime\n"); // Write CSV header

    for (int threads = 1; threads <= max_threads; threads++) {
        char command[256];
        snprintf(command, sizeof(command), "prun -v -np 1 -t 01:00 ./assign_1_1_framework/assign1_1 %d %d %d", imax, tmax, threads);

        // Run the assign1_1 program and open a pipe to read its output
        FILE *fp = popen(command, "r");
        if (fp == NULL) {
            perror("Error running assign1_1");
            fclose(file);
            return 1;
        }

        // Read the output of assign1_1
        char output_line[256];
        double time_taken = 0.0;
        while (fgets(output_line, sizeof(output_line), fp) != NULL) {
            if (strstr(output_line, "Took") != NULL) {
                sscanf(output_line, "Took %lf seconds", &time_taken);
                break;
            }
        }

        // Close the pipe
        pclose(fp);

        // Write the result to the CSV file
        fprintf(file, "%d,%f\n", threads, time_taken);
        printf("Threads: %d, Time: %f seconds\n", threads, time_taken); // Print to console for verification
    }

    // Close the file
    fclose(file);
    printf("Results written to results.csv\n");

    return 0;
}
