# Profiler App for Heat Equation Solver

This profiler app is designed to analyze and visualize the performance of the Heat Equation Solver using NVIDIA's CUDA Profiler (NCU). It includes tools for generating profiling data, processing it, and visualizing the results using a Streamlit web application.

## Directory Structure

```
profiler-app/
├── app.py
├── creat-csv.py
├── elur.md
├── euler.csv
├── generate_table.py
├── heat_equation_kernels.txt
├── kernel_analysis_app.py
├── profiling.sh
├── README.md
```

## Components

1. `profiling.sh`: Shell script for running the NCU profiler with various block sizes.
2. `generate_table.py`: Python script to process the profiler output and generate a Markdown table.
3. `kernel_analysis_app.py`: Streamlit application for visualizing the profiler results.
4. `heat_equation_kernels.txt`: Contains information about the heat equation kernels.
5. `elur.md`: Processed Markdown table of profiler results (Note: this might be a typo of `euler.md`).
6. `euler.csv`: Processed CSV of profiler results.
7. `creat-csv.py`: Script to convert Markdown to CSV (Note: this might be a typo of `create-csv.py`).
8. `app.py`: Additional application file (purpose to be specified).

## Usage

### Step 1: Generate Profiling Data

Run the profiling script to generate raw profiler data:

```bash
./profiling.sh
```

This command will run the NCU profiler for various block sizes and save the output to a file (possibly `euler.txt`, though not visible in the current directory structure).

### Step 2: Process Profiling Data

Convert the raw profiler data to a Markdown table:

```bash
python generate_table.py
```

This script processes the raw profiler data and generates a formatted Markdown table.

### Step 3: Create CSV

Convert the Markdown data to a CSV file:

```bash
python creat-csv.py
```

This step creates `euler.csv` from the Markdown data.

### Step 4: Visualize Results

Launch the Streamlit app to visualize the profiling results:

```bash
streamlit run kernel_analysis_app.py
```

This will start a local server, and you can view the app in your web browser.

## Streamlit App Features

The Streamlit app (`kernel_analysis_app.py`) provides interactive visualizations of the profiling data:

1. **Data Table**: Displays the raw profiling data in a table format.
2. **Minimization and Maximization Analysis**: Allows selection of metrics to minimize or maximize, displaying top-k results.
3. **Inter-Kernel Block Size Analysis**: Compares multiple metrics across different kernels and block sizes.
4. **Multiple Metrics Comparison**: Allows comparison of two metrics on a single graph with dual y-axes for a selected kernel.
5. **Metric Explanations**: Provides detailed explanations for each profiled metric.
6. **Occupancy Analysis**: Compares theoretical and achieved occupancy across block sizes.
7. **Performance Efficiency**: Shows the ratio of Compute Throughput to Memory Throughput.
8. **Parallel Coordinates Plot**: Visualizes multiple metrics across different configurations.
9. **Optimal Block Size Suggestion**: Identifies the best-performing block size based on GPU Execution Time.

## Requirements

- NVIDIA GPU with CUDA support
- NVIDIA CUDA Toolkit (including NCU)
- Python 3.6+
- Streamlit
- Pandas
- Matplotlib
- Seaborn
- NumPy
- Plotly

## Installation

1. Ensure CUDA Toolkit is installed and configured.
2. Install Python dependencies:

```bash
pip install streamlit pandas matplotlib seaborn numpy plotly
```

## Notes

- Ensure that the Heat Equation Solver binary is compiled and available in the parent directory.
- The profiling script may take a considerable amount of time to run, depending on the number of configurations tested.
- Adjust the block sizes in `profiling.sh` if you want to profile different configurations.

