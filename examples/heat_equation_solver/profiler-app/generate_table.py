import re
import sys
from typing import List, Dict

def extract_metrics(block_lines: List[str]) -> Dict[str, str]:
    metrics = {}
    kernel_type = None

    for line in block_lines:
        if "Grid size:" in line:
            metrics['Grid Size'] = line.split(":")[1].strip()
        elif "Block Size:" in line:
            metrics['Block Size'] = line.split(":")[1].strip().replace("x", " x ")
        elif "GPU Execution Time:" in line:
            ms_value = float(line.split(":")[1].replace("ms", "").strip())
            metrics['GPU Execution Time (us)'] = f"{ms_value * 1000:.2f} us"
        elif "Kernel Type:" in line:
            kernel_type = line.split(":")[1].strip()
            metrics['Kernel Type'] = kernel_type
        elif "Total GPU Execution Time" in line:
            ms_value = float(line.split(":")[1].replace("ms", "").strip())
            metrics['Total GPU Execution Time (us)'] = f"{ms_value * 1000:.2f} us"
        elif "DRAM Frequency" in line:
            metrics['DRAM Frequency (cycle/nsecond)'] = re.search(r"([\d.]+)", line).group(1)
        elif "SM Frequency" in line:
            metrics['SM Frequency (cycle/usecond)'] = re.search(r"([\d.]+)", line).group(1)
        elif "Elapsed Cycles" in line:
            metrics['Elapsed Cycles (cycle)'] = re.search(r"([\d,]+)", line).group(1).replace(',', '')
        elif "Memory Throughput" in line:
            metrics['Memory Throughput (%)'] = re.search(r"([\d.]+)", line).group(1)
        elif "DRAM Throughput" in line:
            metrics['DRAM Throughput (%)'] = re.search(r"([\d.]+)", line).group(1)
        elif "Duration" in line:
            duration_value = float(re.search(r"([\d.]+)", line).group(1))
            if "msecond" in line:
                metrics['Duration (us)'] = f"{duration_value * 1000:.2f} us"
            elif "usecond" in line:
                metrics['Duration (us)'] = f"{duration_value:.2f} us"
        elif "L1/TEX Cache Throughput" in line:
            metrics['L1/TEX Cache Throughput (%)'] = re.search(r"([\d.]+)", line).group(1)
        elif "L2 Cache Throughput" in line:
            metrics['L2 Cache Throughput (%)'] = re.search(r"([\d.]+)", line).group(1)
        elif "SM Active Cycles" in line:
            metrics['SM Active Cycles (cycle)'] = re.search(r"([\d,.]+)", line).group(1).replace(',', '')
        elif "Compute (SM) Throughput" in line:
            metrics['Compute (SM) Throughput (%)'] = re.search(r"([\d.]+)", line).group(1)
        elif "OPT   This kernel exhibits low compute throughput" in line:
            metrics['Low Compute Throughput Warning'] = "Yes"
        elif "Launch Statistics" in line:
            continue  # To ensure Launch Block Size and Grid Size don't get overwritten
        elif "Block Size" in line and "Function Cache Configuration" in line:
            metrics['Launch Block Size'] = re.search(r"([\d]+)", line).group(1)
        elif "Grid Size" in line and "Launch" in line:
            metrics['Launch Grid Size'] = re.search(r"([\d,]+)", line).group(1).replace(',', '')
        elif "Registers Per Thread" in line:
            metrics['Registers Per Thread'] = re.search(r"([\d.]+)", line).group(1)
        elif "Shared Memory Configuration Size" in line:
            metrics['Shared Memory Configuration Size (Kbyte)'] = re.search(r"([\d.]+)", line).group(1)
        elif "Driver Shared Memory Per Block" in line:
            metrics['Driver Shared Memory Per Block (byte)'] = re.search(r"([\d.]+)", line).group(1)
        elif "Dynamic Shared Memory Per Block" in line:
            metrics['Dynamic Shared Memory Per Block (byte)'] = re.search(r"([\d.]+)", line).group(1)
        elif "Static Shared Memory Per Block" in line:
            metrics['Static Shared Memory Per Block (byte)'] = re.search(r"([\d.]+)", line).group(1)
        elif "Threads" in line and "Launch" in line:
            metrics['Threads Per Block'] = re.search(r"([\d,]+)", line).group(1).replace(',', '')
        elif "Waves Per SM" in line:
            metrics['Waves Per SM'] = re.search(r"([\d.]+)", line).group(1)
        elif "Block Limit SM" in line:
            metrics['Block Limit SM'] = re.search(r"([\d.]+)", line).group(1)
        elif "Block Limit Registers" in line:
            metrics['Block Limit Registers'] = re.search(r"([\d.]+)", line).group(1)
        elif "Block Limit Shared Mem" in line:
            metrics['Block Limit Shared Mem'] = re.search(r"([\d.]+)", line).group(1)
        elif "Block Limit Warps" in line:
            metrics['Block Limit Warps'] = re.search(r"([\d.]+)", line).group(1)
        elif "Theoretical Active Warps per SM" in line:
            metrics['Theoretical Active Warps per SM'] = re.search(r"([\d.]+)", line).group(1)
        elif "Theoretical Occupancy" in line:
            metrics['Theoretical Occupancy (%)'] = re.search(r"([\d.]+)", line).group(1)
        elif "Achieved Occupancy" in line:
            metrics['Achieved Occupancy (%)'] = re.search(r"([\d.]+)", line).group(1)
        elif "Achieved Active Warps Per SM" in line:
            metrics['Achieved Active Warps Per SM'] = re.search(r"([\d.]+)", line).group(1)
        elif "OPT   Est. Speedup:" in line:
            match = re.search(r"Speedup: ([\d.]+%)", line)
            if match:
                metrics['Estimated Speedup (%)'] = match.group(1)

    return metrics

def parse_ncu_results(input_file: str) -> List[Dict[str, str]]:
    results = []
    try:
        with open(input_file, 'r') as infile:
            lines = infile.readlines()
            block_start_indices = [i for i, line in enumerate(lines) if "Running ncu with kernel" in line]
            for i, start_index in enumerate(block_start_indices):
                end_index = block_start_indices[i + 1] if i + 1 < len(block_start_indices) else len(lines)
                block_lines = lines[start_index:end_index]
                metrics = extract_metrics(block_lines)
                if 'Block Size' in metrics:
                    results.append(metrics)

        return results
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return []
    except Exception as e:
        print(f"Error parsing NCU results: {str(e)}")
        return []

def generate_markdown_table(results: List[Dict[str, str]], output_file: str):
    headers = [
        "Block Size", "Grid Size", "GPU Execution Time (us)", "Kernel Type", 
        "Total GPU Execution Time (us)", "DRAM Frequency (cycle/nsecond)", 
        "SM Frequency (cycle/usecond)", "Elapsed Cycles (cycle)", 
        "Memory Throughput (%)", "DRAM Throughput (%)", "Duration (us)", 
        "L1/TEX Cache Throughput (%)", "L2 Cache Throughput (%)", 
        "SM Active Cycles (cycle)", "Compute (SM) Throughput (%)", 
        "Low Compute Throughput Warning", "Launch Block Size", "Launch Grid Size", 
        "Registers Per Thread", "Shared Memory Configuration Size (Kbyte)", 
        "Driver Shared Memory Per Block (byte)", "Dynamic Shared Memory Per Block (byte)", 
        "Static Shared Memory Per Block (byte)", "Threads Per Block", "Waves Per SM", 
        "Block Limit SM", "Block Limit Registers", "Block Limit Shared Mem", 
        "Block Limit Warps", "Theoretical Active Warps per SM", 
        "Theoretical Occupancy (%)", "Achieved Occupancy (%)", 
        "Achieved Active Warps Per SM", "Estimated Speedup (%)"
    ]
    
    try:
        with open(output_file, 'w') as outfile:
            outfile.write("# NCU Results Full Comparison\n\n")
            outfile.write("| " + " | ".join(headers) + " |\n")
            outfile.write("|" + "|".join(["---" for _ in headers]) + "|\n")
            
            for result in results:
                row = [result.get(header, 'N/A') for header in headers]
                outfile.write("| " + " | ".join(row) + " |\n")
        print(f"Markdown file '{output_file}' with full comparison generated successfully.")
    except Exception as e:
        print(f"Error generating markdown table: {str(e)}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    results = parse_ncu_results(input_file)
    if results:
        generate_markdown_table(results, output_file)
    else:
        print("No results to process. Exiting.")

if __name__ == "__main__":
    main()
