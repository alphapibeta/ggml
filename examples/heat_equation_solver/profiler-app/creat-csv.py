import csv
import sys

# Function to clean and parse a line from the markdown table
def parse_line(line):
    # Remove the surrounding '|' and split by '|'
    return [item.strip() for item in line.strip('| \n').split('|')]

# Main function to convert markdown to CSV
def markdown_to_csv(md_file, csv_file):
    # Variables to store header and rows
    header = []
    rows = []

    # Open the markdown file and read line by line
    with open(md_file, 'r') as f:
        for line in f:
            if line.startswith('|---'):  # Skip the separator line
                continue
            elif line.startswith('|'):
                parsed_line = parse_line(line)
                if not header:
                    header = parsed_line
                else:
                    rows.append(parsed_line)

    # Write the extracted table to a CSV file
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)  # Write header
        writer.writerows(rows)   # Write rows

    print(f"Markdown table has been converted to {csv_file}")

# Check if the script is being run directly
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py input_markdown.md output_file.csv")
    else:
        md_file = sys.argv[1]
        csv_file = sys.argv[2]
        markdown_to_csv(md_file, csv_file)
