import os
import csv
import re
import argparse
from pathlib import Path


def parse_qc_filename(filename):
    """
    Parse QC image filename to extract index, validity, diameter, and gray value.
    
    Expected format:
    - VALID: <index>_VALID_<diameter>_<grayval>.jpg (e.g., "1_VALID_12p45_103p22.jpg")
    - INVALID (old): <index>_INVALID_0p00_0p00.jpg (e.g., "2_INVALID_0p00_0p00.jpg")
    - INVALID (new): <index>_INVALID.jpg (e.g., "2_INVALID.jpg")
    
    Returns:
        tuple: (index, valid, diameter, grayval) or None if parsing fails
    """
    # Remove file extension
    name = os.path.splitext(filename)[0]
    
    # First try the full pattern: <index>_<VALID|INVALID>_<diameter>_<grayval>
    pattern_full = r'^(\d+)_(VALID|INVALID)_([\d]+p[\d]+)_([\d]+p[\d]+)$'
    match = re.match(pattern_full, name)
    
    if match:
        index = int(match.group(1))
        valid = match.group(2)
        diameter_str = match.group(3)
        grayval_str = match.group(4)
        
        # Convert "p" notation back to decimal (e.g., "12p45" -> "12.45")
        diameter = diameter_str.replace('p', '.')
        grayval = grayval_str.replace('p', '.')
        
        # For INVALID images, set diameter and grayval to "NA"
        if valid == "INVALID":
            diameter = "NA"
            grayval = "NA"
        else:
            # Keep the original precision from the filename
            diameter = str(float(diameter))
            grayval = str(float(grayval))
        
        return (index, valid, diameter, grayval)
    
    # If full pattern doesn't match, try the simplified INVALID pattern: <index>_INVALID
    pattern_simple = r'^(\d+)_(INVALID)$'
    match = re.match(pattern_simple, name)
    
    if match:
        index = int(match.group(1))
        valid = match.group(2)
        # For simple INVALID format, set diameter and grayval to "NA"
        return (index, valid, "NA", "NA")
    
    return None


def load_experiment_template(template_csv_path):
    """
    Load the experiment template CSV and return a dictionary mapping
    experiment codes to their parameters.
    
    Returns:
        dict: {experiment_code: {param_name: value, ...}, ...}
    """
    experiments = {}
    
    # Try different encodings to handle BOM and other encoding issues
    encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(template_csv_path, 'r', newline='', encoding=encoding) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Strip whitespace from keys in case of formatting issues
                    row = {k.strip(): v for k, v in row.items()}
                    code = row['Code']
                    experiments[code] = row
            return experiments
        except (UnicodeDecodeError, KeyError):
            continue
    
    # If all encodings fail, raise an error with helpful message
    raise ValueError(f"Could not read CSV file. Please ensure it has a 'Code' column and is properly formatted.")


def find_outputs_folders(root_directory):
    """
    Find all 'Outputs' folders within subdirectories of the root directory.
    
    Returns:
        list: [(experiment_code, outputs_path), ...]
    """
    outputs_folders = []
    
    root_path = Path(root_directory)
    
    # Iterate through subdirectories
    for subdir in root_path.iterdir():
        if subdir.is_dir():
            experiment_code = subdir.name
            outputs_path = subdir / "Outputs"
            
            if outputs_path.exists() and outputs_path.is_dir():
                outputs_folders.append((experiment_code, outputs_path))
    
    return outputs_folders


def process_outputs_folder(experiment_code, outputs_path, experiment_params):
    """
    Process all QC images in an Outputs folder and extract data.
    
    Returns:
        list: [row_dict, ...] where each row_dict contains all columns for output CSV
    """
    rows = []
    
    # Get all image files
    image_files = []
    for file in outputs_path.iterdir():
        if file.is_file() and file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            # Only process files that match the QC naming pattern
            # Check for VALID files with full format, or INVALID files (both old and new formats)
            if ('_VALID_' in file.name or '_INVALID_' in file.name or 
                file.name.endswith('_INVALID.jpg')):
                image_files.append(file)
    
    # Sort by index number (extracted from filename)
    def get_index(filepath):
        match = re.match(r'^(\d+)_', filepath.name)
        if match:
            return int(match.group(1))
        return 0
    
    image_files.sort(key=get_index)
    
    # Process each image
    for image_file in image_files:
        parsed = parse_qc_filename(image_file.name)
        
        if parsed is None:
            print(f"Warning: Could not parse filename: {image_file.name}")
            continue
        
        index, valid, diameter, grayval = parsed
        
        # Create output row
        row = {
            'Code': experiment_code,
            'Foil': experiment_params.get('Foil', ''),
            'Femulsion': experiment_params.get('Femulsion', ''),
            'Finject': experiment_params.get('Finject', ''),
            'Voltage': experiment_params.get('Voltage', ''),
            'Measured?': experiment_params.get('Measured?', ''),
            'Index': index,
            'Valid?': valid,
            'Radius (um)': diameter,
            'GrayVal (A.U.)': grayval
        }
        
        rows.append(row)
    
    return rows


def main():
    """
    Main function to process experimental data and create output CSV.
    """
    parser = argparse.ArgumentParser(
        description='Tabulate droplet measurements from experimental data.'
    )
    parser.add_argument(
        'template_csv',
        type=str,
        help='Path to the experiment template CSV file'
    )
    parser.add_argument(
        'data_directory',
        type=str,
        help='Path to the directory containing experiment subfolders'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Path to output CSV file (default: <data_directory>/output.csv)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    template_csv_path = Path(args.template_csv)
    if not template_csv_path.exists():
        print(f"Error: Template CSV file not found: {args.template_csv}")
        return 1
    
    data_directory = Path(args.data_directory)
    if not data_directory.exists():
        print(f"Error: Data directory not found: {args.data_directory}")
        return 1
    
    # Set output path
    if args.output:
        output_csv_path = Path(args.output)
    else:
        output_csv_path = data_directory / "output.csv"
    
    print(f"Loading experiment template from: {template_csv_path}")
    experiment_params = load_experiment_template(template_csv_path)
    print(f"Loaded {len(experiment_params)} experiments from template")
    
    print(f"\nScanning for Outputs folders in: {data_directory}")
    outputs_folders = find_outputs_folders(data_directory)
    print(f"Found {len(outputs_folders)} Outputs folders")
    
    # Collect all rows
    all_rows = []
    
    for experiment_code, outputs_path in outputs_folders:
        print(f"\nProcessing: {experiment_code}")
        
        # Check if experiment code exists in template
        if experiment_code not in experiment_params:
            print(f"  Warning: Experiment code '{experiment_code}' not found in template CSV")
            print(f"  Skipping...")
            continue
        
        params = experiment_params[experiment_code]
        rows = process_outputs_folder(experiment_code, outputs_path, params)
        
        print(f"  Found {len(rows)} droplet measurements")
        all_rows.extend(rows)
    
    # Write output CSV
    print(f"\nWriting output to: {output_csv_path}")
    
    if all_rows:
        fieldnames = [
            'Code', 'Foil', 'Femulsion', 'Finject', 'Voltage', 'Measured?',
            'Index', 'Valid?', 'Radius (um)', 'GrayVal (A.U.)'
        ]
        
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        
        print(f"Successfully wrote {len(all_rows)} rows to output CSV")
    else:
        print("Warning: No data found to write")
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())

