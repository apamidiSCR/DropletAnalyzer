# DropletAnalyzer Command-Line Interface

## Overview
The DropletAnalyzer has been modified to work as a command-line executable that can be called from PowerShell scripts for parallel processing.

## Usage
```bash
DropletAnalyzer.exe <input_filepath> <output_folder>
```

### Parameters
- `input_filepath`: Full path to the image file to process
- `output_folder`: Directory where the QC output image will be saved

### Example
```bash
DropletAnalyzer.exe "C:\images\0_20241203-104610-955_91_00168.jpg" "C:\output"
```

## Output Filename Format
The output QC images are named using the following format:
```
<droplet_number>_<VALID/INVALID>_<diameter>_<mean_gray_value>.jpg
```

### Filename Components
- `droplet_number`: Extracted from the input filename (numbers after the last underscore)
- `VALID/INVALID`: Status based on droplet detection
  - `VALID`: Single droplet detected
  - `INVALID`: No droplets or multiple droplets detected
- `diameter`: Diameter in micrometers (1 pixel = 1 micrometer)
- `mean_gray_value`: Mean grayscale value within the droplet

### Decimal Formatting
Decimals are represented with 'p' instead of '.' to avoid filename issues:
- `32.38` becomes `32p38`
- `0.00` becomes `0p00`

### Example Output Filenames
- `168_VALID_45p67_123p45.jpg` - Valid droplet #168 with diameter 45.67μm and mean gray value 123.45
- `168_INVALID_0p00_0p00.jpg` - Invalid droplet #168 (no valid measurements)

## Input Filename Parsing
The program extracts the droplet number from filenames like:
- `0_20241203-104610-955_91_00168.jpg` → droplet number = 168
- `0_20241203-104614-957_90_00139.jpg` → droplet number = 139

## Error Handling
- Returns exit code 1 if input file doesn't exist
- Returns exit code 1 if output folder cannot be created
- Returns exit code 1 if incorrect number of arguments provided
- Creates output folder automatically if it doesn't exist

## Testing
Use the provided `test_cli.bat` file to test the command-line interface:
```bash
test_cli.bat
```

This will process the test images and save results to the `test_output` folder.

## PowerShell Integration
This executable is designed to be called from PowerShell scripts for parallel processing of multiple images. Each instance processes one image and outputs one QC file with descriptive metrics in the filename.

Example powershell command:
PowerShell -ExecutionPolicy Bypass -File ".\ProcessDropletImages.ps1" -MaxCores 2
# Run the powershell command from the directory containing the images.
