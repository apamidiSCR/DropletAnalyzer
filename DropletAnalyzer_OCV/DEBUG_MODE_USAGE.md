# DropletAnalyzer Debug Mode

## Overview
The DropletAnalyzer now supports an optional `--debug` flag that exports intermediate processing outputs. This is useful for troubleshooting, parameter tuning, and understanding the processing pipeline.

## Usage

### Normal Mode (No Debug Output)
```bash
DropletAnalyzer.exe "input.jpg" "output_folder"
```

### Debug Mode (With Intermediate Outputs)
```bash
DropletAnalyzer.exe "input.jpg" "output_folder" --debug
```

Or use the short form:
```bash
DropletAnalyzer.exe "input.jpg" "output_folder" -d
```

## Debug Outputs

When debug mode is enabled, the following intermediate images are saved to the output folder:

### 1. Upscaled Image (`_UPSCALE.jpg`)
- **Description**: The original grayscale image upscaled by 2x for sub-pixel precision
- **Filename Example**: `166_UPSCALE.jpg`
- **Purpose**: Shows the initial preprocessing step before blur

### 2. Blurred Image (`_BLUR.jpg`)
- **Description**: The upscaled image after Gaussian blur (kernel size 25)
- **Filename Example**: `166_BLUR.jpg`
- **Purpose**: Shows noise reduction and smoothing applied before circle detection

### 3. Thresholded Image (`_THRESH.jpg`)
- **Description**: Binary image created using Otsu's thresholding
- **Filename Example**: `166_THRESH.jpg`
- **Purpose**: Shows the black/white segmentation used for boundary detection

### 4. Original Circles (`_ORIGINALCIRCLES.jpg`)
- **Description**: Initial Hough circle detections drawn on the blurred image
- **Filename Example**: `166_ORIGINALCIRCLES.jpg`
- **Purpose**: Shows all circles detected by HoughCircles before refinement
- **Visual**: Green circles with red center points

## Output Naming Convention

All debug outputs use the same droplet number as the main QC output:

```
<droplet_number>_<SUFFIX>.jpg
```

Examples for droplet #166:
- `166_UPSCALE.jpg`
- `166_BLUR.jpg`
- `166_THRESH.jpg`
- `166_ORIGINALCIRCLES.jpg`
- `166_VALID_45p67_123p45.jpg` (main QC output)

## PowerShell Script Integration

To enable debug mode in the PowerShell script, you would need to modify the executable call to include the `--debug` flag:

```powershell
& $DropletAnalyzerExe "`"$ImagePath`"" "`"$OutputDir`"" "--debug"
```

## Use Cases

1. **Troubleshooting**: Identify which processing step is causing issues
2. **Parameter Tuning**: Visualize effects of blur kernel size, threshold values, etc.
3. **Quality Control**: Verify that preprocessing steps are working correctly
4. **Documentation**: Generate examples of the processing pipeline for reports

## Performance Note

Debug mode adds minimal overhead since it only saves additional images. The processing pipeline remains unchanged.
