# DropletAnalyzer Parallel Processing PowerShell Script (DEBUG MODE)
# Usage: .\ProcessDropletImages_FULL.ps1 -MaxCores 4
# This version runs with --debug flag to export all intermediate processing outputs

param(
    [int]$MaxCores = 4,  # Number of CPU cores to use for parallel processing
    [string]$ImagePattern = "*.jpg"  # Image file pattern to match
)

# Configuration
$DropletAnalyzerExe = "C:\Dev\DropletAnalyzer\DropletAnalyzer_OCV\build\Debug\DropletAnalyzer.exe"

# Create output folder
$OutputFolder = "Outputs"

Write-Host "=== DropletAnalyzer Parallel Processing (DEBUG MODE) ==="
Write-Host "Current Directory: $(Get-Location)"
Write-Host "Output Folder: $OutputFolder"
Write-Host "Max Cores: $MaxCores"
Write-Host "Image Pattern: $ImagePattern"
Write-Host "DEBUG MODE: Intermediate outputs will be saved"
Write-Host ""

# Check if DropletAnalyzer executable exists
if (-not (Test-Path $DropletAnalyzerExe)) {
    Write-Error "DropletAnalyzer executable not found at: $DropletAnalyzerExe"
    Write-Error "Please update the DropletAnalyzerExe variable in the script."
    exit 1
}

# Create output folder if it doesn't exist
if (-not (Test-Path $OutputFolder)) {
    Write-Host "Creating output folder: $OutputFolder"
    New-Item -ItemType Directory -Path $OutputFolder -Force | Out-Null
}

# Get all image files in current directory
$ImageFiles = Get-ChildItem -Path "." -Filter $ImagePattern | Where-Object { $_.Extension -match '\.(jpg|jpeg|png|bmp|tiff)$' }

if ($ImageFiles.Count -eq 0) {
    Write-Warning "No image files found matching pattern: $ImagePattern"
    exit 0
}

Write-Host "Found $($ImageFiles.Count) image files to process"
Write-Host ""

# Process images in batches based on MaxCores
$TotalImages = $ImageFiles.Count
$ProcessedCount = 0
$StartTime = Get-Date

Write-Host "Starting parallel processing with debug outputs..."
Write-Host ""

for ($i = 0; $i -lt $TotalImages; $i += $MaxCores) {
    $BatchNumber = [Math]::Floor($i / $MaxCores) + 1
    $BatchEnd = [Math]::Min($i + $MaxCores - 1, $TotalImages - 1)
    $BatchSize = $BatchEnd - $i + 1
    
    Write-Host "=== Batch $BatchNumber (Images $($i + 1) to $($BatchEnd + 1)) ==="
    
    # Create jobs for parallel processing
    $Jobs = @()
    
    for ($j = $i; $j -le $BatchEnd; $j++) {
        $ImagePath = $ImageFiles[$j].FullName
        $ImageName = $ImageFiles[$j].Name
        Write-Host "Starting: $ImageName (with debug outputs)"
        
        $Job = Start-Job -ScriptBlock {
            param($ExePath, $ImgPath, $OutDir)
            & $ExePath "`"$ImgPath`"" "`"$OutDir`"" "--debug"
        } -ArgumentList $DropletAnalyzerExe, $ImagePath, (Resolve-Path $OutputFolder).Path
        
        $Jobs += $Job
    }
    
    # Wait for all jobs in this batch to complete
    Write-Host "Waiting for $BatchSize parallel processes to complete..."
    $Jobs | Wait-Job | Out-Null
    
    # Collect results
    foreach ($Job in $Jobs) {
        $Result = Receive-Job -Job $Job
        if ($Job.State -eq "Completed") {
            $ProcessedCount++
        } else {
            Write-Host "Job failed: $($Job.State)"
        }
        Remove-Job -Job $Job
    }
    
    Write-Host "Batch $BatchNumber completed. Progress: $ProcessedCount/$TotalImages"
    Write-Host ""
}

$EndTime = Get-Date
$Duration = $EndTime - $StartTime

Write-Host "=== Processing Complete ==="
Write-Host "Total Images Processed: $ProcessedCount/$TotalImages"
Write-Host "Total Time: $($Duration.ToString('hh\:mm\:ss'))"
Write-Host "Output Folder: $OutputFolder"
Write-Host ""

# Show summary of output files
$OutputFiles = Get-ChildItem -Path $OutputFolder -Filter "*.jpg"
Write-Host "Generated $($OutputFiles.Count) output images (including debug outputs):"

# Count different types of outputs
$QCFiles = $OutputFiles | Where-Object { $_.Name -match '_VALID_|_INVALID_' }
$UpscaleFiles = $OutputFiles | Where-Object { $_.Name -match '_UPSCALE' }
$BlurFiles = $OutputFiles | Where-Object { $_.Name -match '_BLUR' }
$ThreshFiles = $OutputFiles | Where-Object { $_.Name -match '_THRESH' }
$CirclesFiles = $OutputFiles | Where-Object { $_.Name -match '_ORIGINALCIRCLES' }

Write-Host "  QC Images: $($QCFiles.Count)"
Write-Host "  Upscale Images: $($UpscaleFiles.Count)"
Write-Host "  Blur Images: $($BlurFiles.Count)"
Write-Host "  Threshold Images: $($ThreshFiles.Count)"
Write-Host "  Original Circles Images: $($CirclesFiles.Count)"

Write-Host ""
Write-Host "Processing completed successfully!"
Write-Host "All intermediate processing outputs have been saved for debugging."
