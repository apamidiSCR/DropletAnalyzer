# DropletAnalyzer Dataset Batch Processing PowerShell Script
# Usage: .\ProcessDropletDataset.ps1 -MaxCores 4
# 
# This script processes multiple experiment folders by running ProcessDropletImages.ps1
# in each subdirectory that contains images.

param(
    [int]$MaxCores = 4,  # Number of CPU cores to use for parallel processing
    [string]$ImagePattern = "*.jpg",  # Image file pattern to match
    [switch]$SkipExisting = $false  # Skip folders that already have an Outputs directory
)

# Configuration
$ProcessImagesScript = ".\ProcessDropletImages.ps1"

# Resolve the full path to ProcessDropletImages.ps1 so it works from any directory
$ProcessImagesScriptFullPath = Resolve-Path $ProcessImagesScript -ErrorAction SilentlyContinue

Write-Host "========================================"
Write-Host "  DropletAnalyzer Dataset Processor"
Write-Host "========================================"
Write-Host ""
Write-Host "Main Directory: $(Get-Location)"
Write-Host "Max Cores per Folder: $MaxCores"
Write-Host "Image Pattern: $ImagePattern"
Write-Host "Skip Existing: $SkipExisting"
Write-Host ""

# Check if ProcessDropletImages.ps1 exists
if (-not $ProcessImagesScriptFullPath -or -not (Test-Path $ProcessImagesScriptFullPath)) {
    Write-Error "ProcessDropletImages.ps1 not found in current directory!"
    Write-Error "Please ensure both scripts are in the same directory."
    exit 1
}

Write-Host "Using script: $ProcessImagesScriptFullPath"
Write-Host ""

# Get all subdirectories (experiment folders)
$ExperimentFolders = Get-ChildItem -Path "." -Directory | Where-Object { 
    $_.Name -ne "Outputs" -and $_.Name -notmatch "^\." 
}

if ($ExperimentFolders.Count -eq 0) {
    Write-Warning "No subdirectories found to process!"
    exit 0
}

Write-Host "Found $($ExperimentFolders.Count) experiment folder(s)"
Write-Host ""

# Filter folders that contain images
$FoldersToProcess = @()
foreach ($Folder in $ExperimentFolders) {
    $ImageFiles = Get-ChildItem -Path $Folder.FullName -Filter $ImagePattern -File | 
                  Where-Object { $_.Extension -match '\.(jpg|jpeg|png|bmp|tiff)$' }
    
    if ($ImageFiles.Count -gt 0) {
        # Check if Outputs folder already exists
        $OutputsPath = Join-Path $Folder.FullName "Outputs"
        if ($SkipExisting -and (Test-Path $OutputsPath)) {
            Write-Host "[SKIP] $($Folder.Name) - Outputs folder already exists"
        } else {
            $FoldersToProcess += @{
                Folder = $Folder
                ImageCount = $ImageFiles.Count
            }
            Write-Host "[QUEUE] $($Folder.Name) - $($ImageFiles.Count) image(s)"
        }
    } else {
        Write-Host "[SKIP] $($Folder.Name) - No images found"
    }
}

if ($FoldersToProcess.Count -eq 0) {
    Write-Warning "No folders with images to process!"
    exit 0
}

Write-Host ""
Write-Host "========================================"
Write-Host "Processing $($FoldersToProcess.Count) folder(s)..."
Write-Host "========================================"
Write-Host ""

$TotalFolders = $FoldersToProcess.Count
$ProcessedFolders = 0
$TotalImagesProcessed = 0
$StartTime = Get-Date
$FailedFolders = @()

foreach ($FolderInfo in $FoldersToProcess) {
    $Folder = $FolderInfo.Folder
    $ImageCount = $FolderInfo.ImageCount
    $ProcessedFolders++
    
    Write-Host "========================================"
    Write-Host "[$ProcessedFolders/$TotalFolders] Processing: $($Folder.Name)"
    Write-Host "Images: $ImageCount"
    Write-Host "========================================"
    Write-Host ""
    
    # Change to the experiment folder
    Push-Location $Folder.FullName
    
    try {
        # Run ProcessDropletImages.ps1 in this folder using the full path
        $Result = & $ProcessImagesScriptFullPath -MaxCores $MaxCores -ImagePattern $ImagePattern
        
        if ($LASTEXITCODE -eq 0 -or $null -eq $LASTEXITCODE) {
            Write-Host ""
            Write-Host "[SUCCESS] Completed: $($Folder.Name)"
            $TotalImagesProcessed += $ImageCount
        } else {
            Write-Warning "[FAILED] Error processing: $($Folder.Name)"
            $FailedFolders += $Folder.Name
        }
    }
    catch {
        Write-Error "[ERROR] Exception processing $($Folder.Name): $_"
        $FailedFolders += $Folder.Name
    }
    finally {
        # Return to main directory
        Pop-Location
    }
    
    Write-Host ""
}

$EndTime = Get-Date
$Duration = $EndTime - $StartTime

# Final summary
Write-Host "========================================"
Write-Host "         Processing Complete!"
Write-Host "========================================"
Write-Host ""
Write-Host "Total Folders Processed: $ProcessedFolders"
Write-Host "Total Images Processed: $TotalImagesProcessed"
Write-Host "Total Time: $($Duration.ToString('hh\:mm\:ss'))"
Write-Host ""

if ($FailedFolders.Count -gt 0) {
    Write-Host "Failed Folders ($($FailedFolders.Count)):"
    foreach ($Failed in $FailedFolders) {
        Write-Host "  - $Failed" -ForegroundColor Red
    }
    Write-Host ""
} else {
    Write-Host "All folders processed successfully!" -ForegroundColor Green
    Write-Host ""
}

# Show summary of all outputs
Write-Host "Summary of Generated Outputs:"
foreach ($FolderInfo in $FoldersToProcess) {
    $Folder = $FolderInfo.Folder
    $OutputsPath = Join-Path $Folder.FullName "Outputs"
    
    if (Test-Path $OutputsPath) {
        $OutputFiles = Get-ChildItem -Path $OutputsPath -Filter "*.jpg"
        Write-Host "  $($Folder.Name): $($OutputFiles.Count) QC images"
    } else {
        Write-Host "  $($Folder.Name): No Outputs folder" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Dataset processing completed!"

