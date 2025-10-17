#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <iomanip>
#include <filesystem>
#include <sstream>
#include <cmath>
#include <fstream>

// Define M_PI if not available (common on Windows MSVC)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Function to convert to grayscale
cv::Mat convertToGrayscale(const cv::Mat& input) {
    cv::Mat gray;
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

// Function to upscale image for sub-pixel precision
cv::Mat upscaleForPrecision(const cv::Mat& input, double scaleFactor = 4.0) {
    cv::Mat upscaled;
    cv::resize(input, upscaled, cv::Size(), scaleFactor, scaleFactor, cv::INTER_LINEAR);
    return upscaled;
}

// Function to scale circles back to original image size
std::vector<cv::Vec3f> scaleCirclesBack(const std::vector<cv::Vec3f>& circles, double scaleFactor = 4.0) {
    std::vector<cv::Vec3f> scaledCircles;
    for (const auto& circle : circles) {
        scaledCircles.push_back(cv::Vec3f(
            static_cast<float>(circle[0] / scaleFactor),
            static_cast<float>(circle[1] / scaleFactor),
            static_cast<float>(circle[2] / scaleFactor)
        ));
    }
    return scaledCircles;
}

// Function to apply Gaussian blur for noise reduction
cv::Mat applyGaussianBlur(const cv::Mat& input, int kernelSize = 5) {
    cv::Mat blurred;
    cv::GaussianBlur(input, blurred, cv::Size(kernelSize, kernelSize), 0);
    return blurred;
}

// Function to enhance contrast using CLAHE
cv::Mat enhanceContrast(const cv::Mat& input, double clipLimit = 2.0, int tileGridSize = 8) {
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clipLimit, cv::Size(tileGridSize, tileGridSize));
    cv::Mat enhanced;
    clahe->apply(input, enhanced);
    return enhanced;
}

// Function to calculate mean gray value from corner regions
cv::Scalar calculateCornerBasedMean(const cv::Mat& input, int cornerSize = -1) {
    if (cornerSize == -1) {
        cornerSize = static_cast<int>(std::min(input.cols, input.rows) * 0.1);
    }
    cornerSize = std::max(1, std::min(cornerSize, std::min(input.cols, input.rows) / 2));
    
    cv::Rect topLeft(0, 0, cornerSize, cornerSize);
    cv::Rect topRight(input.cols - cornerSize, 0, cornerSize, cornerSize);
    cv::Rect bottomLeft(0, input.rows - cornerSize, cornerSize, cornerSize);
    cv::Rect bottomRight(input.cols - cornerSize, input.rows - cornerSize, cornerSize, cornerSize);
    
    cv::Scalar meanTopLeft = cv::mean(input(topLeft));
    cv::Scalar meanTopRight = cv::mean(input(topRight));
    cv::Scalar meanBottomLeft = cv::mean(input(bottomLeft));
    cv::Scalar meanBottomRight = cv::mean(input(bottomRight));
    
    cv::Scalar avgMean;
    avgMean[0] = (meanTopLeft[0] + meanTopRight[0] + meanBottomLeft[0] + meanBottomRight[0]) / 4.0;
    return avgMean;
}

// Function to detect circles using Hough Transform
std::vector<cv::Vec3f> detectCircles(const cv::Mat& input, double dp = 1.0, double minDist = 50, 
    double param1 = 100, double param2 = 30, 
    int minRadius = 20, int maxRadius = 100) {
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(input, circles, cv::HOUGH_GRADIENT, dp, minDist, param1, param2, minRadius, maxRadius);
    return circles;
}

// Enum to represent droplet type
enum DropletType {
    RING_TYPE,
    FILLED_TYPE
};

// Structure to hold circle results
struct CircleResult {
    cv::Vec3f circle;
    double score;
    int blackPixels;
    int translationDistance;
    int outerRadius;
    
    CircleResult(cv::Vec3f c, double s, int bp, int td = 0, int or = 0) : circle(c), score(s), blackPixels(bp), translationDistance(td), outerRadius(or) {}
};

// Function to count black pixels in a circle
int countBlackPixelsInCircle(const cv::Mat& thresholded, const cv::Point& center, int radius) {
    cv::Mat circleMask = cv::Mat::zeros(thresholded.size(), CV_8UC1);
    cv::circle(circleMask, center, radius, cv::Scalar(255), -1);
    
    int blackCount = 0;
    for (int y = 0; y < thresholded.rows; y++) {
        for (int x = 0; x < thresholded.cols; x++) {
            if (circleMask.at<uchar>(y, x) == 255 && thresholded.at<uchar>(y, x) == 0) {
                    blackCount++;
                }
            }
        }
    return blackCount;
}

// Function to count white pixels in a ring
int countWhitePixelsInRing(const cv::Mat& thresholded, const cv::Point& center, int outerRadius, int innerRadius) {
    cv::Mat ringMask = cv::Mat::zeros(thresholded.size(), CV_8UC1);
    cv::circle(ringMask, center, outerRadius, cv::Scalar(255), -1);
    cv::circle(ringMask, center, innerRadius, cv::Scalar(0), -1);
    
    int whiteCount = 0;
    for (int y = 0; y < thresholded.rows; y++) {
        for (int x = 0; x < thresholded.cols; x++) {
            if (ringMask.at<uchar>(y, x) == 255 && thresholded.at<uchar>(y, x) == 255) {
                whiteCount++;
            }
        }
    }
    return whiteCount;
}

// Function to determine droplet type
DropletType determineDropletType(const cv::Mat& thresholded, const cv::Point& center, int radius, int ringThickness) {
    int innerRadius = radius - ringThickness;
    int outerRadius = radius + ringThickness;
    
    // Count pixels in different regions
    int blackInCenter = countBlackPixelsInCircle(thresholded, center, innerRadius);
    int whiteInRing = countWhitePixelsInRing(thresholded, center, outerRadius, innerRadius);
    
    // If there are more white pixels in the ring than black in center, it's a ring type
    return (whiteInRing > blackInCenter) ? RING_TYPE : FILLED_TYPE;
}

// Crossing detection structures
struct CrossingCounts {
    int horizontal;
    int vertical;
    int diagonal;
    std::vector<cv::Point> horizontalCrossings;
    std::vector<cv::Point> verticalCrossings;
    std::vector<cv::Point> diagonalCrossings;
};

CrossingCounts detectCrossings(const cv::Mat& thresholded) {
    CrossingCounts counts = {0, 0, 0, {}, {}, {}};
    
    // Horizontal crossings: scan ONE middle row
    int middleRow = thresholded.rows / 2;
    uchar prevPixelH = thresholded.at<uchar>(middleRow, 0);
    for (int x = 1; x < thresholded.cols; x++) {
        uchar currentPixel = thresholded.at<uchar>(middleRow, x);
        if (currentPixel != prevPixelH) {
            counts.horizontal++;
            counts.horizontalCrossings.push_back(cv::Point(x, middleRow));
            prevPixelH = currentPixel;
        }
    }
    
    // Vertical crossings: scan ONE middle column
    int middleCol = thresholded.cols / 2;
    uchar prevPixelV = thresholded.at<uchar>(0, middleCol);
    for (int y = 1; y < thresholded.rows; y++) {
        uchar currentPixel = thresholded.at<uchar>(y, middleCol);
        if (currentPixel != prevPixelV) {
            counts.vertical++;
            counts.verticalCrossings.push_back(cv::Point(middleCol, y));
            prevPixelV = currentPixel;
        }
    }
    
    // Diagonal crossings: scan both main diagonals
    int minDim = std::min(thresholded.rows, thresholded.cols);
    uchar prevPixelD1 = thresholded.at<uchar>(0, 0);
    for (int i = 1; i < minDim; i++) {
        uchar currentPixel = thresholded.at<uchar>(i, i);
        if (currentPixel != prevPixelD1) {
            counts.diagonal++;
            counts.diagonalCrossings.push_back(cv::Point(i, i));
            prevPixelD1 = currentPixel;
        }
    }
    
    uchar prevPixelD2 = thresholded.at<uchar>(0, thresholded.cols - 1);
    for (int i = 1; i < minDim; i++) {
        uchar currentPixel = thresholded.at<uchar>(i, thresholded.cols - 1 - i);
        if (currentPixel != prevPixelD2) {
            counts.diagonal++;
            counts.diagonalCrossings.push_back(cv::Point(thresholded.cols - 1 - i, i));
            prevPixelD2 = currentPixel;
        }
    }
    
    return counts;
}

// Threshold type classification
enum ThresholdType {
    TYPE_1_FILLED,
    TYPE_2_RING,
    TYPE_3_WEIRD
};

ThresholdType classifyThresholdType(const cv::Mat& thresholded) {
    CrossingCounts counts = detectCrossings(thresholded);
    
    if ((counts.horizontal + counts.vertical) * 2 < counts.diagonal) {
        return TYPE_3_WEIRD;
    }
    if (counts.horizontal <= 4 && counts.vertical <= 4 && counts.diagonal <= 8) {
        return TYPE_1_FILLED;
    }
    return TYPE_2_RING;
}

// Function to extract droplet number from filename
int extractDropletNumber(const std::string& filepath) {
    std::filesystem::path path(filepath);
    std::string filename = path.filename().string();
    
    // Find the last underscore and extract the number after it
    size_t lastUnderscore = filename.find_last_of('_');
    if (lastUnderscore != std::string::npos) {
        std::string numberStr = filename.substr(lastUnderscore + 1);
        // Remove file extension
        size_t dotPos = numberStr.find_last_of('.');
        if (dotPos != std::string::npos) {
            numberStr = numberStr.substr(0, dotPos);
        }
    try {
        return std::stoi(numberStr);
    } catch (const std::exception&) {
            return 0;
    }
    }
    return 0;
}

// Function to format diameter for filename
std::string formatDiameter(double diameter) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << diameter;
    return oss.str();
}

// Function to format mean gray value for filename
std::string formatMeanGrayValue(double meanGray) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << meanGray;
    return oss.str();
}

// Function to format numbers with 'p' instead of '.' for filenames
std::string formatNumberForFilename(double value) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << value;
    std::string str = oss.str();
    // Replace '.' with 'p'
    size_t dotPos = str.find('.');
    if (dotPos != std::string::npos) {
        str[dotPos] = 'p';
    }
    return str;
}

// Function to validate circles (multiple circles or out of bounds)
bool validateCircles(const std::vector<cv::Vec3f>& circles, int dropletNumber, const std::string& outputFolder, const std::string& debugFolder, 
                    const cv::Mat& originalImage, const cv::Mat& enhancedImage, int padding, double scaleFactor, bool debugMode) {
    // Check for multiple circles
    if (circles.size() > 1) {
        // Create invalid image with red circles (always save to main output folder)
        cv::Mat invalidImg = originalImage.clone(); // Use original image, not enhanced
        if (invalidImg.channels() == 1) {
            cv::cvtColor(invalidImg, invalidImg, cv::COLOR_GRAY2BGR);
        }
        for (const auto& circle : circles) {
            // Remove padding offset first, then scale back to original image size
            cv::Point center(cvRound((circle[0] - padding) / scaleFactor), cvRound((circle[1] - padding) / scaleFactor));
            int radius = static_cast<int>(circle[2] / scaleFactor);
            cv::circle(invalidImg, center, radius, cv::Scalar(0, 0, 255), 2); // Red
        }
        std::string invalidFilename = outputFolder + "/" + std::to_string(dropletNumber) + "_INVALID.jpg";
        cv::imwrite(invalidFilename, invalidImg);
        
        if (debugMode) {
            // Also save debug version
            cv::Mat debugInvalidImg = enhancedImage.clone();
            if (debugInvalidImg.channels() == 1) {
                cv::cvtColor(debugInvalidImg, debugInvalidImg, cv::COLOR_GRAY2BGR);
            }
            for (const auto& circle : circles) {
                cv::Point center(cvRound(circle[0]), cvRound(circle[1]));
                int radius = cvRound(circle[2]);
                cv::circle(debugInvalidImg, center, radius, cv::Scalar(0, 0, 255), 2); // Red
            }
            std::string debugInvalidFilename = debugFolder + "/" + std::to_string(dropletNumber) + "_INVALID.jpg";
            cv::imwrite(debugInvalidFilename, debugInvalidImg);
            
            // Update debug.txt
            std::ofstream debugFile(debugFolder + "/debug.txt", std::ios::app);
            debugFile << "VALIDATION: FAILED - Multiple circles (" << circles.size() << ")" << std::endl;
            debugFile.close();
        }
        return false;
    }
    
    if (circles.empty()) {
        // Create invalid image with no circles (always save to main output folder)
        cv::Mat invalidImg = originalImage.clone();
        if (invalidImg.channels() == 1) {
            cv::cvtColor(invalidImg, invalidImg, cv::COLOR_GRAY2BGR);
        }
        // No circles to draw, just save the original image
        std::string invalidFilename = outputFolder + "/" + std::to_string(dropletNumber) + "_INVALID.jpg";
        cv::imwrite(invalidFilename, invalidImg);
        
        if (debugMode) {
            std::ofstream debugFile(debugFolder + "/debug.txt", std::ios::app);
            debugFile << "VALIDATION: FAILED - No circles detected" << std::endl;
            debugFile.close();
        }
        return false;
    }
    
    // Check if single circle is out of bounds (10% margin from unpadded image)
    cv::Vec3f circle = circles[0];
    cv::Point center(cvRound(circle[0]), cvRound(circle[1]));
    int radius = cvRound(circle[2]);
    
    // Calculate unpadded image bounds (remove padding)
    int unpaddedWidth = enhancedImage.cols - (2 * padding);
    int unpaddedHeight = enhancedImage.rows - (2 * padding);
    // Use a smaller, fixed margin instead of percentage-based
    int margin = 5; // Reduced margin to 5 pixels
    
    // Check bounds - circle can extend up to 10 pixels beyond the unpadded boundary
    bool outOfBounds = false;
    if (center.x - radius < padding - margin || 
        center.x + radius > padding + unpaddedWidth + margin ||
        center.y - radius < padding - margin || 
        center.y + radius > padding + unpaddedHeight + margin) {
        outOfBounds = true;
    }
    
    // Debug: Let's also check if the issue is with the coordinate system
    // The center should be roughly in the middle of the unpadded region
    int expectedCenterX = padding + unpaddedWidth / 2;
    int expectedCenterY = padding + unpaddedHeight / 2;
    
    if (outOfBounds) {
        // Create invalid image with orange circle (always save to main output folder)
        cv::Mat invalidImg = originalImage.clone();
        if (invalidImg.channels() == 1) {
            cv::cvtColor(invalidImg, invalidImg, cv::COLOR_GRAY2BGR);
        }
        // Remove padding offset first, then scale back to original image size
        cv::Point scaledCenter(cvRound((center.x - padding) / scaleFactor), cvRound((center.y - padding) / scaleFactor));
        int scaledRadius = static_cast<int>(radius / scaleFactor);
        cv::circle(invalidImg, scaledCenter, scaledRadius, cv::Scalar(0, 165, 255), 2); // Orange
        std::string invalidFilename = outputFolder + "/" + std::to_string(dropletNumber) + "_INVALID.jpg";
        cv::imwrite(invalidFilename, invalidImg);
        
        if (debugMode) {
            // Also save debug version
            cv::Mat debugInvalidImg = enhancedImage.clone();
            if (debugInvalidImg.channels() == 1) {
                cv::cvtColor(debugInvalidImg, debugInvalidImg, cv::COLOR_GRAY2BGR);
            }
            cv::circle(debugInvalidImg, center, radius, cv::Scalar(0, 165, 255), 2); // Orange
            std::string debugInvalidFilename = debugFolder + "/" + std::to_string(dropletNumber) + "_INVALID.jpg";
            cv::imwrite(debugInvalidFilename, debugInvalidImg);
            
            // Update debug.txt
            std::ofstream debugFile(debugFolder + "/debug.txt", std::ios::app);
            debugFile << "VALIDATION: FAILED - Out of bounds" << std::endl;
            debugFile << "Center: (" << center.x << "," << center.y << "), Radius: " << radius << std::endl;
            debugFile << "Expected center: (" << expectedCenterX << "," << expectedCenterY << ")" << std::endl;
            debugFile << "Padding: " << padding << ", Unpadded: " << unpaddedWidth << "x" << unpaddedHeight << std::endl;
            debugFile << "Margin: " << margin << std::endl;
            debugFile << "Valid region: x=[" << (padding - margin) << "," << (padding + unpaddedWidth + margin) << "], y=[" << (padding - margin) << "," << (padding + unpaddedHeight + margin) << "]" << std::endl;
            debugFile << "Circle bounds: x=[" << (center.x - radius) << "," << (center.x + radius) << "], y=[" << (center.y - radius) << "," << (center.y + radius) << "]" << std::endl;
            debugFile.close();
        }
        return false;
    }
    
    if (debugMode) {
        std::ofstream debugFile(debugFolder + "/debug.txt", std::ios::app);
        debugFile << "VALIDATION: PASSED" << std::endl;
        debugFile << "Center: (" << center.x << "," << center.y << "), Radius: " << radius << std::endl;
        debugFile << "Expected center: (" << expectedCenterX << "," << expectedCenterY << ")" << std::endl;
        debugFile << "Padding: " << padding << ", Unpadded: " << unpaddedWidth << "x" << unpaddedHeight << std::endl;
        debugFile << "Margin: " << margin << std::endl;
        debugFile << "Valid region: x=[" << (padding - margin) << "," << (padding + unpaddedWidth + margin) << "], y=[" << (padding - margin) << "," << (padding + unpaddedHeight + margin) << "]" << std::endl;
        debugFile << "Circle bounds: x=[" << (center.x - radius) << "," << (center.x + radius) << "], y=[" << (center.y - radius) << "," << (center.y + radius) << "]" << std::endl;
        debugFile.close();
    }
    return true;
}

// Structure to hold crossing analysis results
struct CrossingAnalysis {
    std::vector<int> upDistances;
    std::vector<int> downDistances;
    std::vector<int> leftDistances;
    std::vector<int> rightDistances;
    int upCrossings;
    int downCrossings;
    int leftCrossings;
    int rightCrossings;
    bool isRing2B;
};

// Function to scan for crossings in a direction
std::vector<int> scanForCrossings(const cv::Mat& thresholded, cv::Point center, int direction, int maxDistance) {
    std::vector<int> crossings;
    uchar prevPixel = thresholded.at<uchar>(center.y, center.x);
    
    for (int i = 1; i <= maxDistance; i++) {
        cv::Point testPoint;
        switch (direction) {
            case 0: testPoint = cv::Point(center.x, center.y - i); break; // Up
            case 1: testPoint = cv::Point(center.x, center.y + i); break; // Down
            case 2: testPoint = cv::Point(center.x - i, center.y); break; // Left
            case 3: testPoint = cv::Point(center.x + i, center.y); break; // Right
        }
        
        if (testPoint.x < 0 || testPoint.x >= thresholded.cols || 
            testPoint.y < 0 || testPoint.y >= thresholded.rows) {
            break;
        }
        
        uchar currentPixel = thresholded.at<uchar>(testPoint.y, testPoint.x);
        if (currentPixel != prevPixel) {
            crossings.push_back(i);
            prevPixel = currentPixel;
        }
    }
    
    return crossings;
}

// Function to analyze ring structure for TYPE_2_RING
CrossingAnalysis analyzeRingStructure(const cv::Mat& thresholded, cv::Point center) {
    CrossingAnalysis analysis;
    
    // Scan in all four directions
    int maxDistance = std::min(thresholded.rows, thresholded.cols) / 2;
    
    analysis.upDistances = scanForCrossings(thresholded, center, 0, maxDistance);
    analysis.downDistances = scanForCrossings(thresholded, center, 1, maxDistance);
    analysis.leftDistances = scanForCrossings(thresholded, center, 2, maxDistance);
    analysis.rightDistances = scanForCrossings(thresholded, center, 3, maxDistance);
    
    analysis.upCrossings = static_cast<int>(analysis.upDistances.size());
    analysis.downCrossings = static_cast<int>(analysis.downDistances.size());
    analysis.leftCrossings = static_cast<int>(analysis.leftDistances.size());
    analysis.rightCrossings = static_cast<int>(analysis.rightDistances.size());
    
    // Determine if it's RING_2_B (has 3 crossings in any direction)
    analysis.isRing2B = (analysis.upCrossings == 3 || analysis.downCrossings == 3 || 
                        analysis.leftCrossings == 3 || analysis.rightCrossings == 3);
    
    return analysis;
}

// Function to find optimal center and radius for ring
cv::Vec3f findOptimalRingCenterAndRadius(const CrossingAnalysis& analysis, cv::Point originalCenter, bool isRing2B) {
    // Determine which crossing to use (first for RING_2_A, second for RING_2_B)
    int targetIndex = isRing2B ? 1 : 0;
    
    // Collect valid crossing distances for each direction
    std::vector<int> upDist, downDist, leftDist, rightDist;
    
    if (analysis.upCrossings > targetIndex && targetIndex < analysis.upDistances.size()) {
        upDist.push_back(analysis.upDistances[targetIndex]);
    }
    if (analysis.downCrossings > targetIndex && targetIndex < analysis.downDistances.size()) {
        downDist.push_back(analysis.downDistances[targetIndex]);
    }
    if (analysis.leftCrossings > targetIndex && targetIndex < analysis.leftDistances.size()) {
        leftDist.push_back(analysis.leftDistances[targetIndex]);
    }
    if (analysis.rightCrossings > targetIndex && targetIndex < analysis.rightDistances.size()) {
        rightDist.push_back(analysis.rightDistances[targetIndex]);
    }
    
    // Calculate optimal center by balancing distances
    cv::Point optimalCenter = originalCenter;
    
    // If we have both vertical directions, balance them
    if (!upDist.empty() && !downDist.empty()) {
        int verticalOffset = (downDist[0] - upDist[0]) / 2;
        optimalCenter.y += verticalOffset;
    }
    
    // If we have both horizontal directions, balance them
    if (!leftDist.empty() && !rightDist.empty()) {
        int horizontalOffset = (rightDist[0] - leftDist[0]) / 2;
        optimalCenter.x += horizontalOffset;
    }
    
    // Calculate radius using opposite directions
    double radius = 0.0;
    int radiusCount = 0;
    
    // Vertical radius (top-bottom)
    if (!upDist.empty() && !downDist.empty()) {
        radius += (upDist[0] + downDist[0]) / 2.0;
        radiusCount++;
    }
    
    // Horizontal radius (left-right)
    if (!leftDist.empty() && !rightDist.empty()) {
        radius += (leftDist[0] + rightDist[0]) / 2.0;
        radiusCount++;
    }
    
    // If we don't have opposite directions, use individual distances
    if (radiusCount == 0) {
        if (!upDist.empty()) radius += upDist[0];
        if (!downDist.empty()) radius += downDist[0];
        if (!leftDist.empty()) radius += leftDist[0];
        if (!rightDist.empty()) radius += rightDist[0];
        
        int totalDirections = (!upDist.empty() ? 1 : 0) + (!downDist.empty() ? 1 : 0) + 
                             (!leftDist.empty() ? 1 : 0) + (!rightDist.empty() ? 1 : 0);
        if (totalDirections > 0) {
            radius /= totalDirections;
        } else {
            radius = 50.0; // Fallback
        }
    } else {
        radius /= radiusCount;
    }
    
    return cv::Vec3f(static_cast<float>(optimalCenter.x), static_cast<float>(optimalCenter.y), static_cast<float>(radius));
}

// Function to process a single image
void processImage(const std::string& imagePath, const std::string& outputFolder, bool debugMode = false) {
    std::filesystem::path path(imagePath);
    std::string imageName = path.filename().string();
    int dropletNumber = extractDropletNumber(imagePath);
    
    // Processing image (debug info in debug.txt)
    
    // Create debug subfolder if in debug mode
    std::string debugFolder = outputFolder;
    if (debugMode) {
        debugFolder = outputFolder + "/" + std::to_string(dropletNumber) + "_debug";
        std::filesystem::create_directories(debugFolder);
    }
    
    // Load the image
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "ERROR: Could not load image from " << imagePath << std::endl;
        return;
    }
    
    // Convert to grayscale and upscale
    cv::Mat gray = convertToGrayscale(image);
    double scaleFactor = 2.0;
    cv::Mat upscaledGray = upscaleForPrecision(gray, scaleFactor);
    
    // Add padding and apply Gaussian blur
    int padding = static_cast<int>(50 * scaleFactor);
    cv::Scalar paddingColor = calculateCornerBasedMean(upscaledGray);
    cv::Mat padded;
    cv::copyMakeBorder(upscaledGray, padded, padding, padding, padding, padding, cv::BORDER_CONSTANT, paddingColor);
    cv::Mat blurred = applyGaussianBlur(padded, 25);
    
    // Enhance contrast and threshold
    cv::Mat enhanced = enhanceContrast(blurred);
    cv::Mat thresholded;
    cv::threshold(enhanced, thresholded, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    
    // Classify threshold type
    ThresholdType thresholdType = classifyThresholdType(thresholded);
    // Threshold type determined (logged to debug.txt)
    
    // Create debug.txt file with key information
    if (debugMode) {
        std::ofstream debugFile(debugFolder + "/debug.txt");
        debugFile << "Droplet: " << dropletNumber << std::endl;
        debugFile << "Threshold Type: " << (thresholdType == TYPE_1_FILLED ? "TYPE_1_FILLED" : 
                                          thresholdType == TYPE_2_RING ? "TYPE_2_RING" : "TYPE_3_WEIRD") << std::endl;
        debugFile.close();
    }
    
    // Save debug images
    if (debugMode) {
        // Save original image
        std::string originalFilename = debugFolder + "/" + std::to_string(dropletNumber) + "_ORIGINAL.jpg";
        cv::imwrite(originalFilename, image);
        
        std::string threshFilename = debugFolder + "/" + std::to_string(dropletNumber) + "_THRESH.jpg";
        cv::imwrite(threshFilename, thresholded);
        
        // Crossing visualization
        cv::Mat crossingImg = thresholded.clone();
        if (crossingImg.channels() == 1) {
            cv::cvtColor(crossingImg, crossingImg, cv::COLOR_GRAY2BGR);
        }
        CrossingCounts counts = detectCrossings(thresholded);
        for (const auto& point : counts.horizontalCrossings) { 
            cv::circle(crossingImg, point, 2, cv::Scalar(0, 255, 255), -1); // Yellow
        }
        for (const auto& point : counts.verticalCrossings) { 
            cv::circle(crossingImg, point, 2, cv::Scalar(0, 255, 255), -1); // Yellow
        }
        for (const auto& point : counts.diagonalCrossings) { 
            cv::circle(crossingImg, point, 2, cv::Scalar(255, 0, 255), -1); // Light purple
        }
        std::string crossingFilename = debugFolder + "/" + std::to_string(dropletNumber) + "_CROSSINGS.jpg";
        cv::imwrite(crossingFilename, crossingImg);
    }
    
    
    // Initial circle detection
    int scaledMinDist = static_cast<int>(50 * scaleFactor);
    int scaledMinRadius = static_cast<int>(13 * scaleFactor);
    int scaledMaxRadius = static_cast<int>(200 * scaleFactor);
    
    std::vector<cv::Vec3f> outerCircles = detectCircles(enhanced, 1.0, scaledMinDist, 45, 18, scaledMinRadius, scaledMaxRadius);
    // Initial circle detection completed
    
    // Update debug.txt with circle detection info
    if (debugMode) {
        std::ofstream debugFile(debugFolder + "/debug.txt", std::ios::app);
        debugFile << "Initial Circles Found: " << outerCircles.size() << std::endl;
        debugFile.close();
    }
    
    // Validate circles (multiple circles or out of bounds)
    if (!validateCircles(outerCircles, dropletNumber, outputFolder, debugFolder, image, enhanced, padding, scaleFactor, debugMode)) {
        return; // Stop processing if validation fails
    }
    
    // Apply new assistance logic based on threshold type
    cv::Vec3f circleB = outerCircles[0]; // Default to CIRCLE_A
    bool assistanceApplied = false;
    cv::Mat thresholded_e; // Declare at function scope for debug output
    
    if (thresholdType == TYPE_1_FILLED) {
        // Type 1 detected: No assistance needed
        if (debugMode) {
            std::ofstream debugFile(debugFolder + "/debug.txt", std::ios::app);
            debugFile << "Assistance: NONE (TYPE_1_FILLED)" << std::endl;
            debugFile.close();
        }
    } else if (thresholdType == TYPE_2_RING) {
        // Type 2 detected: Applying ring analysis assistance
        // Apply more aggressive erosion to thresholded image for smaller CIRCLE_B
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::erode(thresholded, thresholded_e, kernel);
        
        cv::Point originalCenter(cvRound(outerCircles[0][0]), cvRound(outerCircles[0][1]));
        CrossingAnalysis analysis = analyzeRingStructure(thresholded_e, originalCenter);
        
        // Find optimal center and radius
        circleB = findOptimalRingCenterAndRadius(analysis, originalCenter, analysis.isRing2B);
        assistanceApplied = true;
        
        if (debugMode) {
            std::ofstream debugFile(debugFolder + "/debug.txt", std::ios::app);
            debugFile << "Assistance: TYPE_2_RING" << std::endl;
            debugFile << "Erosion: Applied (5x5 elliptical kernel)" << std::endl;
            debugFile << "Ring Type: " << (analysis.isRing2B ? "RING_2_B" : "RING_2_A") << std::endl;
            debugFile << "Crossings: U" << analysis.upCrossings << " D" << analysis.downCrossings 
                      << " L" << analysis.leftCrossings << " R" << analysis.rightCrossings << std::endl;
            debugFile << "Original Center: (" << originalCenter.x << "," << originalCenter.y << ")" << std::endl;
            debugFile << "New Center: (" << circleB[0] << "," << circleB[1] << ")" << std::endl;
            debugFile << "New Radius: " << circleB[2] << std::endl;
            debugFile.close();
        }
    } else if (thresholdType == TYPE_3_WEIRD) {
        // Type 3 detected: Applying ring analysis assistance (same as RING_2_A)
        // Apply more aggressive erosion to thresholded image for smaller CIRCLE_B
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::erode(thresholded, thresholded_e, kernel);
        
        cv::Point originalCenter(cvRound(outerCircles[0][0]), cvRound(outerCircles[0][1]));
        CrossingAnalysis analysis = analyzeRingStructure(thresholded_e, originalCenter);
        
        // Find optimal center and radius (treat as RING_2_A)
        circleB = findOptimalRingCenterAndRadius(analysis, originalCenter, false);
        assistanceApplied = true;
        
        if (debugMode) {
            std::ofstream debugFile(debugFolder + "/debug.txt", std::ios::app);
            debugFile << "Assistance: TYPE_3_RING" << std::endl;
            debugFile << "Erosion: Applied (5x5 elliptical kernel)" << std::endl;
            debugFile << "Crossings: U" << analysis.upCrossings << " D" << analysis.downCrossings 
                      << " L" << analysis.leftCrossings << " R" << analysis.rightCrossings << std::endl;
            debugFile << "Original Center: (" << originalCenter.x << "," << originalCenter.y << ")" << std::endl;
            debugFile << "New Center: (" << circleB[0] << "," << circleB[1] << ")" << std::endl;
            debugFile << "New Radius: " << circleB[2] << std::endl;
            debugFile.close();
        }
    }
    
    
    // Save original circles debug
        if (debugMode) {
        cv::Mat originalCirclesImg = enhanced.clone();
        if (originalCirclesImg.channels() == 1) {
            cv::cvtColor(originalCirclesImg, originalCirclesImg, cv::COLOR_GRAY2BGR);
        }
        for (const auto& circle : outerCircles) {
            cv::Point center(cvRound(circle[0]), cvRound(circle[1]));
            int radius = cvRound(circle[2]);
            cv::circle(originalCirclesImg, center, radius, cv::Scalar(0, 255, 0), 2);
        }
        std::string originalFilename = debugFolder + "/" + std::to_string(dropletNumber) + "_ORIGINALCIRCLES.jpg";
        cv::imwrite(originalFilename, originalCirclesImg);
    }
    
    
    // Save THRESHOLD_E debug image with CIRCLE_B and crossings drawn on eroded thresholded image
    if (debugMode && assistanceApplied) {
        cv::Mat thresholdEImg = thresholded_e.clone();
        if (thresholdEImg.channels() == 1) {
            cv::cvtColor(thresholdEImg, thresholdEImg, cv::COLOR_GRAY2BGR);
        }
        
        // Draw CIRCLE_B on the eroded thresholded image
        cv::Point circleBCenter(cvRound(circleB[0]), cvRound(circleB[1]));
        int circleBRadius = static_cast<int>(circleB[2]);
        cv::circle(thresholdEImg, circleBCenter, circleBRadius, cv::Scalar(0, 255, 0), 2); // Green
        cv::circle(thresholdEImg, circleBCenter, 3, cv::Scalar(0, 0, 255), -1); // Red center
        
        // Draw actual crossing points detected on the eroded thresholded image
        cv::Point originalCenter(cvRound(outerCircles[0][0]), cvRound(outerCircles[0][1]));
        CrossingAnalysis analysis = analyzeRingStructure(thresholded_e, originalCenter);
        
        // Draw actual crossing points with their distances
        // Up crossings
        for (int distance : analysis.upDistances) {
            cv::Point crossingPoint(originalCenter.x, originalCenter.y - distance);
            cv::circle(thresholdEImg, crossingPoint, 2, cv::Scalar(255, 255, 0), -1); // Yellow
        }
        
        // Down crossings
        for (int distance : analysis.downDistances) {
            cv::Point crossingPoint(originalCenter.x, originalCenter.y + distance);
            cv::circle(thresholdEImg, crossingPoint, 2, cv::Scalar(255, 255, 0), -1); // Yellow
        }
        
        // Left crossings
        for (int distance : analysis.leftDistances) {
            cv::Point crossingPoint(originalCenter.x - distance, originalCenter.y);
            cv::circle(thresholdEImg, crossingPoint, 2, cv::Scalar(255, 255, 0), -1); // Yellow
        }
        
        // Right crossings
        for (int distance : analysis.rightDistances) {
            cv::Point crossingPoint(originalCenter.x + distance, originalCenter.y);
            cv::circle(thresholdEImg, crossingPoint, 2, cv::Scalar(255, 255, 0), -1); // Yellow
        }
        
        std::string thresholdEFilename = debugFolder + "/" + std::to_string(dropletNumber) + "_THRESHOLD_E.jpg";
        cv::imwrite(thresholdEFilename, thresholdEImg);
    }
    
    // Use CIRCLE_B for final calculations
    cv::Vec3f finalCircle = circleB;
    
    // Calculate droplet radius from CIRCLE_B
    double dropletRadius = finalCircle[2] / scaleFactor;
    double dropletDiameter = dropletRadius * 2.0;
    
    // Calculate mean gray value using CIRCLE_B on the blurred (not contrast-enhanced) image
    cv::Mat mask = cv::Mat::zeros(blurred.size(), CV_8UC1);
    cv::circle(mask, cv::Point(cvRound(finalCircle[0]), cvRound(finalCircle[1])), static_cast<int>(finalCircle[2]), cv::Scalar(255), -1);
    cv::Scalar meanScalar = cv::mean(blurred, mask);
    double meanGrayValue = meanScalar[0];
    
    // Create QC image using CIRCLE_B on the blurred, upscaled, padded image
    cv::Mat qcImage = blurred.clone();
    if (qcImage.channels() == 1) {
        cv::cvtColor(qcImage, qcImage, cv::COLOR_GRAY2BGR);
    }
    // Use CIRCLE_B coordinates directly (no scaling needed - already in padded coordinate system)
    cv::Point qcCenter(cvRound(finalCircle[0]), cvRound(finalCircle[1]));
    int qcRadius = static_cast<int>(finalCircle[2]);
    cv::circle(qcImage, qcCenter, qcRadius, cv::Scalar(0, 255, 0), 2);
    cv::circle(qcImage, qcCenter, 3, cv::Scalar(0, 0, 255), -1);
    
    // Debug: Log coordinate information
    if (debugMode) {
        std::ofstream debugFile(debugFolder + "/debug.txt", std::ios::app);
        debugFile << "QC Image Coordinates:" << std::endl;
        debugFile << "Circle B: (" << finalCircle[0] << "," << finalCircle[1] << "), Radius: " << finalCircle[2] << std::endl;
        debugFile << "QC Center: (" << qcCenter.x << "," << qcCenter.y << "), Radius: " << qcRadius << std::endl;
        debugFile << "QC Image Size: " << qcImage.cols << "x" << qcImage.rows << std::endl;
        debugFile.close();
    }
    
    // Save QC image with new filename format (using 'p' instead of '.')
    std::string radiusStr = formatNumberForFilename(dropletRadius);
    std::string meanGrayStr = formatNumberForFilename(meanGrayValue);
    std::string qcFilename = outputFolder + "/" + std::to_string(dropletNumber) + "_VALID_" + radiusStr + "_" + meanGrayStr + ".jpg";
        cv::imwrite(qcFilename, qcImage);
            // QC image saved
    
    // Update debug.txt with final results
    if (debugMode) {
        std::ofstream debugFile(debugFolder + "/debug.txt", std::ios::app);
        debugFile << "Final Circle B - Center: (" << finalCircle[0] << "," << finalCircle[1] << ")" << std::endl;
        debugFile << "Final Circle B - Radius: " << finalCircle[2] << std::endl;
        debugFile << "Final Droplet Radius: " << dropletRadius << std::endl;
        debugFile << "Final Mean Gray: " << meanGrayValue << std::endl;
        debugFile.close();
    }
    
    // Image processing completed
}

int main(int argc, char* argv[]) {
    std::cout << "=== DropletAnalyzer Starting ===" << std::endl;
    
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " <input_filepath> <output_folder> [--debug]" << std::endl;
        std::cerr << "Example: " << argv[0] << " \"C:/images/0_20241203-104610-955_91_00168.jpg\" \"C:/output\"" << std::endl;
        std::cerr << "         " << argv[0] << " \"C:/images/0_20241203-104610-955_91_00168.jpg\" \"C:/output\" --debug" << std::endl;
        return 1;
    }
    
    std::string inputFilepath = argv[1];
    std::string outputFolder = argv[2];
    bool debugMode = false;
    
    if (argc == 4) {
        std::string debugArg = argv[3];
        if (debugArg == "--debug" || debugArg == "-d") {
            debugMode = true;
            std::cout << "Debug mode enabled - intermediate outputs will be saved" << std::endl;
        }
    }
    
    if (!std::filesystem::exists(inputFilepath)) {
        std::cerr << "ERROR: Input file does not exist: " << inputFilepath << std::endl;
        return 1;
    }
    
    if (!std::filesystem::exists(outputFolder)) {
            std::filesystem::create_directories(outputFolder);
        std::cout << "Created output directory: " << outputFolder << std::endl;
    }
    
    processImage(inputFilepath, outputFolder, debugMode);
    
    std::cout << "=== DropletAnalyzer Complete ===" << std::endl;
    return 0;
}