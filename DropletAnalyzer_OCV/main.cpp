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
cv::Mat enhanceContrast(const cv::Mat& input, double clipLimit = 2, int tileGridSize = 8) {
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

    if (((counts.horizontal + counts.vertical) == counts.diagonal) && (counts.horizontal + counts.vertical + counts.diagonal) <=8) {
        return TYPE_1_FILLED;
    }
    if (counts.horizontal <= 4 && counts.vertical <= 4 && counts.diagonal <= 8) {
        return TYPE_2_RING;
    }
    return TYPE_1_FILLED;
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
    int margin = 10; // Reduced margin to 5 pixels
    
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
std::vector<int> scanForCrossings(const cv::Mat& thresholded, cv::Point center, int direction, int maxDistance, int minRadius) {
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
            // Only add crossing if it's beyond the minimum radius
            if (i >= minRadius) {
                crossings.push_back(i);
            }
            prevPixel = currentPixel;
        }
    }
    
    return crossings;
}

// Function to analyze ring structure for TYPE_2_RING
CrossingAnalysis analyzeRingStructure(const cv::Mat& thresholded, cv::Point center, int minRadius) {
    CrossingAnalysis analysis;
    
    // Scan in all four directions
    int maxDistance = std::min(thresholded.rows, thresholded.cols) / 2;
    
    analysis.upDistances = scanForCrossings(thresholded, center, 0, maxDistance, minRadius);
    analysis.downDistances = scanForCrossings(thresholded, center, 1, maxDistance, minRadius);
    analysis.leftDistances = scanForCrossings(thresholded, center, 2, maxDistance, minRadius);
    analysis.rightDistances = scanForCrossings(thresholded, center, 3, maxDistance, minRadius);
    
    analysis.upCrossings = static_cast<int>(analysis.upDistances.size());
    analysis.downCrossings = static_cast<int>(analysis.downDistances.size());
    analysis.leftCrossings = static_cast<int>(analysis.leftDistances.size());
    analysis.rightCrossings = static_cast<int>(analysis.rightDistances.size());
    
    // Determine if it's RING_2_B (has 3 crossings in any direction)
    analysis.isRing2B = (analysis.upCrossings == 3 || analysis.downCrossings == 3 || 
                        analysis.leftCrossings == 3 || analysis.rightCrossings == 3);
    
    return analysis;
}

// Function to balance crossing points for TYPE_2_RING (remove extra crossings from one direction)
CrossingAnalysis balanceCrossingPoints(const CrossingAnalysis& analysis) {
    CrossingAnalysis balanced = analysis;
    
    // Check vertical directions (up vs down)
    int verticalDiff = analysis.upCrossings - analysis.downCrossings;
    if (verticalDiff == 1) {
        // Up has one more crossing, remove the first one
        if (!balanced.upDistances.empty()) {
            balanced.upDistances.erase(balanced.upDistances.begin());
            balanced.upCrossings--;
        }
    } else if (verticalDiff == -1) {
        // Down has one more crossing, remove the first one
        if (!balanced.downDistances.empty()) {
            balanced.downDistances.erase(balanced.downDistances.begin());
            balanced.downCrossings--;
        }
    }
    
    // Check horizontal directions (left vs right)
    int horizontalDiff = analysis.leftCrossings - analysis.rightCrossings;
    if (horizontalDiff == 1) {
        // Left has one more crossing, remove the first one
        if (!balanced.leftDistances.empty()) {
            balanced.leftDistances.erase(balanced.leftDistances.begin());
            balanced.leftCrossings--;
        }
    } else if (horizontalDiff == -1) {
        // Right has one more crossing, remove the first one
        if (!balanced.rightDistances.empty()) {
            balanced.rightDistances.erase(balanced.rightDistances.begin());
            balanced.rightCrossings--;
        }
    }
    
    return balanced;
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
    
    // Validate that we have at least one vertical and one horizontal crossing
    bool hasVertical = !upDist.empty() || !downDist.empty();
    bool hasHorizontal = !leftDist.empty() || !rightDist.empty();
    
    if (!hasVertical || !hasHorizontal) {
        // Return invalid circle (negative radius indicates failure)
        return cv::Vec3f(static_cast<float>(originalCenter.x), static_cast<float>(originalCenter.y), -1.0f);
    }
    
    // Calculate average distances for validation
    double avgVertical = 0.0, avgHorizontal = 0.0;
    int verticalCount = 0, horizontalCount = 0;
    
    if (!upDist.empty()) { avgVertical += upDist[0]; verticalCount++; }
    if (!downDist.empty()) { avgVertical += downDist[0]; verticalCount++; }
    if (!leftDist.empty()) { avgHorizontal += leftDist[0]; horizontalCount++; }
    if (!rightDist.empty()) { avgHorizontal += rightDist[0]; horizontalCount++; }
    
    if (verticalCount > 0) avgVertical /= verticalCount;
    if (horizontalCount > 0) avgHorizontal /= horizontalCount;
    
    // Validate that vertical and horizontal crossings are within 30% of each other
    if (avgVertical > 0 && avgHorizontal > 0) {
        double ratio = std::max(avgVertical, avgHorizontal) / std::min(avgVertical, avgHorizontal);
        if (ratio > 1.3) { // More than 30% difference
            // Return invalid circle (negative radius indicates failure)
            return cv::Vec3f(static_cast<float>(originalCenter.x), static_cast<float>(originalCenter.y), -1.0f);
        }
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
    cv::Mat blurred = applyGaussianBlur(padded, 35);
    
    // Enhance contrast and threshold
    cv::Mat enhanced = enhanceContrast(blurred);
    cv::Mat thresholded;
    cv::threshold(enhanced, thresholded, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    
    // Save enhanced image in debug mode
    if (debugMode) {
        std::string enhancedFilename = debugFolder + "/" + std::to_string(dropletNumber) + "_ENHANCECON.jpg";
        cv::imwrite(enhancedFilename, enhanced);
    }
    
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
    int scaledMinDist = static_cast<int>(60 * scaleFactor); // Increased from 50 to 60
    int scaledMinRadius = static_cast<int>(10 * scaleFactor);
    int scaledMaxRadius = static_cast<int>(200 * scaleFactor);
    
    std::vector<cv::Vec3f> outerCircles = detectCircles(enhanced, 1.0, scaledMinDist, 45, 22, scaledMinRadius, scaledMaxRadius); // Increased param2 from 18 to 22
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
    cv::Vec3f circleC = outerCircles[0]; // Default to CIRCLE_A for scope
    bool assistanceApplied = false;
    cv::Mat thresholded_e; // Declare at function scope for debug output
    std::vector<cv::Point> rightScanPoints, leftScanPoints; // Store scan points for visualization
    
    // Structure to hold scan results for each direction
    struct ScanResult {
        std::vector<cv::Point> outwardPoints, inwardPoints;
        std::vector<double> outwardGrayValues, inwardGrayValues;
        std::vector<double> outwardSlopes, inwardSlopes;
        double outwardMin, inwardMin;
        double outwardMinPos, inwardMinPos;
        double outwardDrop, inwardDrop;
        bool votesUnderestimate, votesOverestimate;
    };
    
    std::vector<ScanResult> scanResults(8); // 8 directions: Right, BR, Bottom, BL, Left, TL, Top, TR
    
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
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(8, 8));
        cv::erode(thresholded, thresholded_e, kernel);
        
        // Apply morphological opening with 6x6 kernel
        cv::Mat openingKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(6, 6));
        cv::morphologyEx(thresholded_e, thresholded_e, cv::MORPH_OPEN, openingKernel);
        
        // Reclassify circle type after morphological opening (small white islands may have been removed)
        ThresholdType reclassifiedType = classifyThresholdType(thresholded_e);
        
        // Declare variables for debug output
        cv::Point originalCenter(cvRound(outerCircles[0][0]), cvRound(outerCircles[0][1]));
        CrossingAnalysis analysis;
        CrossingAnalysis balancedAnalysis;
        
        // If reclassified as TYPE_1_FILLED, use original circle (no assistance needed)
        if (reclassifiedType == TYPE_1_FILLED) {
            circleB = outerCircles[0];
            assistanceApplied = false;
        } else {
            // Still TYPE_2_RING or TYPE_3_WEIRD, proceed with ring analysis
            analysis = analyzeRingStructure(thresholded_e, originalCenter, scaledMinRadius);
            
            // For TYPE_2_RING, balance crossing points (remove extra crossings from one direction)
            balancedAnalysis = balanceCrossingPoints(analysis);
            
            // Find optimal center and radius using balanced analysis
            circleB = findOptimalRingCenterAndRadius(balancedAnalysis, originalCenter, balancedAnalysis.isRing2B);
            
            // Check if assistance failed (negative radius indicates failure)
            if (circleB[2] < 0) {
                // Assistance failed, fall back to original circle
                circleB = outerCircles[0];
                assistanceApplied = false;
            } else {
                assistanceApplied = true;
            }
        }
        
        // Radius validation for TYPE_2_RING using 8-directional scanning technique
        circleC = circleB; // Update CIRCLE_C
        if (assistanceApplied) {
            // Perform 8-directional radius validation scanning
            cv::Point centerB(cvRound(circleB[0]), cvRound(circleB[1]));
            int radiusB = static_cast<int>(circleB[2]);
            
            // Calculate scan distance as 1/2 of the original circle radius (CIRCLE_A)
            int originalRadius = static_cast<int>(outerCircles[0][2]);
            int scanDistance = originalRadius / 2;
            
            // Define 8 direction vectors (outward directions from circle perimeter)
            std::vector<cv::Point> directions = {
                cv::Point(1, 0),    // Right
                cv::Point(1, 1),    // Bottom-Right (normalized)
                cv::Point(0, 1),    // Bottom
                cv::Point(-1, 1),   // Bottom-Left (normalized)
                cv::Point(-1, 0),   // Left
                cv::Point(-1, -1),  // Top-Left (normalized)
                cv::Point(0, -1),   // Top
                cv::Point(1, -1)    // Top-Right (normalized)
            };
            
            // Normalize diagonal directions
            for (int i = 0; i < 8; i++) {
                if (i % 2 == 1) { // Diagonal directions (1, 3, 5, 7)
                    double length = std::sqrt(directions[i].x * directions[i].x + directions[i].y * directions[i].y);
                    directions[i].x = static_cast<int>(std::round(directions[i].x / length));
                    directions[i].y = static_cast<int>(std::round(directions[i].y / length));
                }
            }
            
            // Scan in each of the 8 directions
            for (int dir = 0; dir < 8; dir++) {
                // Calculate starting point on circle perimeter
                cv::Point startPoint(centerB.x + radiusB * directions[dir].x, 
                                   centerB.y + radiusB * directions[dir].y);
                
                // Outward scan (away from center)
                for (int i = 0; i <= scanDistance; i++) {
                    cv::Point scanPoint(startPoint.x + i * directions[dir].x, 
                                      startPoint.y + i * directions[dir].y);
                    if (scanPoint.x >= 0 && scanPoint.x < enhanced.cols && 
                        scanPoint.y >= 0 && scanPoint.y < enhanced.rows) {
                        scanResults[dir].outwardPoints.push_back(scanPoint);
                        double grayValue = static_cast<double>(enhanced.at<uchar>(scanPoint.y, scanPoint.x));
                        scanResults[dir].outwardGrayValues.push_back(grayValue);
                        
                        // Calculate slope (except for last point)
                        if (i < scanDistance) {
                            cv::Point nextPoint(startPoint.x + (i + 1) * directions[dir].x,
                                              startPoint.y + (i + 1) * directions[dir].y);
                            if (nextPoint.x >= 0 && nextPoint.x < enhanced.cols && 
                                nextPoint.y >= 0 && nextPoint.y < enhanced.rows) {
                                double nextGrayValue = static_cast<double>(enhanced.at<uchar>(nextPoint.y, nextPoint.x));
                                double slope = grayValue - nextGrayValue;
                                scanResults[dir].outwardSlopes.push_back(slope);
                            }
                        }
                    }
                }
                
                // Inward scan (toward center)
                for (int i = 0; i <= scanDistance; i++) {
                    cv::Point scanPoint(startPoint.x - i * directions[dir].x, 
                                      startPoint.y - i * directions[dir].y);
                    if (scanPoint.x >= 0 && scanPoint.x < enhanced.cols && 
                        scanPoint.y >= 0 && scanPoint.y < enhanced.rows) {
                        scanResults[dir].inwardPoints.push_back(scanPoint);
                        double grayValue = static_cast<double>(enhanced.at<uchar>(scanPoint.y, scanPoint.x));
                        scanResults[dir].inwardGrayValues.push_back(grayValue);
                        
                        // Calculate slope (except for last point)
                        if (i < scanDistance) {
                            cv::Point nextPoint(startPoint.x - (i + 1) * directions[dir].x,
                                              startPoint.y - (i + 1) * directions[dir].y);
                            if (nextPoint.x >= 0 && nextPoint.x < enhanced.cols && 
                                nextPoint.y >= 0 && nextPoint.y < enhanced.rows) {
                                double nextGrayValue = static_cast<double>(enhanced.at<uchar>(nextPoint.y, nextPoint.x));
                                double slope = grayValue - nextGrayValue;
                                scanResults[dir].inwardSlopes.push_back(slope);
                            }
                        }
                    }
                }
            }
            
            // First, determine if CIRCLE_B is an overestimate or underestimate
            // by comparing its radius to the second and third diagonal crossing distances
            double secondCrossingAvg = 0.0;
            double thirdCrossingAvg = 0.0;
            int secondCount = 0, thirdCount = 0;
            
            // Calculate average second crossing distance
            if (analysis.upCrossings > 1 && analysis.upDistances.size() > 1) {
                secondCrossingAvg += analysis.upDistances[1];
                secondCount++;
            }
            if (analysis.downCrossings > 1 && analysis.downDistances.size() > 1) {
                secondCrossingAvg += analysis.downDistances[1];
                secondCount++;
            }
            if (analysis.leftCrossings > 1 && analysis.leftDistances.size() > 1) {
                secondCrossingAvg += analysis.leftDistances[1];
                secondCount++;
            }
            if (analysis.rightCrossings > 1 && analysis.rightDistances.size() > 1) {
                secondCrossingAvg += analysis.rightDistances[1];
                secondCount++;
            }
            if (secondCount > 0) secondCrossingAvg /= secondCount;
            
            // Calculate average third crossing distance
            if (analysis.upCrossings > 2 && analysis.upDistances.size() > 2) {
                thirdCrossingAvg += analysis.upDistances[2];
                thirdCount++;
            }
            if (analysis.downCrossings > 2 && analysis.downDistances.size() > 2) {
                thirdCrossingAvg += analysis.downDistances[2];
                thirdCount++;
            }
            if (analysis.leftCrossings > 2 && analysis.leftDistances.size() > 2) {
                thirdCrossingAvg += analysis.leftDistances[2];
                thirdCount++;
            }
            if (analysis.rightCrossings > 2 && analysis.rightDistances.size() > 2) {
                thirdCrossingAvg += analysis.rightDistances[2];
                thirdCount++;
            }
            if (thirdCount > 0) thirdCrossingAvg /= thirdCount;
            
            // Determine if CIRCLE_B is overestimated or underestimated
            bool isOverestimated = false;
            bool isUnderestimated = false;
            double circleBRadius = circleB[2];
            
            if (thirdCount > 0) {
                // We have both second and third crossings - compare to both
                double distToSecond = std::abs(circleBRadius - secondCrossingAvg);
                double distToThird = std::abs(circleBRadius - thirdCrossingAvg);
                
                if (distToThird < distToSecond) {
                    isOverestimated = true; // Closer to third crossings (overestimate)
                } else {
                    isUnderestimated = true; // Closer to second crossings (underestimate)
                }
            } else {
                // Only second crossings available - assume underestimate if radius is smaller
                if (circleBRadius < secondCrossingAvg) {
                    isUnderestimated = true;
                } else {
                    isOverestimated = true;
                }
            }
            
            // Analyze patterns for each direction and collect votes
            int overestimateVotes = 0;
            int underestimateVotes = 0;
            std::vector<double> overestimateAdjustments;
            std::vector<double> underestimateAdjustments;
            
            for (int dir = 0; dir < 8; dir++) {
                // Calculate minimums for this direction
                if (!scanResults[dir].outwardGrayValues.empty()) {
                    scanResults[dir].outwardMin = *std::min_element(scanResults[dir].outwardGrayValues.begin(), 
                                                                  scanResults[dir].outwardGrayValues.end());
                }
                if (!scanResults[dir].inwardGrayValues.empty()) {
                    scanResults[dir].inwardMin = *std::min_element(scanResults[dir].inwardGrayValues.begin(), 
                                                                 scanResults[dir].inwardGrayValues.end());
                }
                
                // Find minimum positions
                for (size_t i = 0; i < scanResults[dir].outwardGrayValues.size(); i++) {
                    if (scanResults[dir].outwardGrayValues[i] == scanResults[dir].outwardMin) {
                        scanResults[dir].outwardMinPos = static_cast<double>(i);
                        break;
                    }
                }
                
                for (size_t i = 0; i < scanResults[dir].inwardGrayValues.size(); i++) {
                    if (scanResults[dir].inwardGrayValues[i] == scanResults[dir].inwardMin) {
                        scanResults[dir].inwardMinPos = static_cast<double>(i);
                        break;
                    }
                }
                
                // Calculate gray value drops
                if (!scanResults[dir].outwardGrayValues.empty() && !scanResults[dir].inwardGrayValues.empty()) {
                    double outwardStartGrayValue = scanResults[dir].outwardGrayValues[0];
                    scanResults[dir].outwardDrop = outwardStartGrayValue - scanResults[dir].outwardMin;
                    
                    double inwardStartGrayValue = scanResults[dir].inwardGrayValues[0];
                    scanResults[dir].inwardDrop = inwardStartGrayValue - scanResults[dir].inwardMin;
                    
                    // Smart scanning: only look in the appropriate direction based on over/underestimate
                    if (isUnderestimated) {
                        // For underestimates, only look at outward scans to find the true boundary
                        if (scanResults[dir].outwardDrop > 30.0 && 
                            scanResults[dir].outwardMinPos > 5 && scanResults[dir].outwardMinPos < scanDistance) {
                            scanResults[dir].votesUnderestimate = true;
                            scanResults[dir].votesOverestimate = false;
                            underestimateVotes++;
                            underestimateAdjustments.push_back(scanResults[dir].outwardMinPos);
                        } else {
                            scanResults[dir].votesOverestimate = false;
                            scanResults[dir].votesUnderestimate = false;
                        }
                    } else if (isOverestimated) {
                        // For overestimates, only look at inward scans to find the true boundary
                        if (scanResults[dir].inwardDrop > 30.0 && 
                            scanResults[dir].inwardMinPos > 5 && scanResults[dir].inwardMinPos < scanDistance) {
                            scanResults[dir].votesOverestimate = true;
                            scanResults[dir].votesUnderestimate = false;
                            overestimateVotes++;
                            overestimateAdjustments.push_back(scanResults[dir].inwardMinPos);
                        } else {
                            scanResults[dir].votesOverestimate = false;
                            scanResults[dir].votesUnderestimate = false;
                        }
                    } else {
                        // No clear determination - use original logic
                        if (scanResults[dir].inwardDrop > scanResults[dir].outwardDrop && 
                            scanResults[dir].inwardMinPos > 5 && scanResults[dir].inwardMinPos < scanDistance && 
                            scanResults[dir].inwardDrop > 30.0) {
                            scanResults[dir].votesOverestimate = true;
                            scanResults[dir].votesUnderestimate = false;
                            overestimateVotes++;
                            overestimateAdjustments.push_back(scanResults[dir].inwardMinPos);
                        } else if (scanResults[dir].outwardDrop > scanResults[dir].inwardDrop && 
                                  scanResults[dir].outwardMinPos > 5 && scanResults[dir].outwardMinPos < scanDistance && 
                                  scanResults[dir].outwardDrop > 30.0) {
                            scanResults[dir].votesUnderestimate = true;
                            scanResults[dir].votesOverestimate = false;
                            underestimateVotes++;
                            underestimateAdjustments.push_back(scanResults[dir].outwardMinPos);
                        } else {
                            scanResults[dir].votesOverestimate = false;
                            scanResults[dir].votesUnderestimate = false;
                        }
                    }
                }
            }
            
            // Determine final adjustment based on majority vote
            double adjustedRadius = circleB[2];
            
            if (overestimateVotes > underestimateVotes && overestimateVotes > 0) {
                // Majority votes for overestimate
                isOverestimated = true;
                double avgAdjustment = 0.0;
                for (double adj : overestimateAdjustments) avgAdjustment += adj;
                avgAdjustment /= overestimateAdjustments.size();
                adjustedRadius = circleB[2] - avgAdjustment;
                adjustedRadius = std::max(adjustedRadius, circleB[2] * 0.7); // Don't reduce by more than 30%
            } else if (underestimateVotes > overestimateVotes && underestimateVotes > 0) {
                // Majority votes for underestimate
                isUnderestimated = true;
                double avgAdjustment = 0.0;
                for (double adj : underestimateAdjustments) avgAdjustment += adj;
                avgAdjustment /= underestimateAdjustments.size();
                adjustedRadius = circleB[2] + avgAdjustment;
                adjustedRadius = std::min(adjustedRadius, circleB[2] * 1.3); // Don't increase by more than 30%
            }
            
            // Update CIRCLE_C with adjusted radius
            circleC = cv::Vec3f(circleB[0], circleB[1], static_cast<float>(adjustedRadius));
            
            if (debugMode) {
                std::ofstream debugFile(debugFolder + "/debug.txt", std::ios::app);
                debugFile << "8-Directional Radius Validation Scan Results:" << std::endl;
                debugFile << "Crossing Distance Analysis (Diagonal Crossings):" << std::endl;
                debugFile << "  Second Crossing Average: " << secondCrossingAvg << " (from " << secondCount << " directions)" << std::endl;
                debugFile << "  Third Crossing Average: " << thirdCrossingAvg << " (from " << thirdCount << " directions)" << std::endl;
                debugFile << "  Circle B Radius: " << circleBRadius << std::endl;
                debugFile << "  Distance to Second: " << std::abs(circleBRadius - secondCrossingAvg) << std::endl;
                if (thirdCount > 0) {
                    debugFile << "  Distance to Third: " << std::abs(circleBRadius - thirdCrossingAvg) << std::endl;
                }
                debugFile << "Original Circle A Radius: " << originalRadius << std::endl;
                debugFile << "Scan Distance (1/2r): " << scanDistance << std::endl;
                debugFile << "Voting Results: Overestimate=" << overestimateVotes << ", Underestimate=" << underestimateVotes << std::endl;
                debugFile << "Is Overestimated: " << (isOverestimated ? "YES" : "NO") << std::endl;
                debugFile << "Is Underestimated: " << (isUnderestimated ? "YES" : "NO") << std::endl;
                debugFile << "Original Radius B: " << circleB[2] << std::endl;
                debugFile << "Adjusted Radius C: " << adjustedRadius << std::endl;
                
                // Output individual direction results
                std::vector<std::string> directionNames = {"Right", "Bottom-Right", "Bottom", "Bottom-Left", "Left", "Top-Left", "Top", "Top-Right"};
                for (int dir = 0; dir < 8; dir++) {
                    debugFile << "Direction " << dir << " (" << directionNames[dir] << "): ";
                    debugFile << "Outward Drop=" << scanResults[dir].outwardDrop << ", Inward Drop=" << scanResults[dir].inwardDrop << ", ";
                    if (scanResults[dir].votesOverestimate) {
                        debugFile << "Vote=OVERESTIMATE, Adjustment=" << scanResults[dir].inwardMinPos << std::endl;
                    } else if (scanResults[dir].votesUnderestimate) {
                        debugFile << "Vote=UNDERESTIMATE, Adjustment=" << scanResults[dir].outwardMinPos << std::endl;
                    } else {
                        debugFile << "Vote=NONE" << std::endl;
                    }
                }
                
                debugFile.close();
            }
        }
        
        if (debugMode) {
            std::ofstream debugFile(debugFolder + "/debug.txt", std::ios::app);
            debugFile << "Assistance: TYPE_2_RING" << std::endl;
            debugFile << "Erosion: Applied (8x8 elliptical kernel)" << std::endl;
            debugFile << "Morphological Opening: Applied (6x6 elliptical kernel)" << std::endl;
            debugFile << "Reclassified Type: " << (reclassifiedType == TYPE_1_FILLED ? "TYPE_1_FILLED" : 
                                                  reclassifiedType == TYPE_2_RING ? "TYPE_2_RING" : "TYPE_3_WEIRD") << std::endl;
            if (reclassifiedType == TYPE_1_FILLED) {
                debugFile << "Strategy: Reclassified as FILLED - Using original CIRCLE_A (no assistance applied)" << std::endl;
            } else {
                debugFile << "Ring Type: " << (analysis.isRing2B ? "RING_2_B" : "RING_2_A") << std::endl;
                debugFile << "Original Crossings: U" << analysis.upCrossings << " D" << analysis.downCrossings 
                          << " L" << analysis.leftCrossings << " R" << analysis.rightCrossings << std::endl;
                debugFile << "Balanced Crossings: U" << balancedAnalysis.upCrossings << " D" << balancedAnalysis.downCrossings 
                          << " L" << balancedAnalysis.leftCrossings << " R" << balancedAnalysis.rightCrossings << std::endl;
            }
            debugFile << "Original Center: (" << originalCenter.x << "," << originalCenter.y << ")" << std::endl;
            if (assistanceApplied) {
                debugFile << "New Center: (" << circleB[0] << "," << circleB[1] << ")" << std::endl;
                debugFile << "New Radius: " << circleB[2] << std::endl;
            } else {
                debugFile << "Assistance FAILED - Using original circle" << std::endl;
                debugFile << "Original Center: (" << circleB[0] << "," << circleB[1] << ")" << std::endl;
                debugFile << "Original Radius: " << circleB[2] << std::endl;
            }
            debugFile.close();
        }
    } else if (thresholdType == TYPE_3_WEIRD) {
        // Type 3 detected: Use original CIRCLE_A as final circle (no assistance)
        circleB = outerCircles[0];
        assistanceApplied = false;
        
        if (debugMode) {
            std::ofstream debugFile(debugFolder + "/debug.txt", std::ios::app);
            debugFile << "Assistance: TYPE_3_WEIRD" << std::endl;
            debugFile << "Strategy: Using original CIRCLE_A (no assistance applied)" << std::endl;
            debugFile << "Original Center: (" << circleB[0] << "," << circleB[1] << ")" << std::endl;
            debugFile << "Original Radius: " << circleB[2] << std::endl;
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
        CrossingAnalysis analysis = analyzeRingStructure(thresholded_e, originalCenter, scaledMinRadius);
        
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
    
    // Use CIRCLE_C for TYPE_2_RING final calculations, otherwise CIRCLE_B
    cv::Vec3f finalCircle = (thresholdType == TYPE_2_RING && assistanceApplied) ? circleC : circleB;
    
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
    
    // Create debug QC image with both CIRCLE_B and CIRCLE_C for TYPE_2_RING
    if (debugMode && thresholdType == TYPE_2_RING && assistanceApplied) {
        cv::Mat debugQcImage = blurred.clone();
        if (debugQcImage.channels() == 1) {
            cv::cvtColor(debugQcImage, debugQcImage, cv::COLOR_GRAY2BGR);
        }
        
        // Draw CIRCLE_B (original assisted circle) in blue
        cv::Point circleBCenter(cvRound(circleB[0]), cvRound(circleB[1]));
        int circleBRadius = static_cast<int>(circleB[2]);
        cv::circle(debugQcImage, circleBCenter, circleBRadius, cv::Scalar(255, 0, 0), 2); // Blue
        cv::circle(debugQcImage, circleBCenter, 3, cv::Scalar(0, 0, 255), -1); // Red center
        
        // Draw CIRCLE_C (validated circle) in green
        cv::Point circleCCenter(cvRound(circleC[0]), cvRound(circleC[1]));
        int circleCRadius = static_cast<int>(circleC[2]);
        cv::circle(debugQcImage, circleCCenter, circleCRadius, cv::Scalar(0, 255, 0), 2); // Green
        cv::circle(debugQcImage, circleCCenter, 3, cv::Scalar(0, 0, 255), -1); // Red center
        
        // Draw all scanned pixels from 8 directions
        for (int dir = 0; dir < 8; dir++) {
            // Draw outward scan points in red
            for (size_t i = 0; i < scanResults[dir].outwardPoints.size(); i++) {
                cv::circle(debugQcImage, scanResults[dir].outwardPoints[i], 1, cv::Scalar(0, 0, 255), -1); // Red dots
            }
            // Draw inward scan points in yellow
            for (size_t i = 0; i < scanResults[dir].inwardPoints.size(); i++) {
                cv::circle(debugQcImage, scanResults[dir].inwardPoints[i], 1, cv::Scalar(0, 255, 255), -1); // Yellow dots
            }
        }
        
        // Save debug QC image
        std::string debugQcFilename = debugFolder + "/" + std::to_string(dropletNumber) + "_DEBUG_QC_CIRCLES_B_C.jpg";
        cv::imwrite(debugQcFilename, debugQcImage);
        
        // Update debug.txt with CIRCLE_C information
        std::ofstream debugFile(debugFolder + "/debug.txt", std::ios::app);
        debugFile << "Debug QC Image: Both CIRCLE_B and CIRCLE_C drawn with 8-directional scan pixels" << std::endl;
        debugFile << "Circle B (Blue): (" << circleB[0] << "," << circleB[1] << "), Radius: " << circleB[2] << std::endl;
        debugFile << "Circle C (Green): (" << circleC[0] << "," << circleC[1] << "), Radius: " << circleC[2] << std::endl;
        debugFile << "Scan Points: Red=Outward, Yellow=Inward (8 directions total)" << std::endl;
        debugFile.close();
    }
    
    // Update debug.txt with final results
    if (debugMode) {
        std::ofstream debugFile(debugFolder + "/debug.txt", std::ios::app);
        if (thresholdType == TYPE_2_RING && assistanceApplied) {
            debugFile << "Final Circle C (Validated) - Center: (" << finalCircle[0] << "," << finalCircle[1] << ")" << std::endl;
            debugFile << "Final Circle C (Validated) - Radius: " << finalCircle[2] << std::endl;
        } else {
            debugFile << "Final Circle B - Center: (" << finalCircle[0] << "," << finalCircle[1] << ")" << std::endl;
            debugFile << "Final Circle B - Radius: " << finalCircle[2] << std::endl;
        }
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