#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <iomanip>
#include <filesystem>
#include <sstream>

// Function to convert to grayscale
cv::Mat convertToGrayscale(const cv::Mat& input) {
    // Make an empty Mat to store the grayscale image.
    cv::Mat gray;
    // Convert the image to grayscale.
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
            static_cast<float>(circle[0] / scaleFactor),  // x-coordinate
            static_cast<float>(circle[1] / scaleFactor),  // y-coordinate
            static_cast<float>(circle[2] / scaleFactor)   // radius
        ));
    }
    return scaledCircles;
}

// Function to resize image for consistent display
cv::Mat resizeForDisplay(const cv::Mat& input, int maxWidth = 800) {
    if (input.cols <= maxWidth) {
        return input.clone();
    }
    
    double scale = static_cast<double>(maxWidth) / input.cols;
    cv::Mat resized;
    cv::resize(input, resized, cv::Size(), scale, scale, cv::INTER_AREA);
    return resized;
}

// Function to apply Gaussian blur for noise reduction
cv::Mat applyGaussianBlur(const cv::Mat& input, int kernelSize = 5) {
    // Make an empty Mat to store the blurred image.
    cv::Mat blurred;
    // Apply Gaussian blur to the image.
    cv::GaussianBlur(input, blurred, cv::Size(kernelSize, kernelSize), 0);
    return blurred;
}

// Function to apply bilateral filter for edge-preserving noise reduction
cv::Mat applyBilateralFilter(const cv::Mat& input, int d = 9, double sigmaColor = 75, double sigmaSpace = 75) {
    // Make an empty Mat to store the filtered image.
    cv::Mat filtered;
    // Apply bilateral filter to the image (preserves edges while reducing noise)
    cv::bilateralFilter(input, filtered, d, sigmaColor, sigmaSpace);
    return filtered;
}

// Function to enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
cv::Mat enhanceContrast(const cv::Mat& input, double clipLimit = 2.0, int tileGridSize = 8) {
    // Create CLAHE object
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clipLimit, cv::Size(tileGridSize, tileGridSize));
    
    // Apply CLAHE to the input image
    cv::Mat enhanced;
    clahe->apply(input, enhanced);
    
    return enhanced;
}

// Function to calculate mean gray value from corner regions of the image
cv::Scalar calculateCornerBasedMean(const cv::Mat& input, int cornerSize = -1) {
    // Default corner size is 10% of the smaller dimension
    if (cornerSize == -1) {
        cornerSize = static_cast<int>(std::min(input.cols, input.rows) * 0.1);
    }
    
    // Ensure corner size is at least 1 and doesn't exceed image dimensions
    cornerSize = std::max(1, std::min(cornerSize, std::min(input.cols, input.rows) / 2));
    
    // Define corner regions (top-left, top-right, bottom-left, bottom-right)
    cv::Rect topLeft(0, 0, cornerSize, cornerSize);
    cv::Rect topRight(input.cols - cornerSize, 0, cornerSize, cornerSize);
    cv::Rect bottomLeft(0, input.rows - cornerSize, cornerSize, cornerSize);
    cv::Rect bottomRight(input.cols - cornerSize, input.rows - cornerSize, cornerSize, cornerSize);
    
    // Calculate mean for each corner
    cv::Scalar meanTopLeft = cv::mean(input(topLeft));
    cv::Scalar meanTopRight = cv::mean(input(topRight));
    cv::Scalar meanBottomLeft = cv::mean(input(bottomLeft));
    cv::Scalar meanBottomRight = cv::mean(input(bottomRight));
    
    // Average all four corners
    cv::Scalar avgMean;
    avgMean[0] = (meanTopLeft[0] + meanTopRight[0] + meanBottomLeft[0] + meanBottomRight[0]) / 4.0;
    
    return avgMean;
}

// Function to visualize corner regions used for padding calculation
void visualizeCornerRegions(cv::Mat& image, int cornerSize = -1) {
    // Default corner size is 10% of the smaller dimension
    if (cornerSize == -1) {
        cornerSize = static_cast<int>(std::min(image.cols, image.rows) * 0.1);
    }
    
    // Ensure corner size is at least 1 and doesn't exceed image dimensions
    cornerSize = std::max(1, std::min(cornerSize, std::min(image.cols, image.rows) / 2));
    
    // Define corner regions
    cv::Rect topLeft(0, 0, cornerSize, cornerSize);
    cv::Rect topRight(image.cols - cornerSize, 0, cornerSize, cornerSize);
    cv::Rect bottomLeft(0, image.rows - cornerSize, cornerSize, cornerSize);
    cv::Rect bottomRight(image.cols - cornerSize, image.rows - cornerSize, cornerSize, cornerSize);
    
    // Draw red rectangles on each corner
    cv::rectangle(image, topLeft, cv::Scalar(0, 0, 255), 2);
    cv::rectangle(image, topRight, cv::Scalar(0, 0, 255), 2);
    cv::rectangle(image, bottomLeft, cv::Scalar(0, 0, 255), 2);
    cv::rectangle(image, bottomRight, cv::Scalar(0, 0, 255), 2);
}

// Function to detect circles using Hough Transform (directly on blurred image)
std::vector<cv::Vec3f> detectCircles(const cv::Mat& input, double dp = 1.0, double minDist = 50, 
    double param1 = 100, double param2 = 30, 
    int minRadius = 20, int maxRadius = 100) {
    
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(input, circles, cv::HOUGH_GRADIENT, dp, minDist, param1, param2, minRadius, maxRadius);
    return circles;
}

// Function to count black pixels in a ring (between outer and inner radius)
int countBlackPixelsInRing(const cv::Mat& thresholded, const cv::Point& center, int outerRadius, int innerRadius) {
    cv::Mat ringMask = cv::Mat::zeros(thresholded.size(), CV_8UC1);
    cv::circle(ringMask, center, outerRadius, cv::Scalar(255), -1);
    cv::circle(ringMask, center, innerRadius, cv::Scalar(0), -1);
    
    // Count black pixels (value = 0) within the ring mask
    int blackCount = 0;
    for (int y = 0; y < thresholded.rows; y++) {
        for (int x = 0; x < thresholded.cols; x++) {
            if (ringMask.at<uchar>(y, x) == 255 && thresholded.at<uchar>(y, x) == 0) {
                blackCount++;
            }
        }
    }
    return blackCount;
}

// Function to test a specific center position with quadrant analysis
double testCenterPosition(const cv::Mat& thresholded, const cv::Point& center, int testRadius, int outerRadius) {
    // Create ring mask for outer radius (10-pixel thick ring)
    int sliceWidth = 10;
    cv::Mat outerRingMask = cv::Mat::zeros(thresholded.size(), CV_8UC1);
    cv::circle(outerRingMask, center, testRadius, cv::Scalar(255), -1);
    cv::circle(outerRingMask, center, testRadius - sliceWidth, cv::Scalar(0), -1);
    
    // Create ring mask for inner radius (testRadius - sliceWidth, also sliceWidth thick)
    cv::Mat innerRingMask = cv::Mat::zeros(thresholded.size(), CV_8UC1);
    cv::circle(innerRingMask, center, testRadius - sliceWidth, cv::Scalar(255), -1);
    cv::circle(innerRingMask, center, testRadius - (2 * sliceWidth), cv::Scalar(0), -1);
    
    // This function is no longer used - quadrant analysis moved to calculateBlackPixelScore
    return 0.0;
}

// Enum to represent droplet type based on thresholded image appearance
enum DropletType {
    RING_TYPE,      // Light droplets: white padding → black ring → white center
    FILLED_TYPE     // Dark droplets: white padding → filled black circle
};

// Structure to hold circle results with scores and black pixel counts
struct CircleResult {
    cv::Vec3f circle;
    double score;
    int blackPixels;
    int translationDistance; // Distance from original center
    int outerRadius; // The winning outer radius (testRadius)
    
    CircleResult(cv::Vec3f c, double s, int bp, int td = 0, int or = 0) : circle(c), score(s), blackPixels(bp), translationDistance(td), outerRadius(or) {}
};

// Function to count black pixels in a specific quadrant of a ring
int countBlackPixelsInQuadrant(const cv::Mat& thresholded, const cv::Point& center, int outerRadius, int innerRadius, int quadrant) {
    int blackCount = 0;
    
    for (int y = 0; y < thresholded.rows; y++) {
        for (int x = 0; x < thresholded.cols; x++) {
            // Check if pixel is within ring
            int dx = x - center.x;
            int dy = y - center.y;
            int distance = static_cast<int>(sqrt(dx*dx + dy*dy));
            
            if (distance <= outerRadius && distance > innerRadius) {
                // Determine quadrant (0=top-right, 1=top-left, 2=bottom-left, 3=bottom-right)
                bool right = (dx >= 0);
                bool bottom = (dy >= 0);
                int pixelQuadrant = (bottom ? 2 : 0) + (right ? 0 : 1);
                
                if (pixelQuadrant == quadrant && thresholded.at<uchar>(y, x) == 0) {
                    blackCount++;
                }
            }
        }
    }
    
    return blackCount;
}

// Function to detect droplet type by sampling rings at increasing radii
DropletType detectDropletType(const cv::Mat& thresholded, const cv::Point& center, int detectedRadius) {
    // Sample from (detectedRadius - 10) to (detectedRadius + 10) in 1-pixel increments
    // Use 2-pixel wide slices
    std::vector<bool> isWhiteSlice;  // true = >50% white, false = >50% black
    std::vector<int> sliceRadii;
    
    int startRadius = std::max(5, detectedRadius - 10);  // Don't go below 5 pixels
    int endRadius = detectedRadius + 10;
    
    for (int testRadius = startRadius; testRadius <= endRadius; testRadius++) {
        // Create a thin ring (2 pixels wide)
        int ringThickness = 2;
        cv::Mat ringMask = cv::Mat::zeros(thresholded.size(), CV_8UC1);
        cv::circle(ringMask, center, testRadius, cv::Scalar(255), -1);
        cv::circle(ringMask, center, std::max(1, testRadius - ringThickness), cv::Scalar(0), -1);
        
        // Count white vs total pixels in ring
        int totalPixels = 0;
        int whitePixels = 0;
        for (int y = 0; y < thresholded.rows; y++) {
            for (int x = 0; x < thresholded.cols; x++) {
                if (ringMask.at<uchar>(y, x) == 255) {
                    totalPixels++;
                    if (thresholded.at<uchar>(y, x) == 255) {
                        whitePixels++;
                    }
                }
            }
        }
        
        if (totalPixels > 0) {
            double whiteRatio = static_cast<double>(whitePixels) / totalPixels;
            isWhiteSlice.push_back(whiteRatio > 0.5);  // true if >50% white
            sliceRadii.push_back(testRadius);
        }
    }
    
    // Print debug info about the pattern
    std::cout << "  Slice pattern (radius " << startRadius << "-" << endRadius << "): [";
    for (size_t i = 0; i < isWhiteSlice.size(); i++) {
        std::cout << (isWhiteSlice[i] ? "W" : "B");
        if (i < isWhiteSlice.size() - 1 && i % 5 == 4) std::cout << " ";  // Space every 5 for readability
    }
    std::cout << "]" << std::endl;
    
    // Look for transition from white to black (ring characteristic)
    // Need to find 2 consecutive black slices after seeing white
    bool hasSeenWhite = false;
    int consecutiveBlackCount = 0;
    
    for (size_t i = 0; i < isWhiteSlice.size(); i++) {
        if (isWhiteSlice[i]) {
            // White slice
            hasSeenWhite = true;
            consecutiveBlackCount = 0;  // Reset black count
        } else {
            // Black slice
            if (hasSeenWhite) {
                consecutiveBlackCount++;
                if (consecutiveBlackCount >= 2) {
                    // Found white → black transition with 2 consecutive blacks
                    std::cout << "  → Detected white-to-black transition at radius " << sliceRadii[i] << " → RING_TYPE" << std::endl;
                    return RING_TYPE;
                }
            }
        }
    }
    
    // No white-to-black transition found → filled type
    std::cout << "  → No white-to-black transition detected → FILLED_TYPE" << std::endl;
    return FILLED_TYPE;
}

// Function to count white pixels in a ring (for dark droplet refinement)
int countWhitePixelsInRing(const cv::Mat& thresholded, const cv::Point& center, int outerRadius, int innerRadius) {
    cv::Mat ringMask = cv::Mat::zeros(thresholded.size(), CV_8UC1);
    cv::circle(ringMask, center, outerRadius, cv::Scalar(255), -1);
    cv::circle(ringMask, center, innerRadius, cv::Scalar(0), -1);
    
    // Count white pixels (value = 255) within the ring mask
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

// Function to count white pixels in a specific quadrant of a ring
int countWhitePixelsInQuadrant(const cv::Mat& thresholded, const cv::Point& center, int outerRadius, int innerRadius, int quadrant) {
    int whiteCount = 0;
    
    for (int y = 0; y < thresholded.rows; y++) {
        for (int x = 0; x < thresholded.cols; x++) {
            // Check if pixel is within ring
            int dx = x - center.x;
            int dy = y - center.y;
            int distance = static_cast<int>(sqrt(dx*dx + dy*dy));
            
            if (distance <= outerRadius && distance > innerRadius) {
                // Determine quadrant (0=top-right, 1=top-left, 2=bottom-left, 3=bottom-right)
                bool right = (dx >= 0);
                bool bottom = (dy >= 0);
                int pixelQuadrant = (bottom ? 2 : 0) + (right ? 0 : 1);
                
                if (pixelQuadrant == quadrant && thresholded.at<uchar>(y, x) == 255) {
                    whiteCount++;
                }
            }
        }
    }
    
    return whiteCount;
}

// Function to calculate score based on white pixels with quadrant uniformity penalty (for dark droplets)
double calculateWhitePixelScore(int totalWhitePixels, int translationDistance, const cv::Mat& thresholded, const cv::Point& center, int outerRadius, int innerRadius) {
    // Base score is just the number of white pixels
    double baseScore = static_cast<double>(totalWhitePixels);
    
    // If no white pixels, return 0
    if (totalWhitePixels == 0) {
        return 0.0;
    }
    
    // Count white pixels in each quadrant of the ring
    std::vector<int> quadrantCounts;
    for (int q = 0; q < 4; q++) {
        quadrantCounts.push_back(countWhitePixelsInQuadrant(thresholded, center, outerRadius, innerRadius, q));
    }
    
    // Calculate quadrant uniformity penalty (compare adjacent quadrants)
    double maxQuadrantDifference = 0;
    
    // Compare adjacent quadrants
    double diff01 = std::abs(quadrantCounts[0] - quadrantCounts[1]);
    maxQuadrantDifference = std::max(maxQuadrantDifference, diff01);
    
    double diff12 = std::abs(quadrantCounts[1] - quadrantCounts[2]);
    maxQuadrantDifference = std::max(maxQuadrantDifference, diff12);
    
    double diff23 = std::abs(quadrantCounts[2] - quadrantCounts[3]);
    maxQuadrantDifference = std::max(maxQuadrantDifference, diff23);
    
    double diff30 = std::abs(quadrantCounts[3] - quadrantCounts[0]);
    maxQuadrantDifference = std::max(maxQuadrantDifference, diff30);
    
    // Translation penalty removed - focusing on quadrant penalty only
    double translationPenalty = 1.0;
    
    // Apply quadrant uniformity penalty: reduce score for uneven quadrants
    double quadrantPenalty = 1.0;
    
    // Check for unevenness
    int maxQuadrantCount = *std::max_element(quadrantCounts.begin(), quadrantCounts.end());
    double maxQuadrantRatio = static_cast<double>(maxQuadrantCount) / totalWhitePixels;
    
    if (maxQuadrantRatio > 0.5) {
        quadrantPenalty = 0.001;
    } else if (maxQuadrantDifference > totalWhitePixels * 0.2) {
        quadrantPenalty = 0.01;
    }
    
    return baseScore * translationPenalty * quadrantPenalty;
}

// Function to calculate score based on black pixels with quadrant uniformity penalty
double calculateBlackPixelScore(int totalBlackPixels, int translationDistance, const cv::Mat& thresholded, const cv::Point& center, int outerRadius, int innerRadius) {
    // Base score is just the number of black pixels
    double baseScore = static_cast<double>(totalBlackPixels);
    
    // If no black pixels, return 0
    if (totalBlackPixels == 0) {
        return 0.0;
    }
    
    // Count black pixels in each quadrant of the ring
    std::vector<int> quadrantCounts;
    for (int q = 0; q < 4; q++) {
        quadrantCounts.push_back(countBlackPixelsInQuadrant(thresholded, center, outerRadius, innerRadius, q));
    }
    
    // Calculate quadrant uniformity penalty (compare adjacent quadrants)
    double maxQuadrantDifference = 0;
    
    // Compare quadrant 0 vs 1 (top-right vs top-left)
    double diff01 = std::abs(quadrantCounts[0] - quadrantCounts[1]);
    maxQuadrantDifference = std::max(maxQuadrantDifference, diff01);
    
    // Compare quadrant 1 vs 2 (top-left vs bottom-left)
    double diff12 = std::abs(quadrantCounts[1] - quadrantCounts[2]);
    maxQuadrantDifference = std::max(maxQuadrantDifference, diff12);
    
    // Compare quadrant 2 vs 3 (bottom-left vs bottom-right)
    double diff23 = std::abs(quadrantCounts[2] - quadrantCounts[3]);
    maxQuadrantDifference = std::max(maxQuadrantDifference, diff23);
    
    // Compare quadrant 3 vs 0 (bottom-right vs top-right)
    double diff30 = std::abs(quadrantCounts[3] - quadrantCounts[0]);
    maxQuadrantDifference = std::max(maxQuadrantDifference, diff30);
    
    // Translation penalty removed - focusing on quadrant penalty only
    double translationPenalty = 1.0;
    
    // Apply quadrant uniformity penalty: reduce score for uneven quadrants
    double quadrantPenalty = 1.0;
    
    // EXTREME TEST: Check for ANY unevenness at all
    int maxQuadrantCount = *std::max_element(quadrantCounts.begin(), quadrantCounts.end());
    double maxQuadrantRatio = static_cast<double>(maxQuadrantCount) / totalBlackPixels;
    
    if (maxQuadrantRatio > 0.5) { // If any single quadrant has >50% of all black pixels
        // EXTREME penalty for ANY unevenness - should eliminate all cheaters
        quadrantPenalty = 0.001; // Reduce to 0.1% of original score (EXTREMELY harsh)
    } else if (maxQuadrantDifference > totalBlackPixels * 0.2) { // If adjacent quadrants differ by >20%
        // Also harsh penalty for moderate unevenness
        quadrantPenalty = 0.01; // Reduce to 1% of original score
    }
    
    return baseScore * translationPenalty * quadrantPenalty;
}

// Function to analyze gradient changes in quadrants with two-stage center refinement
std::tuple<std::vector<cv::Vec3f>, std::vector<CircleResult>, std::vector<DropletType>> refineCircleCenterWithTwoStageAnalysis(const cv::Mat& input, const std::vector<cv::Vec3f>& outerCircles) {
    std::vector<cv::Vec3f> refinedCircles;
    std::vector<CircleResult> winningCandidates;
    std::vector<DropletType> dropletTypes;
    
    // Enhance contrast before thresholding for consistent results across images
    cv::Mat enhanced = enhanceContrast(input);
    
    // Create thresholded image for clearer boundary detection
    cv::Mat thresholded;
    cv::threshold(enhanced, thresholded, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    
    for (const auto& outerCircle : outerCircles) {
        cv::Point originalCenter(cvRound(outerCircle[0]), cvRound(outerCircle[1]));
        int outerRadius = cvRound(outerCircle[2]);
        
        // Detect droplet type
        DropletType dropletType = detectDropletType(thresholded, originalCenter, outerRadius);
        std::cout << "Detected droplet type: " << (dropletType == RING_TYPE ? "RING (light)" : "FILLED (dark)") << std::endl;
        dropletTypes.push_back(dropletType); // Store for later use
        
        std::vector<CircleResult> candidates;
        
        // STAGE 1: Preliminary refinement within 2-pixel radius
        for (int dx = -2; dx <= 2; dx++) {
            for (int dy = -2; dy <= 2; dy++) {
                cv::Point testCenter = originalCenter + cv::Point(dx, dy);
                
                // Skip if center is too close to image edges
                if (testCenter.x < outerRadius || testCenter.y < outerRadius || 
                    testCenter.x >= thresholded.cols - outerRadius || 
                    testCenter.y >= thresholded.rows - outerRadius) {
                    continue;
                }
                
                // Choose radius range and target pixels based on droplet type
                int minRadius, maxRadius;
                if (dropletType == RING_TYPE) {
                    // RING TYPE: Search range 0.8x-1.3x, look for BLACK pixels (the ring)
                    minRadius = static_cast<int>(outerRadius * 0.8);
                    maxRadius = static_cast<int>(outerRadius * 1.3);
                    
                    // Look for BLACK pixels - find the dark ring boundary
                    for (int testRadius = minRadius; testRadius <= maxRadius; testRadius += 4) {
                        if (testRadius <= 20) continue; // Need space for 10-pixel rings
                        
                        int innerRadius = testRadius - 10;
                        int blackPixels = countBlackPixelsInRing(thresholded, testCenter, testRadius, innerRadius);
                        
                        // Calculate translation distance from original center
                        int translationDistance = static_cast<int>(sqrt(dx*dx + dy*dy));
                        
                        // Calculate score based on black pixels with translation and quadrant penalties
                        double score = calculateBlackPixelScore(blackPixels, translationDistance, thresholded, testCenter, testRadius, innerRadius);
                        
                        candidates.push_back(CircleResult(
                            cv::Vec3f(static_cast<float>(testCenter.x), static_cast<float>(testCenter.y), static_cast<float>(innerRadius)),
                            score, blackPixels, translationDistance, testRadius
                        ));
                    }
                } else {
                    // FILLED TYPE: Search range 0.6x-1.4x, look for WHITE pixels (outer boundary)
                    minRadius = static_cast<int>(outerRadius * 0.6);
                    maxRadius = static_cast<int>(outerRadius * 1.4);
                    
                    // Look for WHITE pixels - find where black circle meets white oil
                    for (int testRadius = minRadius; testRadius <= maxRadius; testRadius += 4) {
                        if (testRadius <= 20) continue; // Need space for 10-pixel rings
                        
                        int innerRadius = testRadius - 10;
                        int whitePixels = countWhitePixelsInRing(thresholded, testCenter, testRadius, innerRadius);
                        
                        // Calculate translation distance from original center
                        int translationDistance = static_cast<int>(sqrt(dx*dx + dy*dy));
                        
                        // Calculate score based on white pixels with translation and quadrant penalties
                        double score = calculateWhitePixelScore(whitePixels, translationDistance, thresholded, testCenter, testRadius, innerRadius);
                        
                        candidates.push_back(CircleResult(
                            cv::Vec3f(static_cast<float>(testCenter.x), static_cast<float>(testCenter.y), static_cast<float>(innerRadius)),
                            score, whitePixels, translationDistance, testRadius
                        ));
                    }
                }
            }
        }
        
        // STAGE 2: Secondary refinement with ±3 pixel translations (3-pixel increments)
        
        // Test Y-direction translations (±3 pixels in 3-pixel increments)
        for (int yOffset = -3; yOffset <= 3; yOffset += 3) {
            if (yOffset == 0) continue; // Skip center (already tested in stage 1)
            
            cv::Point yTestCenter = originalCenter + cv::Point(0, yOffset);
            
            if (yTestCenter.y >= outerRadius && yTestCenter.y < thresholded.rows - outerRadius) {
                // Use same strategy as STAGE 1
                int minRadius, maxRadius;
                if (dropletType == RING_TYPE) {
                    minRadius = static_cast<int>(outerRadius * 0.8);
                    maxRadius = static_cast<int>(outerRadius * 1.3);
                    
                    // RING TYPE: Look for BLACK pixels
                    for (int testRadius = minRadius; testRadius <= maxRadius; testRadius += 4) {
                        if (testRadius <= 20) continue;
                        
                        int innerRadius = testRadius - 10;
                        int blackPixels = countBlackPixelsInRing(thresholded, yTestCenter, testRadius, innerRadius);
                        
                        // Calculate translation distance from original center
                        int translationDistance = static_cast<int>(sqrt(yOffset * yOffset));
                        
                        // Calculate score based on black pixels with translation and quadrant penalties
                        double score = calculateBlackPixelScore(blackPixels, translationDistance, thresholded, yTestCenter, testRadius, innerRadius);
                        
                        candidates.push_back(CircleResult(
                            cv::Vec3f(static_cast<float>(yTestCenter.x), static_cast<float>(yTestCenter.y), static_cast<float>(innerRadius)),
                            score, blackPixels, translationDistance, testRadius
                        ));
                    }
                } else {
                    minRadius = static_cast<int>(outerRadius * 0.6);
                    maxRadius = static_cast<int>(outerRadius * 1.4);
                    
                    // FILLED TYPE: Look for WHITE pixels
                    for (int testRadius = minRadius; testRadius <= maxRadius; testRadius += 4) {
                        if (testRadius <= 20) continue;
                        
                        int innerRadius = testRadius - 10;
                        int whitePixels = countWhitePixelsInRing(thresholded, yTestCenter, testRadius, innerRadius);
                        
                        // Calculate translation distance from original center
                        int translationDistance = static_cast<int>(sqrt(yOffset * yOffset));
                        
                        // Calculate score based on white pixels with translation and quadrant penalties
                        double score = calculateWhitePixelScore(whitePixels, translationDistance, thresholded, yTestCenter, testRadius, innerRadius);
                        
                        candidates.push_back(CircleResult(
                            cv::Vec3f(static_cast<float>(yTestCenter.x), static_cast<float>(yTestCenter.y), static_cast<float>(innerRadius)),
                            score, whitePixels, translationDistance, testRadius
                        ));
                    }
                }
            }
        }
        
        // Test X-direction translations (±3 pixels in 3-pixel increments)
        for (int xOffset = -3; xOffset <= 3; xOffset += 3) {
            if (xOffset == 0) continue; // Skip center (already tested in stage 1)
            
            cv::Point xTestCenter = originalCenter + cv::Point(xOffset, 0);
            
            if (xTestCenter.x >= outerRadius && xTestCenter.x < thresholded.cols - outerRadius) {
                // Use same strategy as STAGE 1
                int minRadius, maxRadius;
                if (dropletType == RING_TYPE) {
                    minRadius = static_cast<int>(outerRadius * 0.8);
                    maxRadius = static_cast<int>(outerRadius * 1.3);
                    
                    // RING TYPE: Look for BLACK pixels
                    for (int testRadius = minRadius; testRadius <= maxRadius; testRadius += 4) {
                        if (testRadius <= 20) continue;
                        
                        int innerRadius = testRadius - 10;
                        int blackPixels = countBlackPixelsInRing(thresholded, xTestCenter, testRadius, innerRadius);
                        
                        // Calculate translation distance from original center
                        int translationDistance = static_cast<int>(sqrt(xOffset * xOffset));
                        
                        // Calculate score based on black pixels with translation and quadrant penalties
                        double score = calculateBlackPixelScore(blackPixels, translationDistance, thresholded, xTestCenter, testRadius, innerRadius);
                        
                        candidates.push_back(CircleResult(
                            cv::Vec3f(static_cast<float>(xTestCenter.x), static_cast<float>(xTestCenter.y), static_cast<float>(innerRadius)),
                            score, blackPixels, translationDistance, testRadius
                        ));
                    }
                } else {
                    minRadius = static_cast<int>(outerRadius * 0.6);
                    maxRadius = static_cast<int>(outerRadius * 1.4);
                    
                    // FILLED TYPE: Look for WHITE pixels
                    for (int testRadius = minRadius; testRadius <= maxRadius; testRadius += 4) {
                        if (testRadius <= 20) continue;
                        
                        int innerRadius = testRadius - 10;
                        int whitePixels = countWhitePixelsInRing(thresholded, xTestCenter, testRadius, innerRadius);
                        
                        // Calculate translation distance from original center
                        int translationDistance = static_cast<int>(sqrt(xOffset * xOffset));
                        
                        // Calculate score based on white pixels with translation and quadrant penalties
                        double score = calculateWhitePixelScore(whitePixels, translationDistance, thresholded, xTestCenter, testRadius, innerRadius);
                        
                        candidates.push_back(CircleResult(
                            cv::Vec3f(static_cast<float>(xTestCenter.x), static_cast<float>(xTestCenter.y), static_cast<float>(innerRadius)),
                            score, whitePixels, translationDistance, testRadius
                        ));
                    }
                }
            }
        }
        
        // Sort candidates by penalized score (highest to lowest)
        std::sort(candidates.begin(), candidates.end(), [](const CircleResult& a, const CircleResult& b) {
            return a.score > b.score;
        });
        
        // Debug output: Show top candidates by penalized score with quadrant info
        std::cout << "Circle " << (refinedCircles.size() + 1) << " candidates (sorted by penalized score):" << std::endl;
        for (size_t i = 0; i < std::min(candidates.size(), size_t(5)); i++) {
            cv::Point candidateCenter(cvRound(candidates[i].circle[0]), cvRound(candidates[i].circle[1]));
            int candidateRadius = cvRound(candidates[i].circle[2]);
            
            // Show quadrant distribution for top candidates
            std::cout << "  Rank " << (i+1) << ": score=" << candidates[i].score 
                      << ", " << (dropletType == RING_TYPE ? "black" : "white") << "_pixels=" << candidates[i].blackPixels
                      << ", translation_dist=" << candidates[i].translationDistance
                      << ", center=(" << candidates[i].circle[0] << "," << candidates[i].circle[1] << ")"
                      << ", radius=" << candidates[i].circle[2];
            
            if (i < 3) { // Show quadrant details for top 3 candidates
                std::cout << ", quadrants=[";
                for (int q = 0; q < 4; q++) {
                    int quadrantCount;
                    if (dropletType == RING_TYPE) {
                        quadrantCount = countBlackPixelsInQuadrant(thresholded, candidateCenter, candidates[i].outerRadius, candidateRadius, q);
                    } else {
                        quadrantCount = countWhitePixelsInQuadrant(thresholded, candidateCenter, candidates[i].outerRadius, candidateRadius, q);
                    }
                    std::cout << quadrantCount;
                    if (q < 3) std::cout << ",";
                }
                std::cout << "]";
            }
            std::cout << std::endl;
        }
        
        // Choose the candidate with highest penalized score
        CircleResult bestCandidate = candidates[0];
        
        // Use the actual winning inner radius from the candidate
        int finalInnerRadius = static_cast<int>(bestCandidate.circle[2]);
        
        // Create refined circle with optimized center and radius
        refinedCircles.push_back(cv::Vec3f(bestCandidate.circle[0], bestCandidate.circle[1], static_cast<float>(finalInnerRadius)));
        winningCandidates.push_back(bestCandidate);
    }
    
    return {refinedCircles, winningCandidates, dropletTypes};
}

// Function to draw winning outer and inner circles on thresholded image for visualization
cv::Mat drawWinningRings(const cv::Mat& thresholded, const std::vector<cv::Vec3f>& refinedCircles, const std::vector<cv::Vec3f>& originalCircles, const std::vector<CircleResult>& winningCandidates) {
    cv::Mat output = thresholded.clone();
    if (output.channels() == 1) {
        cv::cvtColor(output, output, cv::COLOR_GRAY2BGR);
    }
    
    for (size_t i = 0; i < refinedCircles.size(); i++) {
        cv::Point center(cvRound(refinedCircles[i][0]), cvRound(refinedCircles[i][1]));
        int innerRadius = cvRound(refinedCircles[i][2]);
        int outerRadius = winningCandidates[i].outerRadius; // Use the winning outer radius
        
            // Draw the MEASUREMENT AREA (what we're actually counting black pixels in) in yellow
            cv::circle(output, center, innerRadius, cv::Scalar(0, 255, 255), 2); // Yellow = measurement area
            
            // Draw outer boundary of the ring in green (thick line)
            cv::circle(output, center, outerRadius, cv::Scalar(0, 255, 0), 3);
            
            // Draw inner boundary of the ring in red (thick line) 
            cv::circle(output, center, outerRadius - 10, cv::Scalar(0, 0, 255), 3);
        
        // Draw center point
        cv::circle(output, center, 5, cv::Scalar(255, 255, 0), -1); // Yellow center
    }
    
    return output;
}

// Function to calculate droplet diameter from winning slice
double calculateDropletDiameter(const CircleResult& winningCandidate, double scaleFactor) {
    // Intermediate circle is halfway between outer and inner boundaries
    double intermediateRadius = (winningCandidate.outerRadius + (winningCandidate.outerRadius - 10)) / 2.0;
    double intermediateRadiusOriginal = intermediateRadius / scaleFactor;
    return intermediateRadiusOriginal * 2.0; // Return diameter
}

// Function to create mask for mean gray value calculation
cv::Mat createDropletMask(const cv::Vec3f& circle, int sliceWidth, double scaleFactor) {
    // Create mask with 0.8 * inner radius
    double innerRadiusScaled = circle[2];
    double maskRadius = innerRadiusScaled * 0.8;
    
    // Create mask on the scaled image (we'll use the upscaled image for mean calculation)
    cv::Mat mask = cv::Mat::zeros(cv::Size(static_cast<int>(circle[0] * 2 + maskRadius * 2), 
                                           static_cast<int>(circle[1] * 2 + maskRadius * 2)), CV_8UC1);
    cv::Point center(static_cast<int>(circle[0]), static_cast<int>(circle[1]));
    cv::circle(mask, center, static_cast<int>(maskRadius), cv::Scalar(255), -1);
    
    return mask;
}

// Function to find inner boundary using thresholded image jump analysis
std::vector<cv::Vec3f> findInnerBoundariesByThresholdedGradient(const cv::Mat& input, const std::vector<cv::Vec3f>& outerCircles) {
    std::vector<cv::Vec3f> innerCircles;
    
    // Enhance contrast before thresholding for consistent results across images
    cv::Mat enhanced = enhanceContrast(input);
    
    // Create thresholded image for clearer boundary detection
    cv::Mat thresholded;
    cv::threshold(enhanced, thresholded, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    
    for (const auto& outerCircle : outerCircles) {
        cv::Point center(cvRound(outerCircle[0]), cvRound(outerCircle[1]));
        int outerRadius = cvRound(outerCircle[2]);
        
        // NEW APPROACH: Look for largest jump in mean ring values on thresholded image
        double maxJump = 0;
        int bestInnerRadius = static_cast<int>(outerRadius * 0.95); // Default to 95% of outer radius
        std::vector<double> meanValues;
        std::vector<int> radii;
        
        // Test radii with fine resolution (droplet boundary is very thin)
        for (int testRadius = static_cast<int>(outerRadius * 0.99); testRadius >= static_cast<int>(outerRadius * 0.8); testRadius -= 1) {
            // Ensure testRadius is positive and large enough for ring creation
            if (testRadius <= 2) continue;
            
            // Create thin ring mask (pixels between testRadius and testRadius-2) for finer detection
            cv::Mat ringMask = cv::Mat::zeros(thresholded.size(), CV_8UC1);
            cv::circle(ringMask, center, testRadius, cv::Scalar(255), -1);
            cv::circle(ringMask, center, testRadius - 2, cv::Scalar(0), -1);
            
            // Calculate mean value of the ring on THRESHOLDED image
            cv::Scalar meanColor = cv::mean(thresholded, ringMask);
            double meanValue = meanColor[0];
            
            meanValues.push_back(meanValue);
            radii.push_back(testRadius);
        }
        
        // Find the radius with largest jump in mean values (boundary transition)
        for (size_t i = 1; i < meanValues.size(); i++) {
            double jump = abs(meanValues[i-1] - meanValues[i]);
            if (jump > maxJump) {
                maxJump = jump;
                bestInnerRadius = radii[i];
            }
        }
        
        // Ensure the inner radius is valid (positive and smaller than outer)
        int finalInnerRadius = std::max(5, std::min(bestInnerRadius, static_cast<int>(outerRadius * 0.95)));
        
        // Create inner circle with same center but smaller radius
        innerCircles.push_back(cv::Vec3f(outerCircle[0], outerCircle[1], static_cast<float>(finalInnerRadius)));
    }
    
    return innerCircles;
}

// Function to visualize detected circles (green for original, red for quadrant-refined)
cv::Mat drawCircles(const cv::Mat& input, const std::vector<cv::Vec3f>& circles, const std::vector<cv::Vec3f>& refinedCircles = {}) {
    cv::Mat output = input.clone();
    if (output.channels() == 1) {
        cv::cvtColor(output, output, cv::COLOR_GRAY2BGR);
    }
    
    // Draw original circles in green
    for (size_t i = 0; i < circles.size(); i++) {
        cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        
        // Draw the circle outline in green
        cv::circle(output, center, radius, cv::Scalar(0, 255, 0), 2);
        // Draw the center in blue
        cv::circle(output, center, 3, cv::Scalar(255, 0, 0), -1);
    }
    
    // Draw refined circles in red (quadrant-analyzed)
    for (size_t i = 0; i < refinedCircles.size(); i++) {
        cv::Point center(cvRound(refinedCircles[i][0]), cvRound(refinedCircles[i][1]));
        int radius = cvRound(refinedCircles[i][2]);
        
        // Draw the circle outline in red
        cv::circle(output, center, radius, cv::Scalar(0, 0, 255), 2);
        // Draw the center in red
        cv::circle(output, center, 3, cv::Scalar(0, 0, 255), -1);
    }
    
    return output;
}

//Function to create a new image, representing a mask, of the detected circles.
cv::Mat createCircleMask(const cv::Mat& input, const std::vector<cv::Vec3f>& circles) {
    // Get the dimensions of the input image.
    int rows = input.rows;
    int cols = input.cols;
    // reate a new Mat to store the mask. Make it fully black.
    cv::Mat mask = cv::Mat::zeros(rows, cols, CV_8UC1);
    // Draw a white circle on the black mask at the center of the circle.
    for (size_t i = 0; i < circles.size(); i++) {
        cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        cv::circle(mask, center, radius, cv::Scalar(255), -1);
    }
    // Return the mask.
    return mask;

}

// Function to calculate the mean grayscale value of the blurred image within the mask.
double calculateMeanGrayscaleValue(const cv::Mat& input, const cv::Mat& mask){
    cv::Scalar meanGrayscaleValue = cv::mean(input, mask);
    return meanGrayscaleValue[0];
}

// Function to parse filename and extract droplet number
int extractDropletNumber(const std::string& filename) {
    std::filesystem::path path(filename);
    std::string stem = path.stem().string();
    
    // Find the last underscore
    size_t lastUnderscore = stem.find_last_of('_');
    if (lastUnderscore == std::string::npos) {
        return 1; // Default to 1 if no underscore found
    }
    
    // Extract the number after the last underscore
    std::string numberStr = stem.substr(lastUnderscore + 1);
    
    try {
        return std::stoi(numberStr);
    } catch (const std::exception&) {
        return 1; // Default to 1 if conversion fails
    }
}

// Function to format diameter with 'p' for decimal places
std::string formatDiameter(double diameter) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << diameter;
    std::string result = oss.str();
    
    // Replace decimal point with 'p'
    size_t dotPos = result.find('.');
    if (dotPos != std::string::npos) {
        result[dotPos] = 'p';
    }
    
    return result;
}

// Function to format mean gray value with 'p' for decimal places
std::string formatMeanGrayValue(double meanValue) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << meanValue;
    std::string result = oss.str();
    
    // Replace decimal point with 'p'
    size_t dotPos = result.find('.');
    if (dotPos != std::string::npos) {
        result[dotPos] = 'p';
    }
    
    return result;
}

// Function to process a single image
void processImage(const std::string& imagePath, const std::string& outputFolder, bool debugMode = false) {
    std::filesystem::path path(imagePath);
    std::string imageName = path.filename().string();
    int dropletNumber = extractDropletNumber(imagePath);
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Processing: " << imageName << std::endl;
    std::cout << "Droplet number: " << dropletNumber << std::endl;
    if (debugMode) {
        std::cout << "Debug mode: Intermediate outputs will be saved" << std::endl;
    }
    std::cout << std::string(60, '=') << std::endl;
    
    // Load the image
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "ERROR: Could not load image from " << imagePath << std::endl;
        return;
    }
    
    // Convert to grayscale and upscale for sub-pixel precision
    cv::Mat gray = convertToGrayscale(image);
    double scaleFactor = 2.0; // Reduced from 4.0 to 2.0 for speed
    cv::Mat upscaledGray = upscaleForPrecision(gray, scaleFactor);
    
    // Debug output 1: Save upscaled original image
    if (debugMode) {
        std::string upscaleFilename = outputFolder + "/" + std::to_string(dropletNumber) + "_UPSCALE.jpg";
        cv::imwrite(upscaleFilename, upscaledGray);
        std::cout << "Debug: Saved upscaled image: " << upscaleFilename << std::endl;
    }
    
    // Add padding and apply Gaussian blur
    int padding = static_cast<int>(50 * scaleFactor);
    // Use corner-based mean for better background estimation (avoids dark droplet affecting padding color)
    cv::Scalar paddingColor = calculateCornerBasedMean(upscaledGray);
    
    // Debug output: show old vs new mean calculation
    if (debugMode) {
        cv::Scalar oldMeanColor = cv::mean(upscaledGray);
        std::cout << "Padding color - Old (full image mean): " << oldMeanColor[0] 
                  << ", New (corner-based mean): " << paddingColor[0] << std::endl;
    }
    
    cv::Mat padded;
    cv::copyMakeBorder(upscaledGray, padded, padding, padding, padding, padding, cv::BORDER_CONSTANT, paddingColor);
    // Using a blur value of 25 helps reduce over-detection of droplets.
    cv::Mat blurred = applyGaussianBlur(padded, 25);
    
    // Debug output 2: Save blurred image
    if (debugMode) {
        std::string blurFilename = outputFolder + "/" + std::to_string(dropletNumber) + "_BLUR.jpg";
        cv::imwrite(blurFilename, blurred);
        std::cout << "Debug: Saved blurred image: " << blurFilename << std::endl;
    }
    
    // Enhance contrast before thresholding for consistent results across images
    cv::Mat enhanced = enhanceContrast(blurred);
    
    // Debug output 2.5: Save enhanced contrast image
    if (debugMode) {
        std::string enhanceFilename = outputFolder + "/" + std::to_string(dropletNumber) + "_ENHANCE.jpg";
        cv::imwrite(enhanceFilename, enhanced);
        std::cout << "Debug: Saved enhanced contrast image: " << enhanceFilename << std::endl;
    }
    
    // Create thresholded image for analysis
    cv::Mat thresholded;
    cv::threshold(enhanced, thresholded, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    
    // Debug output 3: Save thresholded image
    if (debugMode) {
        std::string threshFilename = outputFolder + "/" + std::to_string(dropletNumber) + "_THRESH.jpg";
        cv::imwrite(threshFilename, thresholded);
        std::cout << "Debug: Saved thresholded image: " << threshFilename << std::endl;
    }
    
    // Detect circles with scaled parameters (using enhanced image for better consistency)
    int scaledMinDist = static_cast<int>(35 * scaleFactor);
    int scaledMinRadius = static_cast<int>(10 * scaleFactor);  // Reduced from 20 to 10 for smaller droplets
    int scaledMaxRadius = static_cast<int>(180 * scaleFactor);
    std::vector<cv::Vec3f> outerCircles = detectCircles(enhanced, 1.0, scaledMinDist, 45, 18, scaledMinRadius, scaledMaxRadius);
    
    // Debug output 4: Save original Hough circles with corner regions
    if (debugMode) {
        cv::Mat originalCirclesImg = enhanced.clone();
        if (originalCirclesImg.channels() == 1) {
            cv::cvtColor(originalCirclesImg, originalCirclesImg, cv::COLOR_GRAY2BGR);
        }
        
        // Visualize corner regions used for padding calculation (in red)
        // These corners are from the original upscaled image, now offset by padding
        int cornerSize = static_cast<int>(std::min(upscaledGray.cols, upscaledGray.rows) * 0.1);
        cornerSize = std::max(1, std::min(cornerSize, std::min(upscaledGray.cols, upscaledGray.rows) / 2));
        
        // Draw corner regions with offset for padding (showing where they were before padding)
        cv::Rect topLeft(padding, padding, cornerSize, cornerSize);
        cv::Rect topRight(padding + upscaledGray.cols - cornerSize, padding, cornerSize, cornerSize);
        cv::Rect bottomLeft(padding, padding + upscaledGray.rows - cornerSize, cornerSize, cornerSize);
        cv::Rect bottomRight(padding + upscaledGray.cols - cornerSize, padding + upscaledGray.rows - cornerSize, cornerSize, cornerSize);
        
        cv::rectangle(originalCirclesImg, topLeft, cv::Scalar(0, 0, 255), 2);
        cv::rectangle(originalCirclesImg, topRight, cv::Scalar(0, 0, 255), 2);
        cv::rectangle(originalCirclesImg, bottomLeft, cv::Scalar(0, 0, 255), 2);
        cv::rectangle(originalCirclesImg, bottomRight, cv::Scalar(0, 0, 255), 2);
        
        // Draw all detected circles in green
        for (const auto& circle : outerCircles) {
            cv::Point center(cvRound(circle[0]), cvRound(circle[1]));
            int radius = cvRound(circle[2]);
            cv::circle(originalCirclesImg, center, radius, cv::Scalar(0, 255, 0), 2);
            cv::circle(originalCirclesImg, center, 3, cv::Scalar(0, 0, 255), -1);
        }
        std::string originalCirclesFilename = outputFolder + "/" + std::to_string(dropletNumber) + "_ORIGINALCIRCLES.jpg";
        cv::imwrite(originalCirclesFilename, originalCirclesImg);
        std::cout << "Debug: Saved original circles image (" << outerCircles.size() << " circles) with corner regions in red: " << originalCirclesFilename << std::endl;
    }
    
    // STAGE 2 THRESHOLD: Create adaptive threshold based on detected circle edge
    // Tunable parameter: how much darker than edge mean should be considered black
    const double THRESHOLD_OFFSET = -10.0;  // 0 = at edge mean, positive = darker required, negative = lighter accepted
    
    cv::Mat secondThreshold;
    cv::Mat assistedThreshold;  // Will be used for refinement if we assist ring-type
    bool useAssistedThreshold = false;
    
    if (!outerCircles.empty()) {
        // For each detected circle, sample a 3-pixel slice at the edge
        std::vector<double> edgeMeans;
        
        for (const auto& circle : outerCircles) {
            cv::Point center(cvRound(circle[0]), cvRound(circle[1]));
            int detectedRadius = cvRound(circle[2]);
            
            // Create 3-pixel slice: outer radius = detected + 1, inner = detected - 1
            int outerSliceRadius = detectedRadius + 1;
            int innerSliceRadius = std::max(1, detectedRadius - 1);
            
            cv::Mat sliceMask = cv::Mat::zeros(blurred.size(), CV_8UC1);
            cv::circle(sliceMask, center, outerSliceRadius, cv::Scalar(255), -1);
            cv::circle(sliceMask, center, innerSliceRadius, cv::Scalar(0), -1);
            
            // Calculate mean gray value from the unthresholded blurred image
            cv::Scalar sliceMean = cv::mean(blurred, sliceMask);
            edgeMeans.push_back(sliceMean[0]);
            
            std::cout << "Circle edge mean gray value: " << sliceMean[0] << std::endl;
        }
        
        // Use the mean of all edge means as the threshold
        double avgEdgeMean = 0;
        for (double val : edgeMeans) {
            avgEdgeMean += val;
        }
        avgEdgeMean /= edgeMeans.size();
        
        // Apply tunable offset
        double thresholdValue = avgEdgeMean + THRESHOLD_OFFSET;
        std::cout << "Second threshold value (edge-based): " << thresholdValue << " (offset: " << THRESHOLD_OFFSET << ")" << std::endl;
        
        // Create second threshold: anything darker than threshold is black, lighter is white
        secondThreshold = cv::Mat::zeros(blurred.size(), blurred.type());
        for (int y = 0; y < blurred.rows; y++) {
            for (int x = 0; x < blurred.cols; x++) {
                if (blurred.at<uchar>(y, x) < thresholdValue) {
                    secondThreshold.at<uchar>(y, x) = 0;   // Black
                } else {
                    secondThreshold.at<uchar>(y, x) = 255; // White
                }
            }
        }
        
        // Debug output 4.75: Save second threshold image
        if (debugMode) {
            std::string secondThreshFilename = outputFolder + "/" + std::to_string(dropletNumber) + "_THRESH2.jpg";
            cv::imwrite(secondThreshFilename, secondThreshold);
            std::cout << "Debug: Saved second threshold image: " << secondThreshFilename << std::endl;
        }
        
        // ASSISTED RING-TYPE DETECTION: Evaluate if we should assist imperfect ring-type droplets
        assistedThreshold = secondThreshold.clone();  // Start with second threshold
        
        for (const auto& circle : outerCircles) {
            cv::Point center(cvRound(circle[0]), cvRound(circle[1]));
            int detectedRadius = cvRound(circle[2]);
            
            // Create mask for the inner area of the detected circle
            cv::Mat innerMask = cv::Mat::zeros(secondThreshold.size(), CV_8UC1);
            cv::circle(innerMask, center, detectedRadius, cv::Scalar(255), -1);
            
            // Count white and black pixels inside the circle
            int totalPixels = 0;
            int whitePixels = 0;
            for (int y = 0; y < secondThreshold.rows; y++) {
                for (int x = 0; x < secondThreshold.cols; x++) {
                    if (innerMask.at<uchar>(y, x) == 255) {
                        totalPixels++;
                        if (secondThreshold.at<uchar>(y, x) == 255) {
                            whitePixels++;
                        }
                    }
                }
            }
            
            double whiteRatio = static_cast<double>(whitePixels) / totalPixels;
            std::cout << "Inner circle white ratio: " << std::fixed << std::setprecision(3) << whiteRatio 
                      << " (" << whitePixels << "/" << totalPixels << ")" << std::endl;
            
            if (whiteRatio > 0.15) {
                // More than 15% white → likely imperfect ring-type, assist it!
                // Use 90% of detected radius to preserve the black ring
                int fillRadius = static_cast<int>(detectedRadius * 0.9);
                std::cout << "  → ASSISTING: Filling inner circle with white at 90% radius (" << fillRadius << " px, ring-type detected)" << std::endl;
                cv::circle(assistedThreshold, center, fillRadius, cv::Scalar(255), -1);
                useAssistedThreshold = true;
            } else {
                std::cout << "  → No assistance needed (likely fill-type)" << std::endl;
            }
        }
        
        // Debug output 4.8: Save assisted threshold image (if different from second threshold)
        if (debugMode && useAssistedThreshold) {
            std::string assistedThreshFilename = outputFolder + "/" + std::to_string(dropletNumber) + "_ASSISTED.jpg";
            cv::imwrite(assistedThreshFilename, assistedThreshold);
            std::cout << "Debug: Saved ASSISTED threshold image: " << assistedThreshFilename << std::endl;
        }
    }
    
    // Refinement: Use assisted threshold if available, otherwise use original enhanced
    cv::Mat refinementInput = (useAssistedThreshold && !assistedThreshold.empty()) ? assistedThreshold : enhanced;
    std::cout << "Using " << (useAssistedThreshold ? "ASSISTED" : "ORIGINAL") << " threshold for refinement" << std::endl;
    auto [refinedCircles, winningCandidates, detectedDropletTypes] = refineCircleCenterWithTwoStageAnalysis(refinementInput, outerCircles);
    
    // Debug output 4.5: Add droplet type indicator to ORIGINALCIRCLES image
    if (debugMode && !detectedDropletTypes.empty()) {
        // Re-load the saved ORIGINALCIRCLES image to add the type indicator
        std::string originalCirclesFilename = outputFolder + "/" + std::to_string(dropletNumber) + "_ORIGINALCIRCLES.jpg";
        cv::Mat originalCirclesImg = cv::imread(originalCirclesFilename);
        
        if (!originalCirclesImg.empty()) {
            // Draw type indicator square at middle left of image
            int squareSize = 30;
            int yPosition = originalCirclesImg.rows / 2 - squareSize / 2;
            cv::Rect indicatorRect(10, yPosition, squareSize, squareSize);
            
            // White square for RING_TYPE, Black square for FILLED_TYPE
            cv::Scalar squareColor = (detectedDropletTypes[0] == RING_TYPE) ? cv::Scalar(255, 255, 255) : cv::Scalar(0, 0, 0);
            cv::rectangle(originalCirclesImg, indicatorRect, squareColor, -1); // Filled rectangle
            cv::rectangle(originalCirclesImg, indicatorRect, cv::Scalar(128, 128, 128), 2); // Gray border for visibility
            
            // Re-save the image with the indicator
            cv::imwrite(originalCirclesFilename, originalCirclesImg);
            std::cout << "Debug: Added droplet type indicator: " << (detectedDropletTypes[0] == RING_TYPE ? "WHITE (RING)" : "BLACK (FILLED)") << std::endl;
        }
    }
    
    // Scale circles back to original image coordinates and remove padding offset
    std::vector<cv::Vec3f> originalRefinedCircles = scaleCirclesBack(refinedCircles, scaleFactor);
    int originalPadding = static_cast<int>(padding / scaleFactor);
    for (auto& circle : originalRefinedCircles) {
        circle[0] -= originalPadding;
        circle[1] -= originalPadding;
    }
    
    // Create base QC image: upscaled grayscale image
    cv::Mat qcImage = blurred.clone(); // Use the upscaled blurred grayscale image
    if (qcImage.channels() == 1) {
        cv::cvtColor(qcImage, qcImage, cv::COLOR_GRAY2BGR); // Convert to BGR for colored circles
    }
    
    // Check if valid (single droplet detected)
    if (originalRefinedCircles.size() == 0) {
        std::cout << "STATUS: Invalid - No droplets detected" << std::endl;
        std::string qcFilename = outputFolder + "/" + std::to_string(dropletNumber) + "_INVALID_0p00_0p00.jpg";
        cv::imwrite(qcFilename, qcImage);
        std::cout << "QC image saved: " << qcFilename << std::endl;
        // QC image saved - no display needed for command-line usage
    } else if (originalRefinedCircles.size() > 1) {
        std::cout << "STATUS: Invalid - Multiple droplets detected (" << originalRefinedCircles.size() << ")" << std::endl;
        
        // Draw red circles for each detected droplet (invalid style)
        for (size_t i = 0; i < refinedCircles.size(); i++) {
            cv::Point center(static_cast<int>(refinedCircles[i][0]), static_cast<int>(refinedCircles[i][1]));
            int radius = static_cast<int>(refinedCircles[i][2]); // Use inner radius
            cv::circle(qcImage, center, radius, cv::Scalar(0, 0, 255), 2); // Red circle
        }
        
        std::string qcFilename = outputFolder + "/" + std::to_string(dropletNumber) + "_INVALID_0p00_0p00.jpg";
        cv::imwrite(qcFilename, qcImage);
        std::cout << "QC image saved: " << qcFilename << std::endl;
        // QC image saved - no display needed for command-line usage
    } else {
        std::cout << "STATUS: Valid - Single droplet detected" << std::endl;
        
        // Check if droplet is partially out of frame
        cv::Vec3f detectedCircle = originalRefinedCircles[0];
        float centerX = detectedCircle[0];
        float centerY = detectedCircle[1];
        float radius = detectedCircle[2];
        
        // Get original image dimensions (before upscaling and padding)
        int originalWidth = gray.cols;
        int originalHeight = gray.rows;
        
        // Check if droplet extends beyond image boundaries (allow 20% overhang)
        float allowedOverhang = radius * 0.20f; // Allow 20% of radius to extend beyond boundary
        bool outOfFrame = (centerX - radius < -allowedOverhang) || (centerX + radius > originalWidth + allowedOverhang) ||
                          (centerY - radius < -allowedOverhang) || (centerY + radius > originalHeight + allowedOverhang);
        
        if (outOfFrame) {
            std::cout << "STATUS: Invalid - Droplet partially out of frame" << std::endl;
            std::cout << "  Center: (" << centerX << ", " << centerY << "), Radius: " << radius << std::endl;
            std::cout << "  Image bounds: " << originalWidth << "x" << originalHeight << std::endl;
            
            // Draw orange circle for out-of-frame droplet
            cv::Point qcCenter(static_cast<int>(refinedCircles[0][0]), static_cast<int>(refinedCircles[0][1]));
            cv::circle(qcImage, qcCenter, static_cast<int>(refinedCircles[0][2]), cv::Scalar(0, 165, 255), 2); // Orange circle
            
            std::string qcFilename = outputFolder + "/" + std::to_string(dropletNumber) + "_INVALID_0p00_0p00.jpg";
            cv::imwrite(qcFilename, qcImage);
            std::cout << "QC image saved: " << qcFilename << std::endl;
            // QC image saved - no display needed for command-line usage
            std::cout << std::string(60, '=') << std::endl;
            return;
        }
        
        // Calculate droplet diameter (intermediate circle between outer and inner boundaries)
        double dropletDiameter = calculateDropletDiameter(winningCandidates[0], scaleFactor);
        double dropletRadius = dropletDiameter / 2.0;
        
        // Calculate mean gray value using inner radius mask on upscaled image
        cv::Vec3f scaledCircle = refinedCircles[0]; // This is already on the padded image
        double maskRadius = scaledCircle[2]; // Use full inner radius
        cv::Mat mask = cv::Mat::zeros(blurred.size(), CV_8UC1); // Use blurred (padded) image size
        cv::Point center(static_cast<int>(scaledCircle[0]), static_cast<int>(scaledCircle[1]));
        cv::circle(mask, center, static_cast<int>(maskRadius), cv::Scalar(255), -1);
        double meanGrayValue = calculateMeanGrayscaleValue(blurred, mask);
        
        // Output results
        std::cout << "Droplet diameter: " << std::fixed << std::setprecision(2) << dropletDiameter << " pixels" << std::endl;
        std::cout << "Droplet radius: " << std::fixed << std::setprecision(2) << dropletRadius << " pixels" << std::endl;
        std::cout << "Mean gray value: " << std::fixed << std::setprecision(2) << meanGrayValue << std::endl;
        
        // Draw green circle for single droplet (valid style)
        cv::Point qcCenter(static_cast<int>(scaledCircle[0]), static_cast<int>(scaledCircle[1])); // Use scaled coordinates
        cv::circle(qcImage, qcCenter, static_cast<int>(maskRadius), cv::Scalar(0, 255, 0), 2); // Green circle
        
        // Format diameter and mean gray value for filename
        std::string diameterStr = formatDiameter(dropletDiameter);
        std::string meanGrayStr = formatMeanGrayValue(meanGrayValue);
        
        std::string qcFilename = outputFolder + "/" + std::to_string(dropletNumber) + "_VALID_" + diameterStr + "_" + meanGrayStr + ".jpg";
        cv::imwrite(qcFilename, qcImage);
        std::cout << "QC image saved: " << qcFilename << std::endl;
        // QC image saved - no display needed for command-line usage
    }
    
    std::cout << std::string(60, '=') << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "=== DropletAnalyzer Starting ===" << std::endl;
    
    // Check command line arguments
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " <input_filepath> <output_folder> [--debug]" << std::endl;
        std::cerr << "Example: " << argv[0] << " \"C:/images/0_20241203-104610-955_91_00168.jpg\" \"C:/output\"" << std::endl;
        std::cerr << "         " << argv[0] << " \"C:/images/0_20241203-104610-955_91_00168.jpg\" \"C:/output\" --debug" << std::endl;
        return 1;
    }
    
    std::string inputFilepath = argv[1];
    std::string outputFolder = argv[2];
    bool debugMode = false;
    
    // Check for debug flag
    if (argc == 4) {
        std::string debugArg = argv[3];
        if (debugArg == "--debug" || debugArg == "-d") {
            debugMode = true;
            std::cout << "Debug mode enabled - intermediate outputs will be saved" << std::endl;
        }
    }
    
    // Check if input file exists
    if (!std::filesystem::exists(inputFilepath)) {
        std::cerr << "ERROR: Input file does not exist: " << inputFilepath << std::endl;
        return 1;
    }
    
    // Create output folder if it doesn't exist
    if (!std::filesystem::exists(outputFolder)) {
        try {
            std::filesystem::create_directories(outputFolder);
            std::cout << "Created output folder: " << outputFolder << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Could not create output folder " << outputFolder << ": " << e.what() << std::endl;
            return 1;
        }
    }
    
    std::cout << "Input file: " << inputFilepath << std::endl;
    std::cout << "Output folder: " << outputFolder << std::endl;
    
    // Process the single image
    processImage(inputFilepath, outputFolder, debugMode);
    
    std::cout << "\n=== Image processed successfully! ===" << std::endl;
    
    return 0;
}