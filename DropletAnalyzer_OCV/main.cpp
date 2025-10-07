#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <iomanip>

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
std::pair<std::vector<cv::Vec3f>, std::vector<CircleResult>> refineCircleCenterWithTwoStageAnalysis(const cv::Mat& input, const std::vector<cv::Vec3f>& outerCircles) {
    std::vector<cv::Vec3f> refinedCircles;
    std::vector<CircleResult> winningCandidates;
    
    // Create thresholded image for clearer boundary detection
    cv::Mat thresholded;
    cv::threshold(input, thresholded, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    
    // Dilation removed - was making black pixels disappear from detection radius
    
    for (const auto& outerCircle : outerCircles) {
        cv::Point originalCenter(cvRound(outerCircle[0]), cvRound(outerCircle[1]));
        int outerRadius = cvRound(outerCircle[2]);
        
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
                
                // Test different radii for this center - start larger than detected radius
                for (int testRadius = static_cast<int>(outerRadius * 1.2); testRadius >= static_cast<int>(outerRadius * 0.4); testRadius -= 2) {
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
            }
        }
        
        // STAGE 2: Secondary refinement with ±4 pixel translations (2-pixel increments)
        
        // Test Y-direction translations (±4 pixels in 2-pixel increments)
        for (int yOffset = -4; yOffset <= 4; yOffset += 2) {
            if (yOffset == 0) continue; // Skip center (already tested in stage 1)
            
            cv::Point yTestCenter = originalCenter + cv::Point(0, yOffset);
            
            if (yTestCenter.y >= outerRadius && yTestCenter.y < thresholded.rows - outerRadius) {
                for (int testRadius = static_cast<int>(outerRadius * 1.2); testRadius >= static_cast<int>(outerRadius * 0.8); testRadius -= 2) {
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
            }
        }
        
        // Test X-direction translations (±4 pixels in 2-pixel increments)
        for (int xOffset = -4; xOffset <= 4; xOffset += 2) {
            if (xOffset == 0) continue; // Skip center (already tested in stage 1)
            
            cv::Point xTestCenter = originalCenter + cv::Point(xOffset, 0);
            
            if (xTestCenter.x >= outerRadius && xTestCenter.x < thresholded.cols - outerRadius) {
                for (int testRadius = static_cast<int>(outerRadius * 1.2); testRadius >= static_cast<int>(outerRadius * 0.8); testRadius -= 2) {
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
                      << ", black_pixels=" << candidates[i].blackPixels 
                      << ", translation_dist=" << candidates[i].translationDistance
                      << ", center=(" << candidates[i].circle[0] << "," << candidates[i].circle[1] << ")"
                      << ", radius=" << candidates[i].circle[2];
            
            if (i < 3) { // Show quadrant details for top 3 candidates
                std::cout << ", quadrants=[";
                for (int q = 0; q < 4; q++) {
                    int quadrantCount = countBlackPixelsInQuadrant(thresholded, candidateCenter, candidates[i].outerRadius, candidateRadius, q);
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
    
    return {refinedCircles, winningCandidates};
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
    
    // Create thresholded image for clearer boundary detection
    cv::Mat thresholded;
    cv::threshold(input, thresholded, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    
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

// Function to process a single image
void processImage(const std::string& imagePath, const std::string& imageName, int imageIndex) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Processing: " << imageName << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    // Load the image
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "ERROR: Could not load image from " << imagePath << std::endl;
        return;
    }
    
    // Convert to grayscale and upscale for sub-pixel precision
    cv::Mat gray = convertToGrayscale(image);
    double scaleFactor = 4.0;
    cv::Mat upscaledGray = upscaleForPrecision(gray, scaleFactor);
    
    // Add padding and apply Gaussian blur
    int padding = static_cast<int>(50 * scaleFactor);
    cv::Scalar meanColor = cv::mean(upscaledGray);
    cv::Scalar paddingColor = cv::Scalar(meanColor[0]);
    cv::Mat padded;
    cv::copyMakeBorder(upscaledGray, padded, padding, padding, padding, padding, cv::BORDER_CONSTANT, paddingColor);
    cv::Mat blurred = applyGaussianBlur(padded, 5);
    
    // Create thresholded image for analysis
    cv::Mat thresholded;
    cv::threshold(blurred, thresholded, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    
    // Detect circles with scaled parameters
    int scaledMinDist = static_cast<int>(35 * scaleFactor);
    int scaledMinRadius = static_cast<int>(20 * scaleFactor);
    int scaledMaxRadius = static_cast<int>(180 * scaleFactor);
    std::vector<cv::Vec3f> outerCircles = detectCircles(blurred, 1.0, scaledMinDist, 45, 18, scaledMinRadius, scaledMaxRadius);
    
    // Refine centers using two-stage analysis
    auto [refinedCircles, winningCandidates] = refineCircleCenterWithTwoStageAnalysis(blurred, outerCircles);
    
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
        std::string qcFilename = "C:/Dev/DropletAnalyzer/DropletAnalyzer_OCV/test_data_out/QC_INVALID_" + imageName + ".jpg";
        cv::imwrite(qcFilename, qcImage);
        std::cout << "QC image saved: " << qcFilename << std::endl;
        cv::imshow("QC_" + imageName, resizeForDisplay(qcImage));
    } else if (originalRefinedCircles.size() > 1) {
        std::cout << "STATUS: Invalid - Multiple droplets detected (" << originalRefinedCircles.size() << ")" << std::endl;
        
        // Draw red circles for each detected droplet (invalid style)
        for (size_t i = 0; i < refinedCircles.size(); i++) {
            cv::Point center(static_cast<int>(refinedCircles[i][0]), static_cast<int>(refinedCircles[i][1]));
            int radius = static_cast<int>(refinedCircles[i][2]); // Use inner radius
            cv::circle(qcImage, center, radius, cv::Scalar(0, 0, 255), 2); // Red circle
        }
        
        std::string qcFilename = "C:/Dev/DropletAnalyzer/DropletAnalyzer_OCV/test_data_out/QC_INVALID_" + imageName + ".jpg";
        cv::imwrite(qcFilename, qcImage);
        std::cout << "QC image saved: " << qcFilename << std::endl;
        cv::imshow("QC_" + imageName, resizeForDisplay(qcImage));
    } else {
        std::cout << "STATUS: Valid - Single droplet detected" << std::endl;
        
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
        
        std::string qcFilename = "C:/Dev/DropletAnalyzer/DropletAnalyzer_OCV/test_data_out/QC_VALID_" + imageName + ".jpg";
        cv::imwrite(qcFilename, qcImage);
        std::cout << "QC image saved: " << qcFilename << std::endl;
        cv::imshow("QC_" + imageName, resizeForDisplay(qcImage));
    }
    
    std::cout << std::string(60, '=') << std::endl;
}

int main() {
    std::cout << "=== DropletAnalyzer Starting ===" << std::endl;
    
    // Process both test images
    std::vector<std::pair<std::string, std::string>> images = {
        {"C:/Dev/DropletAnalyzer/DropletAnalyzer_OCV/test_data_in/0_20241203-104614-957_90_00139.jpg", "Single Circle Image"},
        {"C:/Dev/DropletAnalyzer/DropletAnalyzer_OCV/test_data_in/0_20241203-104613-962_84_00145.jpg", "Multiple Circles Image"}
    };
    
    for (size_t i = 0; i < images.size(); i++) {
        processImage(images[i].first, images[i].second, static_cast<int>(i + 1));
    }
    
    std::cout << "\n=== All images processed successfully! ===" << std::endl;
    std::cout << "Press any key to close all windows..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    return 0;
}