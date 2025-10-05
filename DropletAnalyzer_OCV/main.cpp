#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

// Function to convert to grayscale
cv::Mat convertToGrayscale(const cv::Mat& input) {
    // Make an empty Mat to store the grayscale image.
    cv::Mat gray;
    // Convert the image to grayscale.
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

// Function to apply Gaussian blur for noise reduction
cv::Mat applyGaussianBlur(const cv::Mat& input, int kernelSize = 5) {
    // Make an empty Mat to store the blurred image.
    cv::Mat blurred;
    // Apply Gaussian blur to the image.
    cv::GaussianBlur(input, blurred, cv::Size(kernelSize, kernelSize), 0);
    return blurred;
}

// Function to detect circles using Hough Transform
std::vector<cv::Vec3f> detectCircles(const cv::Mat& input, double dp = 1.0, double minDist = 50, 
    double param1 = 100, double param2 = 30, 
    int minRadius = 10, int maxRadius = 100) {
std::vector<cv::Vec3f> circles;
cv::HoughCircles(input, circles, cv::HOUGH_GRADIENT, dp, minDist, param1, param2, minRadius, maxRadius);
return circles;
}

// Function to visualize detected circles
cv::Mat drawCircles(const cv::Mat& input, const std::vector<cv::Vec3f>& circles) {
cv::Mat output = input.clone();
if (output.channels() == 1) {
cv::cvtColor(output, output, cv::COLOR_GRAY2BGR);
}
// Draw the circles by iterating over the circle's vector and drawing a circle for each one.
for (size_t i = 0; i < circles.size(); i++) {
cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
int radius = cvRound(circles[i][2]);

// Draw the circle outline
cv::circle(output, center, radius, cv::Scalar(0, 255, 0), 2);
// Draw the center
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

int main() {
    std::cout << "=== DropletAnalyzer Starting ===" << std::endl;
    
    // Test image path - use absolute path to avoid working directory issues
    std::string imagePath = "C:/Dev/DropletAnalyzer/DropletAnalyzer_OCV/test_data_in/0_20241203-104611-960_89_00158.jpg";
    std::cout << "Looking for image at: " << imagePath << std::endl;
    
    // Load the image
    std::cout << "Attempting to load image..." << std::endl;
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    
    if (image.empty()) {
        std::cerr << "ERROR: Could not load image from " << imagePath << std::endl;
        std::cerr << "Make sure the file exists and the path is correct!" << std::endl;
        return -1;
    }
    
    std::cout << "SUCCESS: Image loaded!" << std::endl;
    std::cout << "Dimensions: " << image.cols << " x " << image.rows << std::endl;
    std::cout << "Channels: " << image.channels() << std::endl;
    
    // Step 1: Convert to grayscale
    std::cout << "\n=== Step 1: Converting to grayscale ===" << std::endl;
    cv::Mat gray = convertToGrayscale(image);
    std::cout << "Grayscale conversion completed" << std::endl;
    
    std::cout << "Attempting to save grayscale image..." << std::endl;
    bool save1 = cv::imwrite("C:/Dev/DropletAnalyzer/DropletAnalyzer_OCV/test_data_out/01_grayscale.jpg", gray);
    if (save1) {
        std::cout << "SUCCESS: Saved 01_grayscale.jpg" << std::endl;
    } else {
        std::cout << "ERROR: Failed to save 01_grayscale.jpg" << std::endl;
    }
    
    // Step 2: Apply Gaussian blur
    std::cout << "\n=== Step 2: Applying Gaussian blur ===" << std::endl;
    cv::Mat blurred = applyGaussianBlur(gray, 5);
    std::cout << "Gaussian blur completed" << std::endl;
    
    std::cout << "Attempting to save blurred image..." << std::endl;
    bool save2 = cv::imwrite("C:/Dev/DropletAnalyzer/DropletAnalyzer_OCV/test_data_out/02_blurred.jpg", blurred);
    if (save2) {
        std::cout << "SUCCESS: Saved 02_blurred.jpg" << std::endl;
    } else {
        std::cout << "ERROR: Failed to save 02_blurred.jpg" << std::endl;
    }

    // Step 3: Detect circles
    std::cout << "\n=== Step 4: Detecting circles ===" << std::endl;
    std::cout << "Using HoughCircles with optimized parameters for droplets:" << std::endl;
    std::cout << "  - dp: 1.0 (inverse ratio of accumulator resolution)" << std::endl;
    std::cout << "  - minDist: 30 (minimum distance between centers - reduced for closer droplets)" << std::endl;
    std::cout << "  - param1: 50 (upper threshold for edge detection - lowered for better sensitivity)" << std::endl;
    std::cout << "  - param2: 30 (accumulator threshold for center detection)" << std::endl;
    std::cout << "  - minRadius: 5, maxRadius: 200 (expanded range for various droplet sizes)" << std::endl;
    
    // Try with more sensitive parameters for droplet detection
    std::vector<cv::Vec3f> circles = detectCircles(blurred, 1.0, 30, 50, 30, 5, 200);
    std::cout << "SUCCESS: Detected " << circles.size() << " circle(s)" << std::endl;
    
    if (circles.size() > 0) {
        std::cout << "Circle details:" << std::endl;
        for (size_t i = 0; i < circles.size(); i++) {
            std::cout << "  Circle " << (i+1) << ": center(" << circles[i][0] 
                      << ", " << circles[i][1] << "), radius=" << circles[i][2] << std::endl;
        }
    } else {
        std::cout << "WARNING: No circles detected! This could be due to:" << std::endl;
        std::cout << "  - Wrong parameter values" << std::endl;
        std::cout << "  - Image quality issues" << std::endl;
        std::cout << "  - Droplets not circular enough" << std::endl;
    }

    // Step 4: Visualize detected circles
    cv::Mat circlesVis = drawCircles(image, circles);
    std::cout << "Attempting to save circles visualization..." << std::endl;
    bool save4 = cv::imwrite("C:/Dev/DropletAnalyzer/DropletAnalyzer_OCV/test_data_out/03_circles_detected.jpg", circlesVis);
    if (save4) {
        std::cout << "SUCCESS: Saved 03_circles_detected.jpg" << std::endl;
    } else {
        std::cout << "ERROR: Failed to save 03_circles_detected.jpg" << std::endl;
    }
    // Step 5: Create a mask of the detected circle but only if there is one circle detected.
    cv::Mat mask;
    if (circles.size() == 0) {
        std::cout << "Invalid: No droplets detected" << std::endl;
        // Don't create mask, mark as invalid
    } else if (circles.size() > 1) {
        std::cout << "Invalid: Multiple droplets detected (" << circles.size() << ")" << std::endl;
        // Don't create mask, mark as invalid
    } else {
        std::cout << "Valid: Single droplet detected" << std::endl;
        // Create mask and calculate grayscale value
        mask = createCircleMask(blurred, circles);
        double meanGrayscaleValue = calculateMeanGrayscaleValue(blurred, mask);
        std::cout << "Mean grayscale value: " << meanGrayscaleValue << std::endl;
        std::cout << "Attempting to save mask..." << std::endl;
        bool save5 = cv::imwrite("C:/Dev/DropletAnalyzer/DropletAnalyzer_OCV/test_data_out/04_mask.jpg", mask);
        if (save5) {
            std::cout << "SUCCESS: Saved 04_mask.jpg" << std::endl;
        } else {
            std::cout << "ERROR: Failed to save 04_mask.jpg" << std::endl;
        }
    }
        
    // Display results
    std::cout << "\n=== Displaying Results ===" << std::endl;
    std::cout << "Attempting to show Original window..." << std::endl;
    cv::imshow("Original", image);
    
    std::cout << "Attempting to show Grayscale window..." << std::endl;
    cv::imshow("Grayscale", gray);
    
    std::cout << "Attempting to show Blurred window..." << std::endl;
    cv::imshow("Blurred", blurred);
    
    std::cout << "Attempting to show Circles window..." << std::endl;
    cv::imshow("Circles", circlesVis);

    std::cout << "Attempting to show Mask window..." << std::endl;
    cv::imshow("Mask", mask);
    
    std::cout << "\nAll windows should be visible now." << std::endl;
    std::cout << "Press any key to close all windows..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    std::cout << "\n=== Program completed successfully! ===" << std::endl;
    return 0;
}