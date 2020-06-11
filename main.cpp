#include <iostream>

#include "opencv2/opencv.hpp"
#if !defined(HAVE_OPENCV_XFEATURES2D)
    #error "OpenCV is not compiled with contrib modules. Please use OpenCV with contrib modules."
#endif
#include "opencv2/xfeatures2d.hpp"

#include "paths.hpp"


int main(int const argc, char* argv[])
{
    std::cout << "OpenCV v" << cv::getVersionString() << "\n" << std::endl;
    std::cout << IMAGE_ONE.string() << std::endl;
    std::cout << IMAGE_TWO.string() << std::endl;

    // Load the image
    auto imageOne = cv::imread(IMAGE_ONE.string(), cv::IMREAD_UNCHANGED);
    auto imageTwo = cv::imread(IMAGE_TWO.string(), cv::IMREAD_UNCHANGED);
    cv::rotate(imageOne, imageOne, cv::ROTATE_90_CLOCKWISE);
    cv::rotate(imageTwo, imageTwo, cv::ROTATE_90_CLOCKWISE);

    // Detect keypoints and compute descriptors
    auto const minHessian = 1000;
    auto detector = cv::xfeatures2d::SIFT::create(minHessian);
    std::vector<cv::KeyPoint> keypointsOne, keypointsTwo;
    cv::Mat descriptorsOne, descriptorsTwo;
    detector->detectAndCompute(imageOne, cv::noArray(), keypointsOne, descriptorsOne);
    detector->detectAndCompute(imageTwo, cv::noArray(), keypointsTwo, descriptorsTwo);

    // Match descriptors
    auto matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher->knnMatch(descriptorsOne, descriptorsTwo, knnMatches, 2);

    //-- Filter matches using the Lowe's ratio test
    auto const ratioThresh = 0.6f;
    std::vector<cv::DMatch> goodMatches;
    for (auto& knnMatche : knnMatches)
    {
        if (knnMatche[0].distance < ratioThresh * knnMatche[1].distance)
            goodMatches.push_back(knnMatche[0]);
    }

    // Draw matches
    cv::Mat matchesImage;
    cv::drawMatches(imageOne, keypointsOne, imageTwo, keypointsTwo, goodMatches, matchesImage,
            cv::Scalar::all(-1),
        cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // Draw keypoints
    //cv::Mat imageKeypointsOne, imageKeypointsTwo;
    //cv::drawKeypoints(imageOne, keypointsOne, imageKeypointsOne);
    //cv::drawKeypoints(imageTwo, keypointsTwo, imageKeypointsTwo);

    //cv::imshow("SIFT keypoints ImageOne", imageKeypointsOne);
    cv::imshow("Matches: ", matchesImage);
    //cv::imshow("SIFT keypoints ImageTwo", imageKeypointsTwo);
    cv::waitKey();
    cv::destroyAllWindows();

    // Wait for user to close console
    system("Pause");

    return EXIT_SUCCESS;
}
