#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main( int argc, char** argv ) {

    // input two images
    Mat image1 = imread("../1.png");
    Mat image2 = imread("../2.png");

    // initialize
    vector<KeyPoint> keyPoints1, keyPoints2;
    Mat descriptors1, descriptors2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create( "BruteForce-Hamming" );

    // detect ORB features
    detector->detect(image1, keyPoints1);
    detector->detect(image2, keyPoints2);
    // draw key points of ORB feature in image 1 and image 2, respectively
    Mat detectedImage1, detectedImage2;
    drawKeypoints(image1, keyPoints1, detectedImage1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    drawKeypoints(image2, keyPoints2, detectedImage2);
    imshow("ORB features in image 1", detectedImage1);
    imshow("ORB features in image 2", detectedImage2);
    waitKey(0);

    // computer BRIEF descriptor of each ORB feature
    descriptor->compute(image1, keyPoints1, descriptors1);
    descriptor->compute(image2, keyPoints2, descriptors2);

    // match each BRIEF descriptor in two images using Hamming distance
    vector<DMatch> matches;
    matcher->match( descriptors1, descriptors2, matches);
    cout << "There are " << matches.size() << " matches." << endl;

    // filter the matches
    double maximalHammingDistance = 0, minimalHammingDistance = 1000;
    for (int i = 0; i < descriptors1.rows; i++){
        double interDistance = matches[i].distance;
        if (interDistance < minimalHammingDistance){
            minimalHammingDistance = interDistance;
        }
        if (interDistance > maximalHammingDistance){
            maximalHammingDistance = interDistance;
        }
    }
    cout << "Maximal distance is " << maximalHammingDistance << "." << endl;
    cout << "Minimal distance is " << minimalHammingDistance << "." << endl;
    // using two times minimal Hamming distance to filter large matches
    // or using 30 to filter large matches since sometimes minimal Hamming distance is too small
    vector<DMatch> goodMatches;
    for (int i = 0; i < descriptors1.rows; i++){
        if (matches[i].distance <= 2 * minimalHammingDistance || matches[i].distance <= 30){
            goodMatches.push_back(matches[i]);
        }
    }
    cout << "There are " << goodMatches.size() << " good matches." << endl;
    // draw matches using raw matches and filtered matches, respectively
    Mat matchTwoImages1, matchTwoImages2;
    drawMatches(image1, keyPoints1, image2, keyPoints2, matches, matchTwoImages1);
    imshow("match 2 images by raw matches of ORB features", matchTwoImages1);
    drawMatches(image1, keyPoints1, image2, keyPoints2, goodMatches, matchTwoImages2);
    imshow("matches by filtered match feature points", matchTwoImages2);
    waitKey(0);

    return 0;
}
