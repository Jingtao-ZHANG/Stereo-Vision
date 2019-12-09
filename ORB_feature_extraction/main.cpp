#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main( int argc, char** argv ) {

    Mat image1 = imread("../1.png");
    Mat image2 = imread("../2.png");

    vector<KeyPoint> keyPoints1, keyPoints2;
    Mat descriptors1, descriptors2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create( "BruteForce-Hamming" );

    detector->detect(image1, keyPoints1);
    detector->detect(image2, keyPoints2);

    descriptor->compute(image1, keyPoints1, descriptors1);
    descriptor->compute(image2, keyPoints2, descriptors2);

    Mat detectedImage1;
    drawKeypoints(image1, keyPoints1, detectedImage1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    imshow("ORB features", detectedImage1);
    waitKey(0);

    vector<DMatch> matches;
    matcher->match( descriptors1, descriptors2, matches);

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

    cout << "maximal distance is " << maximalHammingDistance << endl;
    cout << "minimal distance is " << minimalHammingDistance << endl;

    Mat matchTwoImages;
    drawMatches(image1, keyPoints1, image2, keyPoints2, matches, matchTwoImages);

    imshow("match 2 images by raw matches of ORB features", matchTwoImages);

    waitKey(0);

    cout << "Hello, World!" << endl;
    return 0;

}
