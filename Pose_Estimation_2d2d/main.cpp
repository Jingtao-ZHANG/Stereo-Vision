#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void find_feature_matches(
        const Mat& image1,
        const Mat& image2,
        vector<KeyPoint>& keyPoints1,
        vector<KeyPoint>& keyPoints2,
        vector<DMatch>& matches
        );

void pose_estimation_2d2d(
        vector<KeyPoint>& keyPoints1,
        vector<KeyPoint>& keyPoints2,
        vector<DMatch>& matches,
        Mat& R,
        Mat& t
        );

int main() {

    Mat image1 = imread("../1.png");
    Mat image2 = imread("../2.png");
    vector<KeyPoint> keyPoints1;
    vector<KeyPoint> keyPoints2;
    vector<DMatch> matches;

    find_feature_matches(image1, image2, keyPoints1, keyPoints2, matches);

    return 0;
}

void find_feature_matches(
        const Mat& image1, const Mat& image2,
        vector<KeyPoint>& keyPoints1,
        vector<KeyPoint>& keyPoints2,
        vector<DMatch>& matches
        ){

    Mat descriptor1, descriptor2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    detector -> detect(image1, keyPoints1);
    detector -> detect(image2, keyPoints2);

    descriptor -> compute(image1, keyPoints1, descriptor1);
    descriptor -> compute(image2, keyPoints2, descriptor2);

    vector<DMatch> match;
    matcher -> match(descriptor1, descriptor2, match);

    double maximalHammingDistance = 0.0, minimalHammingDistance = 10000.0;
    for (int i = 0; i < descriptor1.rows; i++){
        double distance = match[i].distance;
        if (distance <= minimalHammingDistance){
            minimalHammingDistance = distance;
        }
        if (distance >= maximalHammingDistance){
            maximalHammingDistance = distance;
        }
    }
    cout << "There are " << match.size() << " matches." << endl;

    for (int i = 0; i < descriptor1.rows; i++){
        if (match[i].distance < 2 * minimalHammingDistance || match[i].distance < 30){
            matches.push_back(match[i]);
        }
    }
    cout << "There are " << matches.size() << " good matches." << endl;

    Mat R, t;
    pose_estimation_2d2d(keyPoints1, keyPoints2, matches, R, t);
}


void pose_estimation_2d2d(
        vector<KeyPoint>& keyPoints1,
        vector<KeyPoint>& keyPoints2,
        vector<DMatch>& matches,
        Mat& R,
        Mat& t
        ){
    // camera reference of TUM freiburg2
    Mat K = (Mat_<double> (3, 3) << 520.9, 0.0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    vector<Point2f> points1;
    vector<Point2f> points2;

    int count = matches.size();
    for (int i = 0; i < count; i++){
        points1.push_back(keyPoints1[matches[i].queryIdx].pt);
        points2.push_back(keyPoints2[matches[i].trainIdx].pt);
    }

    Mat fundamentalMatrix = findFundamentalMat(points1, points2, FM_8POINT);
    cout << "Fundamental matrix is " << endl << fundamentalMatrix << endl;

    Point2d principalPoint (325.1, 249.7);
    double focalLength = 521.0;
    Mat essentialMatrix = findEssentialMat(points1, points2, focalLength, principalPoint);
    cout << "Essential matrix is " << endl << essentialMatrix << endl;

    recoverPose(essentialMatrix, points1, points2, R, t, focalLength, principalPoint);

    cout << "R is " << endl << R << endl;
    cout << "T is " << endl << t << endl;

}