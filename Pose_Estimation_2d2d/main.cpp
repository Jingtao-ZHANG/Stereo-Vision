#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// This function is to find feature matches between two images.
void find_feature_matches(
        const Mat& image1,
        const Mat& image2,
        vector<KeyPoint>& keyPoints1,
        vector<KeyPoint>& keyPoints2,
        vector<DMatch>& matches);

// This function is to estimate pose from 2D to 2D.
void pose_estimation_2d2d(
        vector<KeyPoint>& keyPoints1,
        vector<KeyPoint>& keyPoints2,
        vector<DMatch>& matches,
        Mat& R,
        Mat& t);

// This function transform
Point2d pixel2cam (const Point2d& p, const Mat& K);


/* ----------------------- MAIN ------------------------*/
int main() {
    // input two images
    Mat image1 = imread("../1.png");
    Mat image2 = imread("../2.png");

    // initialize
    vector<KeyPoint> keyPoints1;
    vector<KeyPoint> keyPoints2;
    vector<DMatch> matches;

    // call "find_feature_matches" function to calculate features and matches.
    find_feature_matches(image1, image2, keyPoints1, keyPoints2, matches);

    // call "pose_estimation_2d2d" function to estimate pose.
    Mat R, t;
    pose_estimation_2d2d(keyPoints1, keyPoints2, matches, R, t);

    Mat t_x = ( Mat_<double> (3, 3) <<
                0,                             -t.at<double> (2, 0),   t.at<double> (1, 0),
                t.at<double> (2, 0),     0,                            -t.at<double> (0, 0),
                -t.at<double> (1, 0),    t.at<double> (0, 0),    0
                );
    Mat verifiedEssentialMatrix = t_x * R;
    cout << "Verified essential matrix is " << endl << verifiedEssentialMatrix << "." << endl << endl;

    // Define camera reference.
    Mat K = (Mat_<double> (3, 3) <<
             520.9, 0,     325.1,
             0,     521.0, 249.7,
             0,     0,     1
             );
    // Verify epipolar constraint.
    for (DMatch m: matches){
        Point2d pt1 = pixel2cam(keyPoints1[m.queryIdx].pt, K);
        Mat y1 = (Mat_<double> (3, 1) << pt1.x, pt1.y, 1);
        Point2d pt2 = pixel2cam(keyPoints2[m.trainIdx].pt, K);
        Mat y2 = (Mat_<double> (3, 1) << pt2.x, pt2.y, 1);
        Mat d = y2.t() * t_x * R * y1;
        cout << "Epipolar constraint = " << d << endl;
    }

    return 0;
}


/* -----------------------Below is the function.------------------------*/


void find_feature_matches(
        const Mat& image1, const Mat& image2,
        vector<KeyPoint>& keyPoints1,
        vector<KeyPoint>& keyPoints2,
        vector<DMatch>& matches
        ){

    // Initialize ORB feature detector, ORB feature descriptor, and feature matcher.
    Mat descriptor1, descriptor2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    // Detect ORB features in two images.
    detector -> detect(image1, keyPoints1);
    detector -> detect(image2, keyPoints2);

    // Compute descriptors of ORB features.
    descriptor -> compute(image1, keyPoints1, descriptor1);
    descriptor -> compute(image2, keyPoints2, descriptor2);

    // Match features by descriptors using Hamming distance.
    vector<DMatch> match;
    matcher -> match(descriptor1, descriptor2, match);
    cout << "There are " << match.size() << " matches." << endl;

    // Calculate minimal and maximal Hamming distance of the matches.
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

    // Use 2 times minimal Hamming distance or 30 as threshold to filter matches.
    for (int i = 0; i < descriptor1.rows; i++){
        if (match[i].distance < 2 * minimalHammingDistance || match[i].distance < 30){
            matches.push_back(match[i]);
        }
    }
    cout << "There are " << matches.size() << " good matches." << endl << endl;

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

    // Initialize coordinate point format.
    vector<Point2f> points1;
    vector<Point2f> points2;

    // Transform ORB feature key points to image coordinate point format.
    int count = matches.size();
    for (int i = 0; i < count; i++){
        points1.push_back(keyPoints1[matches[i].queryIdx].pt);
        points2.push_back(keyPoints2[matches[i].trainIdx].pt);
    }

    // Calculate fundamental matrix.
    Mat fundamentalMatrix = findFundamentalMat(points1, points2, FM_8POINT);
    cout << "Fundamental matrix is " << endl << fundamentalMatrix << endl << endl;

    // Calculate essential matrix.
    Point2d principalPoint (325.1, 249.7);
    double focalLength = 521.0;
    Mat essentialMatrix = findEssentialMat(points1, points2, focalLength, principalPoint);
    cout << "Essential matrix is " << endl << essentialMatrix << endl << endl;

    // Recovery pose from essential matrix.
    recoverPose(essentialMatrix, points1, points2, R, t, focalLength, principalPoint);
    cout << "R is " << endl << R << endl << endl;
    cout << "T is " << endl << t << endl << endl;
}


Point2d pixel2cam (const Point2d& p, const Mat& K){
     Point2d coordinate
            (
                (p.x - K.at<double> (0, 2)) / K.at<double> (0, 0),
                (p.y - K.at<double> (1, 2)) / K.at<double> (1, 1)
            );
     return coordinate;
}