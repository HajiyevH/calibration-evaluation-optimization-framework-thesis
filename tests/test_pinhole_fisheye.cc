#define _USE_MATH_DEFINES
#include <cmath>

#include <gtest/gtest.h>
#include <Eigen/Core>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <random>
#include "engine/cameras/pinhole_fisheye.h"

using namespace ba;

// Zero distortion: simple pinhole projection
TEST(PinholeFisheye, ZeroDistortion) {
    PinholeFisheyeIntr K{};
    K.fx = 500.0; K.fy = 500.0;
    K.cx = 320.0; K.cy = 240.0;
    K.k1 = 0; K.k2 = 0; K.k3 = 0; K.k4 = 0;
    K.width = 640; K.height = 480;

    Eigen::Vector3d Xc(1.0, 2.0, 10.0);
    Eigen::Vector2d uv = projectPinholeFisheye(Xc, K);

    // With zero distortion, theta_d = theta = atan(r), scale = atan(r)/r
    double x = Xc.x() / Xc.z();
    double y = Xc.y() / Xc.z();
    double r = std::sqrt(x * x + y * y);
    double theta = std::atan(r);
    double scale = theta / r;
    double expected_u = K.fx * scale * x + K.cx;
    double expected_v = K.fy * scale * y + K.cy;

    EXPECT_NEAR(uv.x(), expected_u, 1e-10);
    EXPECT_NEAR(uv.y(), expected_v, 1e-10);
}

// Templated version matches non-templated
TEST(PinholeFisheye, TemplatedMatchesNonTemplated) {
    PinholeFisheyeIntr K{};
    K.fx = 800.0; K.fy = 800.0;
    K.cx = 640.0; K.cy = 400.0;
    K.k1 = -0.12; K.k2 = 0.02; K.k3 = -0.005; K.k4 = 0.001;

    double intr[8] = {K.fx, K.fy, K.cx, K.cy, K.k1, K.k2, K.k3, K.k4};

    std::mt19937 gen(99);
    std::uniform_real_distribution<double> dist_xy(-2.0, 2.0);
    std::uniform_real_distribution<double> dist_z(3.0, 20.0);

    for (int i = 0; i < 50; ++i) {
        Eigen::Vector3d Xc(dist_xy(gen), dist_xy(gen), dist_z(gen));
        Eigen::Vector2d uv_ref = projectPinholeFisheye(Xc, K);

        double Xc_arr[3] = {Xc.x(), Xc.y(), Xc.z()};
        double uv_out[2];
        projectPinholeFisheyeT<double>(Xc_arr, intr, uv_out);

        EXPECT_NEAR(uv_out[0], uv_ref.x(), 1e-12);
        EXPECT_NEAR(uv_out[1], uv_ref.y(), 1e-12);
    }
}

// Parity with cv::fisheye::projectPoints on 100 random points with non-zero distortion
TEST(PinholeFisheye, ParityWithOpenCV) {
    PinholeFisheyeIntr K{};
    K.fx = 350.0; K.fy = 350.0;
    K.cx = 640.0; K.cy = 400.0;
    K.k1 = -0.08; K.k2 = 0.03; K.k3 = -0.005; K.k4 = 0.001;
    K.width = 1280; K.height = 800;

    // OpenCV 3x3 camera matrix
    cv::Mat camMatrix = (cv::Mat_<double>(3, 3) <<
        K.fx, 0.0, K.cx,
        0.0,  K.fy, K.cy,
        0.0,  0.0,  1.0);

    // OpenCV fisheye distortion: 4x1 [k1, k2, k3, k4]
    cv::Mat distCoeffs = (cv::Mat_<double>(4, 1) <<
        K.k1, K.k2, K.k3, K.k4);

    // Identity pose: points are already in camera frame
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);

    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist_xy(-3.0, 3.0);
    std::uniform_real_distribution<double> dist_z(2.0, 30.0);

    for (int i = 0; i < 100; ++i) {
        double x = dist_xy(gen);
        double y = dist_xy(gen);
        double z = dist_z(gen);

        Eigen::Vector3d Xc(x, y, z);
        Eigen::Vector2d uv_ours = projectPinholeFisheye(Xc, K);

        std::vector<cv::Point3d> objPts = {{x, y, z}};
        std::vector<cv::Point2d> imgPts;
        cv::fisheye::projectPoints(objPts, imgPts, rvec, tvec, camMatrix, distCoeffs);

        EXPECT_NEAR(uv_ours.x(), imgPts[0].x, 1e-6)
            << "Point " << i << " u mismatch: Xc=(" << x << "," << y << "," << z << ")";
        EXPECT_NEAR(uv_ours.y(), imgPts[0].y, 1e-6)
            << "Point " << i << " v mismatch: Xc=(" << x << "," << y << "," << z << ")";
    }
}

// Behind-camera guard: projectPinholeFisheye returns (1e4, 1e4) when z <= 1e-6
TEST(PinholeFisheye, BehindCameraGuard) {
    PinholeFisheyeIntr K{};
    K.fx = 500.0; K.fy = 500.0;
    K.cx = 320.0; K.cy = 240.0;
    K.k1 = -0.1; K.k2 = 0.01; K.k3 = 0.0; K.k4 = 0.0;
    K.width = 640; K.height = 480;

    const double sentinel = 1e4;

    // z = 0: exactly on the image plane
    {
        Eigen::Vector2d uv = projectPinholeFisheye({1.0, 2.0, 0.0}, K);
        EXPECT_DOUBLE_EQ(uv.x(), sentinel);
        EXPECT_DOUBLE_EQ(uv.y(), sentinel);
    }

    // z = -1: behind the camera
    {
        Eigen::Vector2d uv = projectPinholeFisheye({0.5, -0.3, -1.0}, K);
        EXPECT_DOUBLE_EQ(uv.x(), sentinel);
        EXPECT_DOUBLE_EQ(uv.y(), sentinel);
    }

    // z = 1e-7: below the 1e-6 threshold
    {
        Eigen::Vector2d uv = projectPinholeFisheye({0.0, 0.0, 1e-7}, K);
        EXPECT_DOUBLE_EQ(uv.x(), sentinel);
        EXPECT_DOUBLE_EQ(uv.y(), sentinel);
    }

    // z = 1e-5: above the threshold, should NOT return sentinel
    {
        Eigen::Vector2d uv = projectPinholeFisheye({0.0, 0.0, 1e-5}, K);
        // On-axis point should project to principal point, not sentinel
        EXPECT_NEAR(uv.x(), K.cx, 1e-3);
        EXPECT_NEAR(uv.y(), K.cy, 1e-3);
    }
}

