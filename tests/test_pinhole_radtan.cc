#include <gtest/gtest.h>
#include <Eigen/Core>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <random>
#include <cmath>
#include "engine/cameras/pinhole_radtan.h"

using namespace ba;

// Zero distortion: simple pinhole projection
TEST(PinholeRadTan, ZeroDistortion) {
    PinholeIntr K{};
    K.fx = 500.0; K.fy = 500.0;
    K.cx = 320.0; K.cy = 240.0;
    K.k1 = 0; K.k2 = 0; K.p1 = 0; K.p2 = 0; K.k3 = 0;
    K.width = 640; K.height = 480;

    Eigen::Vector3d Xc(1.0, 2.0, 10.0);
    Eigen::Vector2d uv = projectPinholeRadTan(Xc, K);

    double expected_u = K.fx * (Xc.x() / Xc.z()) + K.cx;
    double expected_v = K.fy * (Xc.y() / Xc.z()) + K.cy;

    EXPECT_NEAR(uv.x(), expected_u, 1e-10);
    EXPECT_NEAR(uv.y(), expected_v, 1e-10);
}

// Parity with cv::projectPoints on 100 random points with non-zero distortion
TEST(PinholeRadTan, ParityWithOpenCV) {
    PinholeIntr K{};
    K.fx = 950.0; K.fy = 950.0;
    K.cx = 640.0; K.cy = 400.0;
    K.k1 = -0.1; K.k2 = 0.01; K.p1 = 0.001; K.p2 = -0.0005; K.k3 = 0.002;
    K.width = 1280; K.height = 800;

    // OpenCV camera matrix
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<
        K.fx, 0, K.cx,
        0, K.fy, K.cy,
        0, 0, 1);

    // OpenCV distortion: [k1, k2, p1, p2, k3]
    cv::Mat distCoeffs = (cv::Mat_<double>(5, 1) <<
        K.k1, K.k2, K.p1, K.p2, K.k3);

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
        Eigen::Vector2d uv_ours = projectPinholeRadTan(Xc, K);

        std::vector<cv::Point3d> objPts = {{x, y, z}};
        std::vector<cv::Point2d> imgPts;
        cv::projectPoints(objPts, rvec, tvec, cameraMatrix, distCoeffs, imgPts);

        EXPECT_NEAR(uv_ours.x(), imgPts[0].x, 1e-6)
            << "Point " << i << " u mismatch";
        EXPECT_NEAR(uv_ours.y(), imgPts[0].y, 1e-6)
            << "Point " << i << " v mismatch";
    }
}

// Templated version matches non-templated
TEST(PinholeRadTan, TemplatedMatchesNonTemplated) {
    PinholeIntr K{};
    K.fx = 950.0; K.fy = 950.0;
    K.cx = 640.0; K.cy = 400.0;
    K.k1 = -0.1; K.k2 = 0.01; K.p1 = 0.001; K.p2 = -0.0005; K.k3 = 0.002;

    double intr[9] = {K.fx, K.fy, K.cx, K.cy, K.k1, K.k2, K.p1, K.p2, K.k3};

    std::mt19937 gen(99);
    std::uniform_real_distribution<double> dist_xy(-2.0, 2.0);
    std::uniform_real_distribution<double> dist_z(3.0, 20.0);

    for (int i = 0; i < 50; ++i) {
        Eigen::Vector3d Xc(dist_xy(gen), dist_xy(gen), dist_z(gen));
        Eigen::Vector2d uv_ref = projectPinholeRadTan(Xc, K);

        double Xc_arr[3] = {Xc.x(), Xc.y(), Xc.z()};
        double uv_out[2];
        projectPinholeRadTanT<double>(Xc_arr, intr, uv_out);

        EXPECT_NEAR(uv_out[0], uv_ref.x(), 1e-12);
        EXPECT_NEAR(uv_out[1], uv_ref.y(), 1e-12);
    }
}

// Non-templated projectPinholeRadTan guards Z <= 1e-6, returning (1e4, 1e4).
// The templated version has NO guard — it divides by Z directly.
TEST(PinholeRadTan, BehindCamera) {
    PinholeIntr K{};
    K.fx = 500.0; K.fy = 500.0;
    K.cx = 320.0; K.cy = 240.0;
    K.k1 = 0; K.k2 = 0; K.p1 = 0; K.p2 = 0; K.k3 = 0;
    K.width = 640; K.height = 480;

    // Negative Z: behind camera
    Eigen::Vector2d uv_neg = projectPinholeRadTan({1.0, 2.0, -5.0}, K);
    EXPECT_DOUBLE_EQ(uv_neg.x(), 1e4);
    EXPECT_DOUBLE_EQ(uv_neg.y(), 1e4);

    // Z near zero (below threshold)
    Eigen::Vector2d uv_eps = projectPinholeRadTan({1.0, 2.0, 1e-7}, K);
    EXPECT_DOUBLE_EQ(uv_eps.x(), 1e4);
    EXPECT_DOUBLE_EQ(uv_eps.y(), 1e4);

    // Z exactly at threshold boundary
    Eigen::Vector2d uv_boundary = projectPinholeRadTan({1.0, 2.0, 1e-6}, K);
    EXPECT_DOUBLE_EQ(uv_boundary.x(), 1e4);
    EXPECT_DOUBLE_EQ(uv_boundary.y(), 1e4);

    // Z just above threshold: normal projection
    Eigen::Vector2d uv_above = projectPinholeRadTan({0.0, 0.0, 2e-6}, K);
    EXPECT_NEAR(uv_above.x(), K.cx, 1e-6);
    EXPECT_NEAR(uv_above.y(), K.cy, 1e-6);
}

