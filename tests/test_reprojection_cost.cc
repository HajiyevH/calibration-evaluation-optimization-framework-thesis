#define _USE_MATH_DEFINES
#include <cmath>
#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "engine/cameras/pinhole_radtan.h"
#include "engine/cost/reprojection_cost.h"
#include "engine/io/contracts_bridge.h"

using namespace ba;

// Regression test: verify pose convention consistency.
// Compute the residual two ways to ensure q_wv, t_wv are correctly
// treated as vehicle-to-world throughout the projection chain.
TEST(ReprojectionCost, PoseConventionConsistency) {
    // Setup known transform: vehicle-to-world
    Eigen::Quaterniond q_wv(Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitY()));
    Eigen::Vector3d t_wv(1.0, 0.5, 2.0);

    // Camera extrinsic: vehicle-to-camera (identity)
    Eigen::Quaterniond q_vc = Eigen::Quaterniond::Identity();
    Eigen::Vector3d t_vc(0.0, 0.0, 0.0);

    Eigen::Vector3d Xw(3.0, 1.0, 10.0);

    PinholeIntr K;
    K.fx = 500.0; K.fy = 500.0; K.cx = 320.0; K.cy = 240.0;
    K.k1 = 0.0; K.k2 = 0.0; K.p1 = 0.0; K.p2 = 0.0; K.k3 = 0.0;

    // Method 1: explicit inverse
    Eigen::Quaterniond q_vw = q_wv.conjugate();
    Eigen::Vector3d t_vw = -(q_vw * t_wv);
    Eigen::Vector3d X_vehicle_direct = q_vw * Xw + t_vw;
    Eigen::Vector3d X_camera_direct = q_vc * X_vehicle_direct + t_vc;
    Eigen::Vector2d uv_direct = projectPinholeRadTan(X_camera_direct, K);

    // Method 2: cost functor logic
    Eigen::Vector3d X_vehicle_functor = q_wv.conjugate() * (Xw - t_wv);
    Eigen::Vector3d X_camera_functor = q_vc * X_vehicle_functor + t_vc;
    Eigen::Vector2d uv_functor = projectPinholeRadTan(X_camera_functor, K);

    EXPECT_NEAR(X_camera_direct.x(), X_camera_functor.x(), 1e-10);
    EXPECT_NEAR(X_camera_direct.y(), X_camera_functor.y(), 1e-10);
    EXPECT_NEAR(X_camera_direct.z(), X_camera_functor.z(), 1e-10);

    EXPECT_NEAR(uv_direct.x(), uv_functor.x(), 1e-8);
    EXPECT_NEAR(uv_direct.y(), uv_functor.y(), 1e-8);

    // Method 3: via actual Ceres cost functor
    double q_wv_arr[4] = {q_wv.x(), q_wv.y(), q_wv.z(), q_wv.w()};
    double t_wv_arr[3] = {t_wv.x(), t_wv.y(), t_wv.z()};
    double q_vc_arr[4] = {q_vc.x(), q_vc.y(), q_vc.z(), q_vc.w()};
    double t_vc_arr[3] = {t_vc.x(), t_vc.y(), t_vc.z()};
    double intr_arr[9] = {K.fx, K.fy, K.cx, K.cy, K.k1, K.k2, K.p1, K.p2, K.k3};
    double Xw_arr[3] = {Xw.x(), Xw.y(), Xw.z()};

    ReprojectionCostPinhole cost(uv_functor);
    double residuals[2];
    cost(q_wv_arr, t_wv_arr, q_vc_arr, t_vc_arr, intr_arr, Xw_arr, residuals);

    EXPECT_NEAR(residuals[0], 0.0, 1e-8)
        << "Ceres functor residual[0] should be zero when measured == predicted";
    EXPECT_NEAR(residuals[1], 0.0, 1e-8)
        << "Ceres functor residual[1] should be zero when measured == predicted";
}

// Test non-identity extrinsic through the cost functor
TEST(ReprojectionCost, NonIdentityExtrinsic) {
    Eigen::Quaterniond q_wv = Eigen::Quaterniond::Identity();
    Eigen::Vector3d t_wv(0.0, 0.0, 0.0);

    // 90-degree yaw extrinsic
    Eigen::Quaterniond q_vc(Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitY()));
    Eigen::Vector3d t_vc(0.5, 0.0, 0.0);

    // World point to the left of the vehicle
    Eigen::Vector3d Xw(-5.0, 0.0, 0.0);

    PinholeIntr K;
    K.fx = 500.0; K.fy = 500.0; K.cx = 320.0; K.cy = 240.0;
    K.k1 = 0.0; K.k2 = 0.0; K.p1 = 0.0; K.p2 = 0.0; K.k3 = 0.0;

    // Manual computation
    Eigen::Vector3d X_vehicle = q_wv.conjugate() * (Xw - t_wv);
    Eigen::Vector3d X_camera = q_vc * X_vehicle + t_vc;

    // The 90-deg yaw rotates vehicle-frame (-5,0,0) to camera-frame (0,0,5)
    // Then t_vc adds (0.5,0,0) => (0.5, 0, 5)
    EXPECT_NEAR(X_camera.x(), 0.5, 1e-8);
    EXPECT_NEAR(X_camera.y(), 0.0, 1e-8);
    EXPECT_NEAR(X_camera.z(), 5.0, 1e-8);

    Eigen::Vector2d uv = projectPinholeRadTan(X_camera, K);

    double q_wv_arr[4] = {q_wv.x(), q_wv.y(), q_wv.z(), q_wv.w()};
    double t_wv_arr[3] = {t_wv.x(), t_wv.y(), t_wv.z()};
    double q_vc_arr[4] = {q_vc.x(), q_vc.y(), q_vc.z(), q_vc.w()};
    double t_vc_arr[3] = {t_vc.x(), t_vc.y(), t_vc.z()};
    double intr_arr[9] = {K.fx, K.fy, K.cx, K.cy, K.k1, K.k2, K.p1, K.p2, K.k3};
    double Xw_arr[3] = {Xw.x(), Xw.y(), Xw.z()};

    ReprojectionCostPinhole cost(uv);
    double residuals[2];
    cost(q_wv_arr, t_wv_arr, q_vc_arr, t_vc_arr, intr_arr, Xw_arr, residuals);

    EXPECT_NEAR(residuals[0], 0.0, 1e-8);
    EXPECT_NEAR(residuals[1], 0.0, 1e-8);
}

// Behind-camera guard: residuals should be large, not NaN/inf
TEST(ReprojectionCost, BehindCameraGuardProducesLargeResiduals) {
    Eigen::Quaterniond q_wv = Eigen::Quaterniond::Identity();
    Eigen::Vector3d t_wv(0.0, 0.0, 0.0);
    Eigen::Quaterniond q_vc = Eigen::Quaterniond::Identity();
    Eigen::Vector3d t_vc(0.0, 0.0, 0.0);

    // Point behind camera
    Eigen::Vector3d Xw(0.0, 0.0, -5.0);
    Eigen::Vector2d uv_meas(320.0, 240.0);

    double q_wv_arr[4] = {q_wv.x(), q_wv.y(), q_wv.z(), q_wv.w()};
    double t_wv_arr[3] = {t_wv.x(), t_wv.y(), t_wv.z()};
    double q_vc_arr[4] = {q_vc.x(), q_vc.y(), q_vc.z(), q_vc.w()};
    double t_vc_arr[3] = {t_vc.x(), t_vc.y(), t_vc.z()};
    double intr_arr[9] = {500.0, 500.0, 320.0, 240.0, 0, 0, 0, 0, 0};
    double Xw_arr[3] = {Xw.x(), Xw.y(), Xw.z()};

    ReprojectionCostPinhole cost(uv_meas);
    double residuals[2];
    bool ok = cost(q_wv_arr, t_wv_arr, q_vc_arr, t_vc_arr, intr_arr, Xw_arr, residuals);

    EXPECT_TRUE(ok);
    EXPECT_FALSE(std::isnan(residuals[0]));
    EXPECT_FALSE(std::isnan(residuals[1]));
    EXPECT_GT(std::abs(residuals[0]), 100.0) << "Behind-camera should produce large residuals";
    EXPECT_GT(std::abs(residuals[1]), 100.0) << "Behind-camera should produce large residuals";
}
