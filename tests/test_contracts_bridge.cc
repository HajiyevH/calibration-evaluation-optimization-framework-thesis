#define _USE_MATH_DEFINES
#include <cmath>
#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <random>
#include "engine/io/contracts_bridge.h"
#include "engine/contracts/types.h"

using namespace ba;

// Known-value round-trips: identity and 90-degree yaw
TEST(ContractsBridge, KnownValueRoundTrips) {
    // --- Identity ---
    {
        PoseSE3 P; // default = identity
        Eigen::Quaterniond q;
        Eigen::Vector3d t;
        PoseSE3_To_qt(P, q, t);

        EXPECT_NEAR(q.w(), 1.0, 1e-6);
        EXPECT_NEAR(q.x(), 0.0, 1e-6);
        EXPECT_NEAR(q.y(), 0.0, 1e-6);
        EXPECT_NEAR(q.z(), 0.0, 1e-6);
        EXPECT_NEAR(t.norm(), 0.0, 1e-6);

        PoseSE3 P2 = qt_To_PoseSE3(q, t);
        Eigen::Quaterniond q2;
        Eigen::Vector3d t2;
        PoseSE3_To_qt(P2, q2, t2);
        EXPECT_NEAR(q2.w(), 1.0, 1e-6);
        EXPECT_NEAR(t2.norm(), 0.0, 1e-6);
    }

    // --- 90-degree yaw ---
    {
        Eigen::Quaterniond q_in(Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitY()));
        Eigen::Vector3d t_in(1.0, 2.0, 3.0);

        PoseSE3 P = qt_To_PoseSE3(q_in, t_in);
        Eigen::Quaterniond q_out;
        Eigen::Vector3d t_out;
        PoseSE3_To_qt(P, q_out, t_out);

        Eigen::Matrix3d R_in = q_in.normalized().toRotationMatrix();
        Eigen::Matrix3d R_out = q_out.normalized().toRotationMatrix();
        EXPECT_LT((R_in - R_out).norm(), 1e-5) << "Rotation matrix round-trip error too large";
        EXPECT_NEAR(t_out.x(), t_in.x(), 1e-5);
        EXPECT_NEAR(t_out.y(), t_in.y(), 1e-5);
        EXPECT_NEAR(t_out.z(), t_in.z(), 1e-5);
    }
}

// 10 random round-trips
TEST(ContractsBridge, RandomRoundTrips) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist_angle(-M_PI, M_PI);
    std::uniform_real_distribution<double> dist_axis(-1.0, 1.0);
    std::uniform_real_distribution<double> dist_t(-10.0, 10.0);

    for (int i = 0; i < 10; ++i) {
        Eigen::Vector3d axis(dist_axis(gen), dist_axis(gen), dist_axis(gen));
        if (axis.norm() < 1e-6) axis = Eigen::Vector3d::UnitZ();
        axis.normalize();
        double angle = dist_angle(gen);
        Eigen::Quaterniond q_in(Eigen::AngleAxisd(angle, axis));
        q_in.normalize();

        Eigen::Vector3d t_in(dist_t(gen), dist_t(gen), dist_t(gen));

        PoseSE3 P = qt_To_PoseSE3(q_in, t_in);
        Eigen::Quaterniond q_out;
        Eigen::Vector3d t_out;
        PoseSE3_To_qt(P, q_out, t_out);

        // Rotation error via angle between quaternions
        // Account for double cover: q and -q represent same rotation
        double dot = std::abs(q_in.dot(q_out));
        double angle_err = 2.0 * std::acos(std::min(dot, 1.0));
        EXPECT_LT(angle_err, 1e-5)
            << "Random round-trip " << i << " rotation error too large: " << angle_err;

        double t_err = (t_in - t_out).norm();
        EXPECT_LT(t_err, 1e-5)
            << "Random round-trip " << i << " translation error too large: " << t_err;
    }
}

// Array-based overload round-trip
TEST(ContractsBridge, ArrayOverloadRoundTrip) {
    Eigen::Quaterniond q_in(Eigen::AngleAxisd(0.3, Eigen::Vector3d(1, 1, 0).normalized()));
    Eigen::Vector3d t_in(5.0, -3.0, 7.0);

    PoseSE3 P = qt_To_PoseSE3(q_in, t_in);

    // Use array overload
    double q_arr[4], t_arr[3];
    PoseSE3_To_qt(P, q_arr, t_arr);

    // Reconstruct from arrays
    PoseSE3 P2 = qt_To_PoseSE3(q_arr, t_arr);

    Eigen::Quaterniond q_final;
    Eigen::Vector3d t_final;
    PoseSE3_To_qt(P2, q_final, t_final);

    Eigen::Matrix3d R_in = q_in.normalized().toRotationMatrix();
    Eigen::Matrix3d R_out = q_final.normalized().toRotationMatrix();
    EXPECT_LT((R_in - R_out).norm(), 1e-5);
    EXPECT_LT((t_in - t_final).norm(), 1e-5);
}

// Quaternion convention: Eigen internal order [x,y,z,w] in array
TEST(ContractsBridge, QuaternionConvention) {
    // 90-degree rotation about Z axis: q = (w=cos(45), x=0, y=0, z=sin(45))
    Eigen::Quaterniond q(Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitZ()));
    Eigen::Vector3d t(0, 0, 0);

    PoseSE3 P = qt_To_PoseSE3(q, t);
    double q_arr[4], t_arr[3];
    PoseSE3_To_qt(P, q_arr, t_arr);

    // Array should be in Eigen internal order: [x, y, z, w]
    // For 90-deg about Z: x=0, y=0, z=sin(45), w=cos(45)
    EXPECT_NEAR(q_arr[0], 0.0, 1e-5);  // x
    EXPECT_NEAR(q_arr[1], 0.0, 1e-5);  // y
    EXPECT_NEAR(q_arr[2], std::sin(M_PI / 4), 1e-5);  // z
    EXPECT_NEAR(q_arr[3], std::cos(M_PI / 4), 1e-5);  // w
}

// Validate float[16] row-major layout of PoseSE3.T after qt_To_PoseSE3.
// Layout: [R|t; 0 0 0 1] where rows are T[0..3], T[4..7], T[8..11], T[12..15].
TEST(ContractsBridge, RowMajorLayoutValidation) {
    // 90-degree rotation about Z-axis:
    //   R = [ 0 -1  0 ]
    //       [ 1  0  0 ]
    //       [ 0  0  1 ]
    Eigen::Quaterniond q(Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitZ()));
    Eigen::Vector3d t(4.0, 5.0, 6.0);

    PoseSE3 P = qt_To_PoseSE3(q, t);

    // Row 0 of R: [0, -1, 0], tx = 4
    EXPECT_NEAR(P.T[0],  0.0, 1e-5);  // R(0,0)
    EXPECT_NEAR(P.T[1], -1.0, 1e-5);  // R(0,1)
    EXPECT_NEAR(P.T[2],  0.0, 1e-5);  // R(0,2)
    EXPECT_NEAR(P.T[3],  4.0, 1e-5);  // tx

    // Row 1 of R: [1, 0, 0], ty = 5
    EXPECT_NEAR(P.T[4],  1.0, 1e-5);  // R(1,0)
    EXPECT_NEAR(P.T[5],  0.0, 1e-5);  // R(1,1)
    EXPECT_NEAR(P.T[6],  0.0, 1e-5);  // R(1,2)
    EXPECT_NEAR(P.T[7],  5.0, 1e-5);  // ty

    // Row 2 of R: [0, 0, 1], tz = 6
    EXPECT_NEAR(P.T[8],  0.0, 1e-5);  // R(2,0)
    EXPECT_NEAR(P.T[9],  0.0, 1e-5);  // R(2,1)
    EXPECT_NEAR(P.T[10], 1.0, 1e-5);  // R(2,2)
    EXPECT_NEAR(P.T[11], 6.0, 1e-5);  // tz

    // Bottom row: [0, 0, 0, 1]
    EXPECT_NEAR(P.T[12], 0.0, 1e-10);
    EXPECT_NEAR(P.T[13], 0.0, 1e-10);
    EXPECT_NEAR(P.T[14], 0.0, 1e-10);
    EXPECT_NEAR(P.T[15], 1.0, 1e-10);
}

