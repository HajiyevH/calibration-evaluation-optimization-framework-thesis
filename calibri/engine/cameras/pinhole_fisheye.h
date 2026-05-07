#pragma once
#include <Eigen/Core>
#include "engine/contracts/types.h"
#include <cmath>

namespace ba {

// OpenCV fisheye camera model (OPENCV_FISHEYE in COLMAP)
// Parameters: fx, fy, cx, cy, k1, k2, k3, k4
//
// Distortion model:
//   r = sqrt(x^2 + y^2)
//   theta = atan(r)
//   theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
//   x_d = (theta_d / r) * x
//   y_d = (theta_d / r) * y
//   u = fx * x_d + cx
//   v = fy * y_d + cy

struct PinholeFisheyeIntr {
    double fx, fy, cx, cy;
    double k1, k2, k3, k4;
    int width, height;
};

// Templated projection for Ceres auto-differentiation
// intr[8] = {fx, fy, cx, cy, k1, k2, k3, k4}
// Xc[3] = point in camera frame
// uv_out[2] = projected pixel coordinates
//
// Uses ADL (argument-dependent lookup) for sqrt/atan so that:
//   - double calls std::sqrt/std::atan
//   - ceres::Jet<T> calls ceres::sqrt/ceres::atan (via ADL)
template <typename T>
void projectPinholeFisheyeT(const T* Xc, const T* intr, T* uv_out) {
    using std::sqrt;
    using std::atan;

    const T fx = intr[0], fy = intr[1], cx = intr[2], cy = intr[3];
    const T k1 = intr[4], k2 = intr[5], k3 = intr[6], k4 = intr[7];

    // Normalized image coordinates
    const T x = Xc[0] / Xc[2];
    const T y = Xc[1] / Xc[2];

    // Radius in normalized image plane
    const T r_sq = x * x + y * y;
    const T r = sqrt(r_sq);

    // Angle from optical axis
    const T theta = atan(r);
    const T theta2 = theta * theta;
    const T theta4 = theta2 * theta2;
    const T theta6 = theta4 * theta2;
    const T theta8 = theta4 * theta4;

    // Distorted angle
    const T theta_d = theta * (T(1) + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);

    // Distortion scaling factor: scale = theta_d / r.
    // At r→0: theta = atan(r) → 0, so theta_d → 0 (both zero).
    // By L'Hopital: lim_{r→0}(theta_d / r) = d(theta_d)/d(r)|_{r=0} = 1.
    // We use r_safe = sqrt(r_sq + eps^2) instead of a ternary branch because
    // Ceres Jet types track derivatives continuously — a branch at r=0 would
    // create a gradient discontinuity in the autodiff computation graph.
    const T eps = T(1e-10);
    const T r_safe = sqrt(r_sq + eps * eps);
    const T scale = theta_d / r_safe;

    // Apply distortion
    const T x_d = scale * x;
    const T y_d = scale * y;

    // Project to pixel coordinates
    uv_out[0] = fx * x_d + cx;
    uv_out[1] = fy * y_d + cy;
}

// Non-templated version for testing.
// Returns (1e4, 1e4) for points on or behind the image plane (depth <= 1e-6).
inline Eigen::Vector2d projectPinholeFisheye(const Eigen::Vector3d& Xc, const PinholeFisheyeIntr& K) {
    if (Xc.z() <= 1e-6) {
        return {1e4, 1e4};
    }
    double intr[8] = {K.fx, K.fy, K.cx, K.cy, K.k1, K.k2, K.k3, K.k4};
    double Xc_arr[3] = {Xc.x(), Xc.y(), Xc.z()};
    double uv[2];
    projectPinholeFisheyeT<double>(Xc_arr, intr, uv);
    return Eigen::Vector2d(uv[0], uv[1]);
}

} // namespace ba
