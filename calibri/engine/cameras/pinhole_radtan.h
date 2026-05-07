#pragma once
#include <Eigen/Core>
#include "engine/contracts/types.h"

namespace ba {

struct PinholeIntr {
    double fx, fy, cx, cy;
    double k1, k2, p1, p2, k3;
    int width, height;
};

// Convert CameraIntrinsics contract to PinholeIntr
// params[0..4] = k1, k2, p1, p2, k3
PinholeIntr toPinholeIntr(const CameraIntrinsics& cam);

// Non-templated projection for tests and synthetic data generation
Eigen::Vector2d projectPinholeRadTan(const Eigen::Vector3d& Xc, const PinholeIntr& K);

// Templated projection for Ceres auto-differentiation
// intr[9] = {fx, fy, cx, cy, k1, k2, p1, p2, k3}
// Xc[3] = point in camera frame
// uv_out[2] = projected pixel coordinates
template <typename T>
void projectPinholeRadTanT(const T* Xc, const T* intr, T* uv_out) {
    const T x = Xc[0] / Xc[2];
    const T y = Xc[1] / Xc[2];
    const T r2 = x * x + y * y;
    const T r4 = r2 * r2;
    const T r6 = r4 * r2;

    const T fx = intr[0], fy = intr[1], cx = intr[2], cy = intr[3];
    const T k1 = intr[4], k2 = intr[5], p1 = intr[6], p2 = intr[7], k3 = intr[8];

    const T radial = T(1) + k1 * r2 + k2 * r4 + k3 * r6;
    const T x_t = T(2) * p1 * x * y + p2 * (r2 + T(2) * x * x);
    const T y_t = p1 * (r2 + T(2) * y * y) + T(2) * p2 * x * y;

    const T xd = x * radial + x_t;
    const T yd = y * radial + y_t;

    uv_out[0] = fx * xd + cx;
    uv_out[1] = fy * yd + cy;
}

} // namespace ba
