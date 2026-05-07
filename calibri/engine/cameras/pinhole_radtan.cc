#include "cameras/pinhole_radtan.h"

namespace ba {

PinholeIntr toPinholeIntr(const CameraIntrinsics& cam) {
    PinholeIntr K{};
    K.fx = static_cast<double>(cam.fx);
    K.fy = static_cast<double>(cam.fy);
    K.cx = static_cast<double>(cam.cx);
    K.cy = static_cast<double>(cam.cy);
    // params[0..4] = k1, k2, p1, p2, k3
    K.k1 = static_cast<double>(cam.params[0]);
    K.k2 = static_cast<double>(cam.params[1]);
    K.p1 = static_cast<double>(cam.params[2]);
    K.p2 = static_cast<double>(cam.params[3]);
    K.k3 = static_cast<double>(cam.params[4]);
    K.width = static_cast<int>(cam.image_width);
    K.height = static_cast<int>(cam.image_height);
    return K;
}

Eigen::Vector2d projectPinholeRadTan(const Eigen::Vector3d& Xc, const PinholeIntr& K) {
    // Depth guard: avoid division by zero for points on or behind the image plane.
    // Returns a large off-image value so callers can detect the failure.
    if (Xc.z() <= 1e-6) {
        return {1e4, 1e4};
    }
    const double x = Xc.x() / Xc.z();
    const double y = Xc.y() / Xc.z();
    const double r2 = x * x + y * y;
    const double r4 = r2 * r2;
    const double r6 = r4 * r2;

    const double radial = 1.0 + K.k1 * r2 + K.k2 * r4 + K.k3 * r6;
    const double x_t = 2.0 * K.p1 * x * y + K.p2 * (r2 + 2.0 * x * x);
    const double y_t = K.p1 * (r2 + 2.0 * y * y) + 2.0 * K.p2 * x * y;

    const double xd = x * radial + x_t;
    const double yd = y * radial + y_t;

    return {K.fx * xd + K.cx, K.fy * yd + K.cy};
}

} // namespace ba
