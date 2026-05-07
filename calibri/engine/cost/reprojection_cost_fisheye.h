#pragma once
#include <Eigen/Core>
#include <ceres/ceres.h>
#include "engine/cameras/pinhole_fisheye.h"

namespace ba {

// Reprojection cost functor for fisheye camera model
//
// Pose convention (same as radtan):
//   VehiclePose.pose = vehicle-to-world (T_wv)
//   CameraExtrinsics.T_vehicle_to_camera = vehicle-to-camera (T_vc)
//
// Projection chain:
//   X_vehicle = R_wv^(-1) * (X_world - t_wv)
//   X_camera  = R_vc * X_vehicle + t_vc

struct ReprojectionCostFisheye {
    Eigen::Vector2d uv_meas_;

    explicit ReprojectionCostFisheye(const Eigen::Vector2d& uv_meas)
        : uv_meas_(uv_meas) {}

    template <typename T>
    bool operator()(const T* const q_wv_ptr,   // [x,y,z,w] - vehicle-to-world rotation
                    const T* const t_wv_ptr,   // vehicle-to-world translation
                    const T* const q_vc_ptr,   // [x,y,z,w] - vehicle-to-camera rotation
                    const T* const t_vc_ptr,   // vehicle-to-camera translation
                    const T* const intr,       // [fx,fy,cx,cy,k1,k2,k3,k4]
                    const T* const Xw_ptr,     // 3D landmark in world
                    T* residuals) const {
        // Map quaternions using Eigen internal order [x,y,z,w]
        Eigen::Map<const Eigen::Quaternion<T>> q_wv(q_wv_ptr);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_wv(t_wv_ptr);
        Eigen::Map<const Eigen::Quaternion<T>> q_vc(q_vc_ptr);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_vc(t_vc_ptr);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> Xw(Xw_ptr);

        // Transform world point to vehicle frame:
        // X_vehicle = R_wv^(-1) * (X_world - t_wv)
        Eigen::Matrix<T, 3, 1> X_vehicle = q_wv.conjugate() * (Xw - t_wv);

        // Transform vehicle point to camera frame:
        // X_camera = R_vc * X_vehicle + t_vc
        Eigen::Matrix<T, 3, 1> X_camera = q_vc * X_vehicle + t_vc;

        // Behind-camera guard: penalty with nonzero gradient so the solver
        // gets a signal to push the point forward (z increasing).
        // Base penalty of 1e3 is ~500x a typical inlier residual (~2 px),
        // ensuring Ceres treats behind-camera points as clearly worse than
        // any in-front-of-camera observation. The gradient d(penalty)/dz = -1e4
        // provides a strong, constant push toward positive z.
        if (X_camera.z() <= T(1e-6)) {
            T penalty = T(1e3) + T(1e4) * (T(1e-6) - X_camera.z());
            residuals[0] = penalty;
            residuals[1] = penalty;
            return true;
        }

        // Project using fisheye model
        T Xc_arr[3] = {X_camera.x(), X_camera.y(), X_camera.z()};
        T uv_pred[2];
        projectPinholeFisheyeT<T>(Xc_arr, intr, uv_pred);

        residuals[0] = T(uv_meas_.x()) - uv_pred[0];
        residuals[1] = T(uv_meas_.y()) - uv_pred[1];
        return true;
    }

    // Factory: creates Ceres AutoDiff cost function with 6 parameter blocks
    static ceres::CostFunction* Create(const Eigen::Vector2d& uv) {
        return new ceres::AutoDiffCostFunction<ReprojectionCostFisheye,
                                               2,    // residuals
                                               4,    // q_wv
                                               3,    // t_wv
                                               4,    // q_vc
                                               3,    // t_vc
                                               8,    // intrinsics [fx,fy,cx,cy,k1,k2,k3,k4]
                                               3>(   // Xw
            new ReprojectionCostFisheye(uv));
    }
};

} // namespace ba
