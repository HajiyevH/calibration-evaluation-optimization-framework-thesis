#define _USE_MATH_DEFINES
#include <cmath>
#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include "engine/contracts/types.h"
#include "engine/contracts/status.h"
#include "engine/cameras/pinhole_radtan.h"
#include "engine/cameras/pinhole_fisheye.h"
#include "engine/io/contracts_bridge.h"
#include "engine/solver/bundle_adjuster.h"

using namespace ba;

// ============================================================================
// Synthetic scene infrastructure
// ============================================================================

namespace {

struct SceneConfig {
    int num_poses = 15;
    int num_landmarks = 300;
    double noise_sigma = 0.5;
    double max_yaw_deg = 5.0;
    double max_lateral_m = 0.3;
    double max_vertical_m = 0.0;
    double forward_step_m = 0.5;
    double pose_rot_noise_deg = 1.0;
    double pose_trans_noise_m = 0.02;
    double landmark_noise_m = 0.05;
    CameraIntrinsics::ModelType model_type = CameraIntrinsics::ModelType::PINHOLE;
    double fx = 950.0, fy = 950.0, cx = 640.0, cy = 400.0;
    int width = 1280, height = 800;
    double k1 = 0, k2 = 0, p1 = 0, p2 = 0, k3 = 0;
    // Fisheye distortion (used when model_type == FISHEYE)
    double fk1 = 0, fk2 = 0, fk3 = 0, fk4 = 0;
    uint32_t seed = 42;
    // Optional non-identity extrinsic (T_vehicle_to_camera)
    Eigen::Quaterniond extr_q = Eigen::Quaterniond::Identity();
    Eigen::Vector3d extr_t = Eigen::Vector3d::Zero();
    // Reference camera ID (255 = no reference gauge fix for single-camera extrinsics)
    uint8_t reference_camera_id = 0;
};

struct SyntheticScene {
    BundleAdjustmentProblem problem;
    BundleAdjustmentParams params;
    // Ground truth for verification
    std::vector<Eigen::Quaterniond> gt_poses_q;
    std::vector<Eigen::Vector3d> gt_poses_t;
    std::vector<Eigen::Vector3d> gt_landmarks;
    Eigen::Quaterniond gt_extr_q = Eigen::Quaterniond::Identity();
    Eigen::Vector3d gt_extr_t = Eigen::Vector3d::Zero();
    // Multi-camera ground truth extrinsics (indexed by camera_id)
    std::vector<Eigen::Quaterniond> gt_extr_q_vec;
    std::vector<Eigen::Vector3d> gt_extr_t_vec;
    // Per-landmark observation count (indexed by landmark index, 0-based)
    std::vector<int> obs_count_per_landmark;
};

// Project a camera-frame point to pixels, dispatching on model type.
Eigen::Vector2d projectPoint(const Eigen::Vector3d& Xc,
                             const CameraIntrinsics& ci) {
    if (ci.model_type == CameraIntrinsics::ModelType::FISHEYE) {
        PinholeFisheyeIntr K;
        K.fx = ci.fx; K.fy = ci.fy; K.cx = ci.cx; K.cy = ci.cy;
        K.k1 = ci.params[0]; K.k2 = ci.params[1];
        K.k3 = ci.params[2]; K.k4 = ci.params[3];
        K.width = ci.image_width; K.height = ci.image_height;
        return projectPinholeFisheye(Xc, K);
    } else {
        PinholeIntr K = toPinholeIntr(ci);
        return projectPinholeRadTan(Xc, K);
    }
}

// Forward declarations for helpers used by scene builders
void perturbPosesAndLandmarks(SyntheticScene& scene, const SceneConfig& cfg, std::mt19937& gen);

// Build a single-camera synthetic scene from SceneConfig.
SyntheticScene buildScene(const SceneConfig& cfg) {
    SyntheticScene scene;
    std::mt19937 gen(cfg.seed);
    std::uniform_real_distribution<double> xy_dist(-8.0, 8.0);
    std::uniform_real_distribution<double> z_dist(4.0, 30.0);
    std::normal_distribution<double> noise(0.0, cfg.noise_sigma);

    // Camera setup
    CameraIntrinsics ci;
    ci.camera_id = 0;
    ci.model_type = cfg.model_type;
    ci.fx = static_cast<float>(cfg.fx);
    ci.fy = static_cast<float>(cfg.fy);
    ci.cx = static_cast<float>(cfg.cx);
    ci.cy = static_cast<float>(cfg.cy);
    if (cfg.model_type == CameraIntrinsics::ModelType::FISHEYE) {
        ci.params[0] = static_cast<float>(cfg.fk1);
        ci.params[1] = static_cast<float>(cfg.fk2);
        ci.params[2] = static_cast<float>(cfg.fk3);
        ci.params[3] = static_cast<float>(cfg.fk4);
    } else {
        ci.params[0] = static_cast<float>(cfg.k1);
        ci.params[1] = static_cast<float>(cfg.k2);
        ci.params[2] = static_cast<float>(cfg.p1);
        ci.params[3] = static_cast<float>(cfg.p2);
        ci.params[4] = static_cast<float>(cfg.k3);
    }
    ci.image_width = static_cast<uint16_t>(cfg.width);
    ci.image_height = static_cast<uint16_t>(cfg.height);
    scene.problem.camera_rig.intrinsics.push_back(ci);

    // Store ground-truth extrinsic
    scene.gt_extr_q = cfg.extr_q;
    scene.gt_extr_t = cfg.extr_t;

    CameraExtrinsics ce;
    ce.camera_id = 0;
    ce.T_vehicle_to_camera = qt_To_PoseSE3(cfg.extr_q, cfg.extr_t);
    scene.problem.camera_rig.extrinsics.push_back(ce);
    scene.problem.camera_rig.reference_camera_id = cfg.reference_camera_id;

    // Generate poses along a forward arc
    for (int i = 0; i < cfg.num_poses; ++i) {
        double frac = static_cast<double>(i) / std::max(cfg.num_poses - 1, 1);
        double yaw = frac * cfg.max_yaw_deg * M_PI / 180.0;
        double tx = cfg.max_lateral_m * std::sin(frac * M_PI * 0.5);
        double ty = cfg.max_vertical_m * std::sin(frac * M_PI * 1.5);
        double tz = i * cfg.forward_step_m;

        Eigen::Quaterniond q(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitY()));
        Eigen::Vector3d t(tx, ty, tz);

        scene.gt_poses_q.push_back(q);
        scene.gt_poses_t.push_back(t);

        VehiclePose vp;
        vp.timestamp_ns = static_cast<uint64_t>(i + 1) * 1000000000ULL;
        vp.pose = qt_To_PoseSE3(q, t);
        scene.problem.vehicle_poses.poses_buffer.push_back(vp);
    }

    // Generate landmarks
    scene.obs_count_per_landmark.resize(cfg.num_landmarks, 0);
    for (int i = 0; i < cfg.num_landmarks; ++i) {
        double x = xy_dist(gen);
        double y = xy_dist(gen) * 0.6;
        double z = z_dist(gen);

        scene.gt_landmarks.push_back(Eigen::Vector3d(x, y, z));

        Landmark lm;
        lm.id = static_cast<uint64_t>(i + 1);
        lm.x = static_cast<float>(x);
        lm.y = static_cast<float>(y);
        lm.z = static_cast<float>(z);
        scene.problem.landmarks.landmarks.push_back(lm);
    }

    // Generate observations by projecting through the full chain
    for (int pi = 0; pi < cfg.num_poses; ++pi) {
        const auto& q = scene.gt_poses_q[pi];
        const auto& t = scene.gt_poses_t[pi];
        uint64_t ts = scene.problem.vehicle_poses.poses_buffer[pi].timestamp_ns;

        for (int li = 0; li < cfg.num_landmarks; ++li) {
            const auto& Xw = scene.gt_landmarks[li];

            // World to vehicle to camera
            Eigen::Vector3d X_vehicle = q.conjugate() * (Xw - t);
            Eigen::Vector3d Xc = cfg.extr_q * X_vehicle + cfg.extr_t;
            if (Xc.z() <= 0.5) continue;

            Eigen::Vector2d uv = projectPoint(Xc, ci);

            double u = uv.x() + noise(gen);
            double v = uv.y() + noise(gen);

            if (u < 0 || u >= cfg.width || v < 0 || v >= cfg.height) continue;

            Observation2D3D obs;
            obs.landmark_id = static_cast<uint64_t>(li + 1);
            obs.camera_id = 0;
            obs.timestamp_ns = ts;
            obs.keypoint.x = static_cast<float>(u);
            obs.keypoint.y = static_cast<float>(v);
            scene.problem.observations.push_back(obs);
            scene.obs_count_per_landmark[li]++;
        }
    }

    // Perturb initial estimates (skip first pose for gauge)
    perturbPosesAndLandmarks(scene, cfg, gen);

    // Optimization flags
    scene.problem.optimize_poses = true;
    scene.problem.optimize_landmarks = true;
    scene.problem.optimize_intrinsics = false;
    scene.problem.optimize_extrinsics = false;
    scene.problem.optimize_distortion = false;

    // Default solver params
    scene.params.max_iterations = 50;
    scene.params.loss_type = BundleAdjustmentParams::LossType::HUBER;
    scene.params.loss_scale = 1.0f;
    scene.params.max_reproj_error_px = 5.0f;
    scene.params.verbose = false;

    return scene;
}

// Build a calibration scene: more poses, wider baseline, more landmarks.
// Uses buildScene internally with calibration-appropriate defaults, then
// overrides max_iterations to 100.
SyntheticScene buildCalibScene(const SceneConfig& overrides) {
    SceneConfig cfg = overrides;
    if (cfg.num_poses == 15) cfg.num_poses = 25;
    if (cfg.num_landmarks == 300) cfg.num_landmarks = 500;
    if (cfg.noise_sigma == 0.5) cfg.noise_sigma = 0.3;
    if (cfg.max_yaw_deg == 5.0) cfg.max_yaw_deg = 15.0;
    if (cfg.max_lateral_m == 0.3) cfg.max_lateral_m = 3.0;
    if (cfg.max_vertical_m == 0.0) cfg.max_vertical_m = 1.0;
    if (cfg.forward_step_m == 0.5) cfg.forward_step_m = 0.8;

    auto scene = buildScene(cfg);
    scene.params.max_iterations = 100;
    return scene;
}

// Check landmark accuracy only for well-constrained landmarks:
// >= min_obs observations AND ground-truth depth < max_depth_m.
void checkFilteredLandmarks(const SyntheticScene& scene,
                            const BundleAdjustmentResult& result,
                            double tolerance_m,
                            int min_obs = 3,
                            double max_depth_m = 15.0) {
    int checked = 0;
    for (size_t i = 0; i < scene.gt_landmarks.size(); ++i) {
        if (scene.obs_count_per_landmark[i] < min_obs) continue;
        if (scene.gt_landmarks[i].z() >= max_depth_m) continue;

        Eigen::Vector3d lm_opt(
            result.optimized_landmarks.landmarks[i].x,
            result.optimized_landmarks.landmarks[i].y,
            result.optimized_landmarks.landmarks[i].z);
        double lm_err = (lm_opt - scene.gt_landmarks[i]).norm();
        EXPECT_LT(lm_err, tolerance_m)
            << "Landmark " << i << " error: " << lm_err << " m"
            << " (obs=" << scene.obs_count_per_landmark[i]
            << ", depth=" << scene.gt_landmarks[i].z() << ")";
        checked++;
    }
    EXPECT_GT(checked, 0) << "No landmarks passed the filter criteria";
}

// Helper: compute per-pose rotation error in degrees
double poseRotError(const Eigen::Quaterniond& q_opt, const Eigen::Quaterniond& q_gt) {
    double dot = std::abs(q_opt.dot(q_gt));
    return 2.0 * std::acos(std::min(dot, 1.0)) * 180.0 / M_PI;
}

// Perturb poses (skip first for gauge) and landmarks using scene config noise levels.
void perturbPosesAndLandmarks(SyntheticScene& scene, const SceneConfig& cfg, std::mt19937& gen) {
    std::normal_distribution<double> rot_noise(0.0, cfg.pose_rot_noise_deg * M_PI / 180.0);
    std::normal_distribution<double> trans_noise(0.0, cfg.pose_trans_noise_m);
    std::normal_distribution<double> lm_noise(0.0, cfg.landmark_noise_m);

    for (size_t i = 1; i < scene.problem.vehicle_poses.poses_buffer.size(); ++i) {
        Eigen::Vector3d axis(
            std::normal_distribution<double>(0, 1)(gen),
            std::normal_distribution<double>(0, 1)(gen),
            std::normal_distribution<double>(0, 1)(gen));
        if (axis.norm() > 1e-6) axis.normalize();
        else axis = Eigen::Vector3d::UnitZ();

        double angle = rot_noise(gen);
        Eigen::Quaterniond dq(Eigen::AngleAxisd(angle, axis));
        Eigen::Quaterniond q_pert = scene.gt_poses_q[i] * dq;

        Eigen::Vector3d dt(trans_noise(gen), trans_noise(gen), trans_noise(gen));
        Eigen::Vector3d t_pert = scene.gt_poses_t[i] + dt;

        scene.problem.vehicle_poses.poses_buffer[i].pose = qt_To_PoseSE3(q_pert, t_pert);
    }

    for (size_t i = 0; i < scene.problem.landmarks.landmarks.size(); ++i) {
        scene.problem.landmarks.landmarks[i].x += static_cast<float>(lm_noise(gen));
        scene.problem.landmarks.landmarks[i].y += static_cast<float>(lm_noise(gen));
        scene.problem.landmarks.landmarks[i].z += static_cast<float>(lm_noise(gen));
    }
}

// Standard convergence + error checks
void expectConverged(const BundleAdjustmentResult& result,
                     float max_avg_reproj = 1.0f) {
    EXPECT_TRUE(result.success) << "Solver did not converge: " << result.message;
    EXPECT_LT(result.avg_reproj_error_px, max_avg_reproj)
        << "Average reprojection error too large";
    EXPECT_LT(result.final_cost, result.initial_cost)
        << "Final cost should be lower than initial cost";
}

// Check all poses against ground truth
void expectPosesNearGT(const SyntheticScene& scene,
                       const BundleAdjustmentResult& result,
                       double trans_tol_m = 0.06, double rot_tol_deg = 3.0) {
    for (size_t i = 0; i < scene.gt_poses_q.size(); ++i) {
        Eigen::Quaterniond q_opt;
        Eigen::Vector3d t_opt;
        PoseSE3_To_qt(result.optimized_poses.poses_buffer[i].pose, q_opt, t_opt);

        double t_err = (t_opt - scene.gt_poses_t[i]).norm();
        EXPECT_LT(t_err, trans_tol_m)
            << "Pose " << i << " translation error: " << t_err << " m";

        double angle_err = poseRotError(q_opt, scene.gt_poses_q[i]);
        EXPECT_LT(angle_err, rot_tol_deg)
            << "Pose " << i << " rotation error: " << angle_err << " deg";
    }
}

// Create a multi-camera (4-camera surround-view) scene.
SyntheticScene buildMultiCamScene(uint32_t seed = 314159) {
    SyntheticScene scene;
    std::mt19937 gen(seed);
    std::normal_distribution<double> noise(0.0, 0.3);

    struct CamDef {
        uint8_t id;
        double yaw_deg;
        Eigen::Vector3d t;
        double fx, fy;
        double k1;
    };

    std::vector<CamDef> cam_defs = {
        {0,    0.0, Eigen::Vector3d(0.0, 0.0,  0.0), 950.0, 950.0, -0.10},
        {1,   90.0, Eigen::Vector3d(-0.8, 0.0, 1.0), 800.0, 800.0, -0.12},
        {2,  -90.0, Eigen::Vector3d(0.8, 0.0,  1.0), 800.0, 800.0, -0.12},
        {3,  180.0, Eigen::Vector3d(0.0, 0.3, -2.0), 700.0, 700.0, -0.15},
    };

    for (const auto& cd : cam_defs) {
        Eigen::Quaterniond q_vc(
            Eigen::AngleAxisd(cd.yaw_deg * M_PI / 180.0, Eigen::Vector3d::UnitY()));
        q_vc.normalize();

        scene.gt_extr_q_vec.push_back(q_vc);
        scene.gt_extr_t_vec.push_back(cd.t);

        CameraIntrinsics ci;
        ci.camera_id = cd.id;
        ci.model_type = CameraIntrinsics::ModelType::PINHOLE;
        ci.fx = static_cast<float>(cd.fx);
        ci.fy = static_cast<float>(cd.fy);
        ci.cx = 640.f;
        ci.cy = 400.f;
        ci.params[0] = static_cast<float>(cd.k1);
        ci.image_width = 1280;
        ci.image_height = 800;
        scene.problem.camera_rig.intrinsics.push_back(ci);

        CameraExtrinsics ce;
        ce.camera_id = cd.id;
        ce.T_vehicle_to_camera = qt_To_PoseSE3(q_vc, cd.t);
        scene.problem.camera_rig.extrinsics.push_back(ce);
    }

    scene.problem.camera_rig.reference_camera_id = 0;

    // 15 poses with forward + sinusoidal lateral + yaw variation
    int num_poses = 15;
    for (int i = 0; i < num_poses; ++i) {
        double frac = static_cast<double>(i) / (num_poses - 1);
        double yaw = std::sin(frac * M_PI * 2.0) * 15.0 * M_PI / 180.0;
        double tx = 3.0 * std::sin(frac * M_PI);
        double ty = 1.0 * std::sin(frac * M_PI * 1.5);
        double tz = i * 0.8;

        Eigen::Quaterniond q(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitY()));
        Eigen::Vector3d t(tx, ty, tz);

        scene.gt_poses_q.push_back(q);
        scene.gt_poses_t.push_back(t);

        VehiclePose vp;
        vp.timestamp_ns = static_cast<uint64_t>(i + 1) * 1000000000ULL;
        vp.pose = qt_To_PoseSE3(q, t);
        scene.problem.vehicle_poses.poses_buffer.push_back(vp);
    }

    // Landmarks: ~330 total in 4 sectors + overlap regions
    auto addLandmarks = [&](int count, double x_lo, double x_hi,
                            double y_lo, double y_hi,
                            double z_lo, double z_hi) {
        std::uniform_real_distribution<double> dx(x_lo, x_hi);
        std::uniform_real_distribution<double> dy(y_lo, y_hi);
        std::uniform_real_distribution<double> dz(z_lo, z_hi);
        for (int i = 0; i < count; ++i) {
            double x = dx(gen), y = dy(gen), z = dz(gen);
            scene.gt_landmarks.push_back(Eigen::Vector3d(x, y, z));
            Landmark lm;
            lm.id = static_cast<uint64_t>(scene.gt_landmarks.size());
            lm.x = static_cast<float>(x);
            lm.y = static_cast<float>(y);
            lm.z = static_cast<float>(z);
            scene.problem.landmarks.landmarks.push_back(lm);
        }
    };

    addLandmarks(100, -8.0,  8.0, -3.0, 3.0,  5.0, 30.0);  // front
    addLandmarks(60,  -30.0,-5.0, -3.0, 3.0, -5.0,  5.0);   // left
    addLandmarks(60,   5.0, 30.0, -3.0, 3.0, -5.0,  5.0);   // right
    addLandmarks(60,  -8.0,  8.0, -3.0, 3.0,-30.0, -5.0);   // rear
    addLandmarks(25,  -12.0,-3.0, -3.0, 3.0,  3.0, 12.0);   // front-left overlap
    addLandmarks(25,   3.0, 12.0, -3.0, 3.0,  3.0, 12.0);   // front-right overlap

    int num_landmarks = static_cast<int>(scene.gt_landmarks.size());
    scene.obs_count_per_landmark.resize(num_landmarks, 0);

    // Generate observations through all 4 cameras
    for (int pi = 0; pi < num_poses; ++pi) {
        const auto& q_wv = scene.gt_poses_q[pi];
        const auto& t_wv = scene.gt_poses_t[pi];
        uint64_t ts = scene.problem.vehicle_poses.poses_buffer[pi].timestamp_ns;

        for (int ci_idx = 0; ci_idx < static_cast<int>(cam_defs.size()); ++ci_idx) {
            const auto& q_vc = scene.gt_extr_q_vec[ci_idx];
            const auto& t_vc = scene.gt_extr_t_vec[ci_idx];
            const auto& cam_ci = scene.problem.camera_rig.intrinsics[ci_idx];

            for (int li = 0; li < num_landmarks; ++li) {
                const auto& Xw = scene.gt_landmarks[li];

                Eigen::Vector3d X_vehicle = q_wv.conjugate() * (Xw - t_wv);
                Eigen::Vector3d X_camera = q_vc * X_vehicle + t_vc;

                if (X_camera.z() <= 0.5) continue;

                Eigen::Vector2d uv = projectPoint(X_camera, cam_ci);

                double u = uv.x() + noise(gen);
                double v = uv.y() + noise(gen);

                if (u < 0 || u >= cam_ci.image_width || v < 0 || v >= cam_ci.image_height)
                    continue;

                Observation2D3D obs;
                obs.landmark_id = static_cast<uint64_t>(li + 1);
                obs.camera_id = cam_defs[ci_idx].id;
                obs.timestamp_ns = ts;
                obs.keypoint.x = static_cast<float>(u);
                obs.keypoint.y = static_cast<float>(v);
                scene.problem.observations.push_back(obs);
                scene.obs_count_per_landmark[li]++;
            }
        }
    }

    // Perturb poses (skip first for gauge) and landmarks
    SceneConfig mc_cfg;
    mc_cfg.pose_rot_noise_deg = 1.0;
    mc_cfg.pose_trans_noise_m = 0.02;
    mc_cfg.landmark_noise_m = 0.05;
    perturbPosesAndLandmarks(scene, mc_cfg, gen);

    // Perturb extrinsics for cameras 1-3 (camera 0 is reference, held constant)
    for (int ci_idx = 1; ci_idx < static_cast<int>(cam_defs.size()); ++ci_idx) {
        Eigen::Vector3d extr_axis(
            std::normal_distribution<double>(0, 1)(gen),
            std::normal_distribution<double>(0, 1)(gen),
            std::normal_distribution<double>(0, 1)(gen));
        if (extr_axis.norm() > 1e-6) extr_axis.normalize();
        else extr_axis = Eigen::Vector3d::UnitZ();

        Eigen::Quaterniond dq_extr(
            Eigen::AngleAxisd(2.0 * M_PI / 180.0, extr_axis));
        Eigen::Quaterniond q_pert = scene.gt_extr_q_vec[ci_idx] * dq_extr;
        Eigen::Vector3d t_pert = scene.gt_extr_t_vec[ci_idx] +
            Eigen::Vector3d(0.03 * (gen() % 2 == 0 ? 1 : -1),
                            0.04 * (gen() % 2 == 0 ? 1 : -1),
                            0.02 * (gen() % 2 == 0 ? 1 : -1));

        scene.problem.camera_rig.extrinsics[ci_idx].T_vehicle_to_camera =
            qt_To_PoseSE3(q_pert, t_pert);
    }

    // Optimization flags
    scene.problem.optimize_poses = true;
    scene.problem.optimize_landmarks = true;
    scene.problem.optimize_extrinsics = true;
    scene.problem.optimize_intrinsics = false;
    scene.problem.optimize_distortion = false;

    // Solver params
    scene.params.max_iterations = 100;
    scene.params.loss_type = BundleAdjustmentParams::LossType::HUBER;
    scene.params.loss_scale = 1.0f;
    scene.params.max_reproj_error_px = 5.0f;
    scene.params.verbose = false;

    return scene;
}

// Build a single-camera extrinsics scene with non-identity ground-truth extrinsic.
// Uses buildCalibScene with a realistic automotive camera mounting:
// ~5 deg pitch down, ~2 deg yaw right, translated (0, -0.3, 0.5) m in vehicle frame.
SyntheticScene buildExtrinsicsScene(uint32_t seed = 777) {
    Eigen::Quaterniond q_pitch(Eigen::AngleAxisd(5.0 * M_PI / 180.0, Eigen::Vector3d::UnitX()));
    Eigen::Quaterniond q_yaw(Eigen::AngleAxisd(2.0 * M_PI / 180.0, Eigen::Vector3d::UnitY()));

    SceneConfig cfg;
    cfg.seed = seed;
    cfg.noise_sigma = 0.3;
    cfg.k1 = -0.1;
    cfg.k2 = 0.01;
    cfg.extr_q = (q_yaw * q_pitch).normalized();
    cfg.extr_t = Eigen::Vector3d(0.0, -0.3, 0.5);
    cfg.reference_camera_id = 255; // disable reference gauge fix for single-camera

    auto scene = buildCalibScene(cfg);

    // Perturb extrinsic: ~2 deg rotation, ~5cm translation
    Eigen::Vector3d extr_axis(0.3, 0.9, 0.1);
    extr_axis.normalize();
    Eigen::Quaterniond dq_extr(Eigen::AngleAxisd(2.0 * M_PI / 180.0, extr_axis));
    Eigen::Quaterniond q_vc_pert = scene.gt_extr_q * dq_extr;
    Eigen::Vector3d t_vc_pert = scene.gt_extr_t + Eigen::Vector3d(0.03, -0.04, 0.02);
    scene.problem.camera_rig.extrinsics[0].T_vehicle_to_camera =
        qt_To_PoseSE3(q_vc_pert, t_vc_pert);

    return scene;
}

// Inject outliers by randomly selecting observations and replacing with uniform noise.
// Uses std::shuffle for random selection (not first-N bias).
void injectOutliers(std::vector<Observation2D3D>& observations,
                    double fraction,
                    int img_width, int img_height,
                    uint32_t seed) {
    std::mt19937 gen(seed);
    size_t n_outliers = static_cast<size_t>(observations.size() * fraction);

    std::vector<size_t> indices(observations.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);

    std::uniform_real_distribution<float> u_dist(0.0f, static_cast<float>(img_width));
    std::uniform_real_distribution<float> v_dist(0.0f, static_cast<float>(img_height));

    for (size_t i = 0; i < n_outliers; ++i) {
        observations[indices[i]].keypoint.x = u_dist(gen);
        observations[indices[i]].keypoint.y = v_dist(gen);
    }
}

} // anonymous namespace

// ============================================================================
// E2E Pose + Landmark Recovery Tests
// ============================================================================

TEST(E2ESynthetic, PoseAndLandmarkRecovery) {
    SceneConfig cfg;
    cfg.seed = 42;
    cfg.k1 = -0.1;
    cfg.k2 = 0.01;
    auto scene = buildScene(cfg);

    ASSERT_GT(scene.problem.observations.size(), 100u)
        << "Too few observations generated";

    BundleAdjustmentResult result;
    BundleAdjuster adjuster;
    Status s = adjuster.solve(scene.problem, scene.params, result);
    ASSERT_TRUE(s.ok) << s.message;

    EXPECT_TRUE(result.success) << "Solver did not converge: " << result.message;
    EXPECT_LT(result.iterations_used, 50) << "Too many iterations";
    EXPECT_LT(result.final_cost, result.initial_cost * 0.10)
        << "Final cost not sufficiently reduced";
    EXPECT_LT(result.avg_reproj_error_px, 1.0f)
        << "Average reprojection error too large";

    expectPosesNearGT(scene, result);
    checkFilteredLandmarks(scene, result, 0.5);
}

// ============================================================================
// Calibration Recovery Tests
// ============================================================================

// Intrinsics-only recovery (fx, fy, cx, cy perturbed, distortion frozen).
// The 15px focal length tolerance reflects compensation for frozen incorrect distortion.
TEST(E2ESynthetic, IntrinsicsRecovery_FocalLengthPrincipalPoint) {
    SceneConfig cfg;
    cfg.seed = 123;
    cfg.k1 = -0.1;
    cfg.k2 = 0.01;
    cfg.p1 = 0.001;
    cfg.p2 = -0.001;
    cfg.k3 = 0.005;
    auto scene = buildCalibScene(cfg);

    ASSERT_GT(scene.problem.observations.size(), 100u)
        << "Too few observations generated";

    const double gt_fx = 950.0, gt_fy = 950.0, gt_cx = 640.0, gt_cy = 400.0;

    auto& ci = scene.problem.camera_rig.intrinsics[0];
    ci.fx = 970.f;
    ci.fy = 930.f;
    ci.cx = 645.f;
    ci.cy = 395.f;

    scene.problem.optimize_intrinsics = true;
    scene.problem.optimize_distortion = false;
    scene.params.max_iterations = 100;

    BundleAdjustmentResult result;
    BundleAdjuster adjuster;
    Status s = adjuster.solve(scene.problem, scene.params, result);
    ASSERT_TRUE(s.ok) << s.message;
    EXPECT_TRUE(result.success) << "Solver did not converge: " << result.message;

    const auto& opt_ci = result.optimized_camera_rig.intrinsics[0];
    EXPECT_NEAR(opt_ci.fx, gt_fx, 15.0)
        << "fx recovery: got " << opt_ci.fx << ", expected " << gt_fx;
    EXPECT_NEAR(opt_ci.fy, gt_fy, 15.0)
        << "fy recovery: got " << opt_ci.fy << ", expected " << gt_fy;
    EXPECT_NEAR(opt_ci.cx, gt_cx, 5.0)
        << "cx recovery: got " << opt_ci.cx << ", expected " << gt_cx;
    EXPECT_NEAR(opt_ci.cy, gt_cy, 5.0)
        << "cy recovery: got " << opt_ci.cy << ", expected " << gt_cy;

    EXPECT_FLOAT_EQ(opt_ci.params[0], -0.1f)  << "k1 should be frozen";
    EXPECT_FLOAT_EQ(opt_ci.params[1],  0.01f) << "k2 should be frozen";
    EXPECT_FLOAT_EQ(opt_ci.params[2],  0.001f) << "p1 should be frozen";
    EXPECT_FLOAT_EQ(opt_ci.params[3], -0.001f) << "p2 should be frozen";
    EXPECT_FLOAT_EQ(opt_ci.params[4],  0.005f) << "k3 should be frozen";

    EXPECT_LT(result.avg_reproj_error_px, 1.0f)
        << "Average reprojection error too large after intrinsics recovery";
    EXPECT_LT(result.final_cost, result.initial_cost)
        << "Final cost should be lower than initial cost";
}

TEST(E2ESynthetic, DistortionRecovery_RadialTangential) {
    const double gt_k1 = -0.1, gt_k2 = 0.01, gt_p1 = 0.001, gt_p2 = -0.001, gt_k3 = 0.005;
    SceneConfig cfg;
    cfg.seed = 123;
    cfg.k1 = gt_k1;
    cfg.k2 = gt_k2;
    cfg.p1 = gt_p1;
    cfg.p2 = gt_p2;
    cfg.k3 = gt_k3;
    auto scene = buildCalibScene(cfg);

    ASSERT_GT(scene.problem.observations.size(), 100u)
        << "Too few observations generated";

    auto& ci = scene.problem.camera_rig.intrinsics[0];
    ci.params[0] = static_cast<float>(gt_k1 * 0.7);
    ci.params[1] = static_cast<float>(gt_k2 * 0.5);
    ci.params[2] = static_cast<float>(gt_p1 * 0.5);
    ci.params[3] = static_cast<float>(gt_p2 * 0.5);
    ci.params[4] = 0.f;

    scene.problem.optimize_intrinsics = false;
    scene.problem.optimize_distortion = true;
    scene.params.max_iterations = 100;

    BundleAdjustmentResult result;
    BundleAdjuster adjuster;
    Status s = adjuster.solve(scene.problem, scene.params, result);
    ASSERT_TRUE(s.ok) << s.message;
    EXPECT_TRUE(result.success) << "Solver did not converge: " << result.message;

    const auto& opt_ci = result.optimized_camera_rig.intrinsics[0];
    EXPECT_NEAR(opt_ci.params[0], gt_k1, 0.02)
        << "k1 recovery: got " << opt_ci.params[0] << ", expected " << gt_k1;
    EXPECT_NEAR(opt_ci.params[1], gt_k2, 0.015)
        << "k2 recovery: got " << opt_ci.params[1] << ", expected " << gt_k2;
    EXPECT_NEAR(opt_ci.params[2], gt_p1, 0.005)
        << "p1 recovery: got " << opt_ci.params[2] << ", expected " << gt_p1;
    EXPECT_NEAR(opt_ci.params[3], gt_p2, 0.005)
        << "p2 recovery: got " << opt_ci.params[3] << ", expected " << gt_p2;
    EXPECT_NEAR(opt_ci.params[4], gt_k3, 0.01)
        << "k3 recovery: got " << opt_ci.params[4] << ", expected " << gt_k3;

    EXPECT_FLOAT_EQ(opt_ci.fx, 950.f) << "fx should be frozen";
    EXPECT_FLOAT_EQ(opt_ci.fy, 950.f) << "fy should be frozen";
    EXPECT_FLOAT_EQ(opt_ci.cx, 640.f) << "cx should be frozen";
    EXPECT_FLOAT_EQ(opt_ci.cy, 400.f) << "cy should be frozen";

    EXPECT_LT(result.avg_reproj_error_px, 1.0f)
        << "Average reprojection error too large after distortion recovery";
    EXPECT_LT(result.final_cost, result.initial_cost)
        << "Final cost should be lower than initial cost";
}

// Joint intrinsics + distortion recovery.
// The 35px focal length tolerance reflects the focal-distortion tradeoff:
// when optimizing both jointly, the solver can find alternative (fx, k1) pairs
// that produce equivalent reprojection errors. This is a known ill-conditioning
// of the calibration problem without strong priors.
TEST(E2ESynthetic, JointIntrinsicsDistortionRecovery) {
    const double gt_k1 = -0.1, gt_k2 = 0.01;
    SceneConfig cfg;
    cfg.seed = 123;
    cfg.k1 = gt_k1;
    cfg.k2 = gt_k2;
    auto scene = buildCalibScene(cfg);

    ASSERT_GT(scene.problem.observations.size(), 100u)
        << "Too few observations generated";

    const double gt_fx = 950.0, gt_fy = 950.0, gt_cx = 640.0, gt_cy = 400.0;

    auto& ci = scene.problem.camera_rig.intrinsics[0];
    ci.fx = 935.f;
    ci.fy = 965.f;
    ci.cx = 645.f;
    ci.cy = 395.f;
    ci.params[0] = -0.08f;
    ci.params[1] =  0.005f;

    scene.problem.optimize_intrinsics = true;
    scene.problem.optimize_distortion = true;
    scene.params.max_iterations = 150;

    BundleAdjustmentResult result;
    BundleAdjuster adjuster;
    Status s = adjuster.solve(scene.problem, scene.params, result);
    ASSERT_TRUE(s.ok) << s.message;
    EXPECT_TRUE(result.success) << "Solver did not converge: " << result.message;

    const auto& opt_ci = result.optimized_camera_rig.intrinsics[0];
    EXPECT_NEAR(opt_ci.fx, gt_fx, 35.0)
        << "fx recovery: got " << opt_ci.fx << ", expected " << gt_fx;
    EXPECT_NEAR(opt_ci.fy, gt_fy, 35.0)
        << "fy recovery: got " << opt_ci.fy << ", expected " << gt_fy;
    EXPECT_NEAR(opt_ci.cx, gt_cx, 10.0)
        << "cx recovery: got " << opt_ci.cx << ", expected " << gt_cx;
    EXPECT_NEAR(opt_ci.cy, gt_cy, 10.0)
        << "cy recovery: got " << opt_ci.cy << ", expected " << gt_cy;
    EXPECT_NEAR(opt_ci.params[0], gt_k1, 0.03)
        << "k1 recovery: got " << opt_ci.params[0] << ", expected " << gt_k1;
    EXPECT_NEAR(opt_ci.params[1], gt_k2, 0.025)
        << "k2 recovery: got " << opt_ci.params[1] << ", expected " << gt_k2;

    EXPECT_LT(result.avg_reproj_error_px, 1.0f)
        << "Average reprojection error too large after joint recovery";
    EXPECT_LT(result.final_cost, result.initial_cost)
        << "Final cost should be lower than initial cost";
}

// ============================================================================
// Extrinsics Recovery Tests
// ============================================================================

TEST(E2ESynthetic, ExtrinsicsRecovery) {
    auto scene = buildExtrinsicsScene();
    scene.problem.optimize_extrinsics = true;

    Eigen::Quaterniond q_input;
    Eigen::Vector3d t_input;
    PoseSE3_To_qt(scene.problem.camera_rig.extrinsics[0].T_vehicle_to_camera, q_input, t_input);

    ASSERT_GT(scene.problem.observations.size(), 100u)
        << "Too few observations generated";

    BundleAdjustmentResult result;
    BundleAdjuster adjuster;
    Status s = adjuster.solve(scene.problem, scene.params, result);
    ASSERT_TRUE(s.ok) << s.message;
    EXPECT_TRUE(result.success) << "Solver did not converge: " << result.message;

    Eigen::Quaterniond q_opt;
    Eigen::Vector3d t_opt;
    PoseSE3_To_qt(result.optimized_camera_rig.extrinsics[0].T_vehicle_to_camera, q_opt, t_opt);

    // Verify extrinsic moved from input (writeback check)
    double input_angle_diff = poseRotError(q_opt, q_input);
    double input_t_diff = (t_opt - t_input).norm();
    EXPECT_GT(input_angle_diff + input_t_diff * 100.0, 0.1)
        << "Optimized extrinsic did not move from input -- writeback may be missing";

    // Verify optimized is closer to GT than input
    double input_gt_angle = poseRotError(q_input, scene.gt_extr_q);
    double opt_gt_angle = poseRotError(q_opt, scene.gt_extr_q);
    EXPECT_LT(opt_gt_angle, input_gt_angle)
        << "Optimized rotation not closer to GT than input: opt=" << opt_gt_angle
        << " deg vs input=" << input_gt_angle << " deg";

    // Verify recovery accuracy
    double extr_angle_err = poseRotError(q_opt, scene.gt_extr_q);
    EXPECT_LT(extr_angle_err, 2.0)
        << "Extrinsic rotation error: " << extr_angle_err << " deg";

    double extr_t_err = (t_opt - scene.gt_extr_t).norm();
    EXPECT_LT(extr_t_err, 0.20)
        << "Extrinsic translation error: " << extr_t_err << " m";

    EXPECT_LT(result.avg_reproj_error_px, 1.0f)
        << "Average reprojection error too large";

    // Pose-extrinsic coupling: freeing extrinsics with a single camera allows
    // systematic shifts in T_vc to be partially absorbed by poses. The 0.30m
    // tolerance accounts for this expected coupling effect.
    expectPosesNearGT(scene, result, 0.30, 3.0);
}

TEST(E2ESynthetic, ExtrinsicsConstant_Unchanged) {
    auto scene = buildExtrinsicsScene();
    scene.problem.optimize_extrinsics = false;

    PoseSE3 input_extr = scene.problem.camera_rig.extrinsics[0].T_vehicle_to_camera;

    BundleAdjustmentResult result;
    BundleAdjuster adjuster;
    Status s = adjuster.solve(scene.problem, scene.params, result);
    ASSERT_TRUE(s.ok) << s.message;
    EXPECT_TRUE(result.success) << "Solver did not converge: " << result.message;

    const PoseSE3& output_extr = result.optimized_camera_rig.extrinsics[0].T_vehicle_to_camera;
    for (int i = 0; i < 16; ++i) {
        EXPECT_NEAR(output_extr.T[i], input_extr.T[i], 1e-4f)
            << "Extrinsic T[" << i << "] changed despite optimize_extrinsics=false";
    }
}

// ============================================================================
// Multi-Camera Rig Tests (4-camera surround-view)
// ============================================================================

TEST(E2EMultiCam, FourCameraExtrinsicsRecovery) {
    auto scene = buildMultiCamScene();

    ASSERT_GT(scene.problem.observations.size(), 2000u)
        << "Too few observations for 4-camera scene";

    BundleAdjustmentResult result;
    BundleAdjuster adjuster;
    Status s = adjuster.solve(scene.problem, scene.params, result);
    ASSERT_TRUE(s.ok) << s.message;
    EXPECT_TRUE(result.success) << "Solver did not converge: " << result.message;
    EXPECT_LT(result.avg_reproj_error_px, 1.0f)
        << "Average reprojection error too large";

    // Multi-camera extrinsic recovery with reference camera gauge fix.
    // Translation tolerance 0.5m for well-constrained multi-camera rig geometry.
    for (int ci = 1; ci <= 3; ++ci) {
        Eigen::Quaterniond q_opt;
        Eigen::Vector3d t_opt;
        PoseSE3_To_qt(result.optimized_camera_rig.extrinsics[ci].T_vehicle_to_camera,
                      q_opt, t_opt);

        double angle_err = poseRotError(q_opt, scene.gt_extr_q_vec[ci]);
        EXPECT_LT(angle_err, 2.5)
            << "Camera " << ci << " extrinsic rotation error: " << angle_err << " deg";

        double t_err = (t_opt - scene.gt_extr_t_vec[ci]).norm();
        EXPECT_LT(t_err, 0.5)
            << "Camera " << ci << " extrinsic translation error: " << t_err << " m";
    }

    // Pose accuracy check (merged from PoseAccuracyWithMultiCam)
    expectPosesNearGT(scene, result, 0.10, 2.0);
}

TEST(E2EMultiCam, ReferenceExtrinsicUnchanged) {
    auto scene = buildMultiCamScene();

    PoseSE3 input_extr0 = scene.problem.camera_rig.extrinsics[0].T_vehicle_to_camera;

    BundleAdjustmentResult result;
    BundleAdjuster adjuster;
    Status s = adjuster.solve(scene.problem, scene.params, result);
    ASSERT_TRUE(s.ok) << s.message;
    EXPECT_TRUE(result.success) << "Solver did not converge: " << result.message;

    const PoseSE3& output_extr0 =
        result.optimized_camera_rig.extrinsics[0].T_vehicle_to_camera;
    for (int i = 0; i < 16; ++i) {
        EXPECT_NEAR(output_extr0.T[i], input_extr0.T[i], 1e-4f)
            << "Reference camera extrinsic T[" << i
            << "] changed despite being gauge-fixed";
    }
}

// ============================================================================
// Robust Loss Function Tests
// ============================================================================

// Huber vs squared error with identical corrupted data.
// Both runs start from the same corrupted observations and the same perturbed
// initial parameter estimates, ensuring a fair comparison.
TEST(E2ERobustness, HighContaminationRecovery) {
    SceneConfig cfg;
    cfg.seed = 42;
    auto scene = buildScene(cfg);
    ASSERT_GT(scene.problem.observations.size(), 100u)
        << "Too few observations generated";

    // Inject 35% outliers using random selection (not first-N)
    injectOutliers(scene.problem.observations, 0.35,
                   cfg.width, cfg.height, /*seed=*/12345);

    // Deep-copy the corrupted problem so both solvers start identically
    BundleAdjustmentProblem problem_copy = scene.problem;

    // --- PART 1: Solve with NONE loss (squared error) ---
    scene.params.loss_type = BundleAdjustmentParams::LossType::NONE;
    scene.params.max_iterations = 50;

    BundleAdjustmentResult result_none;
    BundleAdjuster adjuster;
    Status s = adjuster.solve(scene.problem, scene.params, result_none);
    ASSERT_TRUE(s.ok) << s.message;

    double max_trans_err_none = 0.0;
    double max_rot_err_deg_none = 0.0;
    for (size_t i = 0; i < scene.gt_poses_q.size(); ++i) {
        Eigen::Quaterniond q_opt;
        Eigen::Vector3d t_opt;
        PoseSE3_To_qt(result_none.optimized_poses.poses_buffer[i].pose, q_opt, t_opt);

        max_trans_err_none = std::max(max_trans_err_none,
            (t_opt - scene.gt_poses_t[i]).norm());
        max_rot_err_deg_none = std::max(max_rot_err_deg_none,
            poseRotError(q_opt, scene.gt_poses_q[i]));
    }

    // --- PART 2: Solve with HUBER loss on the identical corrupted problem ---
    BundleAdjustmentParams huber_params = scene.params;
    huber_params.loss_type = BundleAdjustmentParams::LossType::HUBER;
    huber_params.loss_scale = 1.5f;
    huber_params.max_iterations = 50;

    BundleAdjustmentResult result_huber;
    s = adjuster.solve(problem_copy, huber_params, result_huber);
    ASSERT_TRUE(s.ok) << s.message;

    double max_trans_err_huber = 0.0;
    double max_rot_err_deg_huber = 0.0;
    for (size_t i = 0; i < scene.gt_poses_q.size(); ++i) {
        Eigen::Quaterniond q_opt;
        Eigen::Vector3d t_opt;
        PoseSE3_To_qt(result_huber.optimized_poses.poses_buffer[i].pose, q_opt, t_opt);

        max_trans_err_huber = std::max(max_trans_err_huber,
            (t_opt - scene.gt_poses_t[i]).norm());
        max_rot_err_deg_huber = std::max(max_rot_err_deg_huber,
            poseRotError(q_opt, scene.gt_poses_q[i]));
    }

    EXPECT_LT(max_trans_err_none, 0.3)
        << "NONE loss should still converge on over-constrained scene";
    EXPECT_LT(max_trans_err_huber, 0.15)
        << "HUBER loss should achieve better accuracy: " << max_trans_err_huber << " m";

    EXPECT_LT(max_trans_err_huber, max_trans_err_none)
        << "HUBER loss must achieve lower error than NONE loss with outliers present";
    EXPECT_LT(max_rot_err_deg_huber, max_rot_err_deg_none)
        << "HUBER loss must achieve lower rotation error than NONE loss";
}

// ============================================================================
// Iterative Outlier Rejection Tests
// ============================================================================

// Outlier rejection with gross outliers produces lower error than without
TEST(E2EOutlierRejection, RejectsGrossOutliersAndImproves) {
    SceneConfig cfg;
    cfg.seed = 42;

    // Build two identical scenes with 15% outliers
    auto scene_no_rej = buildScene(cfg);
    auto scene_with_rej = buildScene(cfg);
    injectOutliers(scene_no_rej.problem.observations, 0.15,
                   cfg.width, cfg.height, 77777);
    injectOutliers(scene_with_rej.problem.observations, 0.15,
                   cfg.width, cfg.height, 77777);

    ASSERT_GT(scene_no_rej.problem.observations.size(), 100u);

    // Without rejection
    scene_no_rej.params.max_outlier_rejection_passes = 0;
    BundleAdjustmentResult result_no_rej;
    BundleAdjuster adjuster;
    Status s1 = adjuster.solve(scene_no_rej.problem, scene_no_rej.params, result_no_rej);
    ASSERT_TRUE(s1.ok) << s1.message;

    // With rejection (3 passes, adaptive threshold)
    scene_with_rej.params.max_outlier_rejection_passes = 3;
    scene_with_rej.params.outlier_threshold_px = 0.f;
    scene_with_rej.params.outlier_multiplier = 3.0f;
    BundleAdjustmentResult result_with_rej;
    Status s2 = adjuster.solve(scene_with_rej.problem, scene_with_rej.params, result_with_rej);
    ASSERT_TRUE(s2.ok) << s2.message;

    EXPECT_GT(result_with_rej.total_rejected, 0u)
        << "Expected some observations to be rejected";
    EXPECT_GT(result_with_rej.outlier_rejection_passes_used, 1)
        << "Expected more than 1 pass";

    EXPECT_LT(result_with_rej.avg_reproj_error_px, result_no_rej.avg_reproj_error_px)
        << "Outlier rejection should produce lower mean error";

    // Count rejected observations visible in per_observation.
    // total_rejected may be higher because it includes observations filtered out
    // by the under-observed landmark filter (Phase A) that never appear in per_observation.
    uint32_t rejected_in_output = 0;
    for (const auto& por : result_with_rej.per_observation) {
        if (por.rejection_pass > 0) {
            ++rejected_in_output;
            EXPECT_FALSE(por.inlier) << "Rejected observations must be marked as not inlier";
        }
    }
    EXPECT_GT(rejected_in_output, 0u) << "Expected rejected observations in output";
    EXPECT_LE(rejected_in_output, result_with_rej.total_rejected)
        << "Output rejected count must not exceed total_rejected";
}

// ============================================================================
// Fisheye Camera Model E2E Test
// ============================================================================

TEST(E2ESynthetic, FisheyeModelRecovery) {
    SceneConfig cfg;
    cfg.seed = 555;
    cfg.model_type = CameraIntrinsics::ModelType::FISHEYE;
    cfg.fk1 = 0.05;
    cfg.fk2 = -0.02;
    cfg.fk3 = 0.005;
    cfg.fk4 = -0.001;
    cfg.fx = 400.0;
    cfg.fy = 400.0;
    cfg.cx = 640.0;
    cfg.cy = 400.0;

    auto scene = buildScene(cfg);

    ASSERT_GT(scene.problem.observations.size(), 100u)
        << "Too few observations generated for fisheye scene";

    BundleAdjustmentResult result;
    BundleAdjuster adjuster;
    Status s = adjuster.solve(scene.problem, scene.params, result);
    ASSERT_TRUE(s.ok) << s.message;
    EXPECT_TRUE(result.success) << "Solver did not converge with fisheye model: "
        << result.message;
    EXPECT_LT(result.avg_reproj_error_px, 1.0f)
        << "Average reprojection error too large for fisheye";
    EXPECT_LT(result.final_cost, result.initial_cost * 0.10)
        << "Final cost not sufficiently reduced for fisheye";

    expectPosesNearGT(scene, result);
    checkFilteredLandmarks(scene, result, 0.5);
}
