#pragma once
#include <cstdint>
#include <vector>
#include <string>

namespace ba {

// ---- Feature types ----

struct Keypoint {
    float x = 0.f;
    float y = 0.f;
};

// ---- Geometry ----

struct PoseSE3 {
    float T[16] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    }; // row-major [R|t; 0 0 0 1], default-initialized to identity
};

struct VehiclePose {
    uint64_t timestamp_ns = 0;
    PoseSE3 pose;
};

struct VehiclePosesBuffer {
    std::vector<VehiclePose> poses_buffer;
};

struct Landmark {
    uint64_t id = 0;
    float x = 0.f;
    float y = 0.f;
    float z = 0.f;
};

struct LandmarkSet {
    std::vector<Landmark> landmarks;
};

// ---- Bundle Adjustment ----

struct Observation2D3D {
    uint64_t landmark_id = 0;
    uint8_t camera_id = 0;
    uint64_t timestamp_ns = 0;
    Keypoint keypoint;
};

struct CameraIntrinsics {
    uint8_t camera_id = 0;

    enum class ModelType : uint8_t {
        PINHOLE,           // Pinhole + radial-tangential distortion (Brown-Conrady)
        FISHEYE,           // Pinhole + fisheye distortion (OpenCV equidistant model)
    } model_type = ModelType::PINHOLE;

    float fx = 300.f;
    float fy = 300.f;
    float cx = 320.f;
    float cy = 240.f;

    // Distortion/projection parameters (model-specific)
    // PINHOLE: params[0..4] = k1, k2, p1, p2, k3 (radial-tangential)
    // FISHEYE: params[0..3] = k1, k2, k3, k4 (fisheye radial)
    float params[8] = {0};

    uint16_t image_width = 640;
    uint16_t image_height = 480;
};

struct CameraExtrinsics {
    uint8_t camera_id = 0;
    PoseSE3 T_vehicle_to_camera;
};

struct CameraRigConfig {
    std::vector<CameraIntrinsics> intrinsics;
    std::vector<CameraExtrinsics> extrinsics;
    uint8_t reference_camera_id = 0;
};

struct BundleAdjustmentProblem {
    LandmarkSet landmarks;
    VehiclePosesBuffer vehicle_poses;
    CameraRigConfig camera_rig;
    std::vector<Observation2D3D> observations;
    bool optimize_intrinsics = false;
    bool optimize_distortion = false;
    bool optimize_extrinsics = false;
    bool optimize_landmarks = true;
    bool optimize_poses = true;
};

struct BundleAdjustmentParams {
    enum class LossType : uint8_t {
        NONE,      // No robust loss (squared error)
        HUBER,     // Huber loss (smooth L1)
        CAUCHY,    // Cauchy loss (heavy-tailed)
        SOFT_L1    // Soft L1 loss
    };

    uint16_t max_iterations = 100;
    float convergence_threshold = 1e-6f;
    LossType loss_type = LossType::HUBER;
    float loss_scale = 1.0f;  // Scale parameter for robust loss (delta for Huber, s for Cauchy)
    float max_reproj_error_px = 2.0f;   // Inlier classification threshold (Phase E reporting)
    bool verbose = false;
    uint16_t min_observations_per_landmark = 2;
    bool prune_behind_camera = true;
    bool prune_high_error = true;
    float prune_error_threshold_px = 50.0f;  // Pre-solve pruning threshold (separate from inlier)

    // Iterative outlier rejection: solve → evaluate → reject → re-solve
    uint8_t max_outlier_rejection_passes = 0;  // 0 = disabled (legacy behavior)
    float outlier_threshold_px = 0.f;          // Fixed threshold; 0 = adaptive (k * median)
    float outlier_multiplier = 3.0f;           // k for adaptive threshold
};

// Per-observation result for detailed result.json export (not in Calibri contracts)
struct PerObservationResult {
    uint64_t timestamp_ns = 0;
    uint8_t camera_id = 0;
    uint64_t landmark_id = 0;
    float u_meas = 0.f;
    float v_meas = 0.f;
    float u_hat = 0.f;
    float v_hat = 0.f;
    float err_px = 0.f;
    bool inlier = true;
    bool pruned = false;         // true = excluded from optimization by pre-solve pruning
    bool behind_camera = false;  // true = point behind camera after optimization
    uint8_t rejection_pass = 0;  // 0 = not rejected; 1+ = rejected in pass N
};

// Per-iteration snapshot from the solver (Ceres IterationSummary)
struct IterationRecord {
    int iteration = 0;
    double cost = 0.0;
    double gradient_norm = 0.0;
    double step_norm = 0.0;
    double relative_decrease = 0.0;
};

struct BundleAdjustmentResult {
    bool success = false;              // true if solver produced a usable result
    bool converged = false;            // true only if Ceres reported CONVERGENCE
    uint16_t iterations_used = 0;      // iterations in last pass
    uint16_t total_iterations = 0;     // iterations across all passes
    float initial_cost = 0.f;          // initial cost of first pass (true initial)
    float final_cost = 0.f;
    std::string message;
    std::string termination_type;       // CONVERGENCE, NO_CONVERGENCE, FAILURE, USER_FAILURE
    std::string linear_solver_type;     // SPARSE_SCHUR, DENSE_QR, etc.
    double total_time_ms = 0.0;         // Total solver wall-clock time across ALL passes

    // Iterative outlier rejection statistics
    uint8_t outlier_rejection_passes_used = 0;
    std::vector<uint32_t> rejected_per_pass;
    uint32_t total_rejected = 0;
    uint32_t num_pruned = 0;            // observations pruned before optimization
    uint32_t num_behind_camera = 0;     // observations behind camera after optimization

    // Per-iteration convergence curve
    std::vector<IterationRecord> iteration_curve;

    // num_observations counts ACTIVE observations only (excludes pruned, rejected, behind-camera).
    // per_observation vector contains ALL observations with status flags for full traceability.
    uint32_t num_observations = 0;
    uint32_t num_inliers = 0;
    uint32_t num_outliers = 0;
    float avg_reproj_error_px = 0.f;
    float median_reproj_error_px = 0.f; // median reprojection error (excludes pruned observations)
    float p95_reproj_error_px = 0.f;    // 95th percentile reprojection error (excludes pruned observations)
    float rmse_reproj_error_px = 0.f;   // sqrt(mean(err^2))
    float std_reproj_error_px = 0.f;    // standard deviation of reprojection errors
    float max_reproj_error_px = 0.f;

    LandmarkSet optimized_landmarks;
    VehiclePosesBuffer optimized_poses;
    CameraRigConfig optimized_camera_rig;

    // Detailed per-observation results for result.json
    std::vector<PerObservationResult> per_observation;
};

} // namespace ba
