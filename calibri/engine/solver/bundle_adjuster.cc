#include "solver/bundle_adjuster.h"
#include "cameras/pinhole_radtan.h"
#include "cameras/pinhole_fisheye.h"
#include "cost/reprojection_cost.h"
#include "cost/reprojection_cost_fisheye.h"
#include "io/contracts_bridge.h"

#include <ceres/ceres.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <memory>
#include <cmath>
#include <numeric>
#include <iostream>

namespace ba {

namespace {

struct PoseBlock {
    double q[4]; // Eigen internal order: x, y, z, w
    double t[3];
};

// Intrinsic block size depends on camera model:
//   Pinhole+RadTan: 9 params [fx, fy, cx, cy, k1, k2, p1, p2, k3]
//   Fisheye:        8 params [fx, fy, cx, cy, k1, k2, k3, k4]
constexpr int PINHOLE_INTR_SIZE = 9;
constexpr int FISHEYE_INTR_SIZE = 8;

struct IntrinsicBlock {
    double params[9] = {};  // Zero-initialized (fixes uninitialized params[8] for fisheye)
    int size = PINHOLE_INTR_SIZE;
};

struct LandmarkBlock {
    double xyz[3];
};

// Compute reprojection error for one observation.
// Encapsulates the transform chain (world -> vehicle -> camera) and projection.
// Returns the error in pixels, and writes the predicted pixel coordinates to uv_pred.
// Returns -1.0 if the point is behind the camera (uv_pred is set to NaN).
double computeReprojError(
    const PoseBlock& pose, const PoseBlock& extr,
    const IntrinsicBlock& intr, const LandmarkBlock& lm,
    CameraIntrinsics::ModelType model_type,
    double obs_u, double obs_v,
    Eigen::Vector2d& uv_pred) {

    // Quaternion index reordering: Eigen internal order is [x,y,z,w]
    // (indices 0,1,2,3) but the Quaterniond constructor takes (w,x,y,z).
    Eigen::Quaterniond q_wv(pose.q[3], pose.q[0], pose.q[1], pose.q[2]);
    Eigen::Vector3d t_wv(pose.t[0], pose.t[1], pose.t[2]);
    Eigen::Quaterniond q_vc(extr.q[3], extr.q[0], extr.q[1], extr.q[2]);
    Eigen::Vector3d t_vc(extr.t[0], extr.t[1], extr.t[2]);
    Eigen::Vector3d Xw(lm.xyz[0], lm.xyz[1], lm.xyz[2]);

    Eigen::Vector3d X_vehicle = q_wv.conjugate() * (Xw - t_wv);
    Eigen::Vector3d X_camera = q_vc * X_vehicle + t_vc;

    // Point behind camera — set uv_pred to NaN to make misuse obvious
    if (X_camera.z() <= 1e-6) {
        uv_pred = Eigen::Vector2d(std::numeric_limits<double>::quiet_NaN(),
                                  std::numeric_limits<double>::quiet_NaN());
        return -1.0;
    }

    if (model_type == CameraIntrinsics::ModelType::FISHEYE) {
        PinholeFisheyeIntr K;
        K.fx = intr.params[0]; K.fy = intr.params[1];
        K.cx = intr.params[2]; K.cy = intr.params[3];
        K.k1 = intr.params[4]; K.k2 = intr.params[5];
        K.k3 = intr.params[6]; K.k4 = intr.params[7];
        uv_pred = projectPinholeFisheye(X_camera, K);
    } else {
        PinholeIntr K;
        K.fx = intr.params[0]; K.fy = intr.params[1];
        K.cx = intr.params[2]; K.cy = intr.params[3];
        K.k1 = intr.params[4]; K.k2 = intr.params[5];
        K.p1 = intr.params[6]; K.p2 = intr.params[7];
        K.k3 = intr.params[8];
        uv_pred = projectPinholeRadTan(X_camera, K);
    }

    double du = obs_u - uv_pred.x();
    double dv = obs_v - uv_pred.y();
    return std::sqrt(du * du + dv * dv);
}

} // anonymous namespace

Status BundleAdjuster::solve(const BundleAdjustmentProblem& problem,
                             const BundleAdjustmentParams& params,
                             BundleAdjustmentResult& result) {
    
    // Phase A: Extract parameter blocks (ONCE, before rejection loop)

    // Pose blocks indexed by timestamp_ns
    std::unordered_map<uint64_t, size_t> pose_index;
    std::vector<PoseBlock> pose_blocks;
    pose_blocks.reserve(problem.vehicle_poses.poses_buffer.size());

    for (size_t i = 0; i < problem.vehicle_poses.poses_buffer.size(); ++i) {
        const auto& vp = problem.vehicle_poses.poses_buffer[i];
        PoseBlock pb;
        PoseSE3_To_qt(vp.pose, pb.q, pb.t);
        pose_blocks.push_back(pb);
        pose_index[vp.timestamp_ns] = i;
    }

    // Camera intrinsic blocks indexed by camera_id
    std::unordered_map<uint8_t, size_t> cam_index;
    std::vector<IntrinsicBlock> intr_blocks;
    intr_blocks.reserve(problem.camera_rig.intrinsics.size());

    for (size_t i = 0; i < problem.camera_rig.intrinsics.size(); ++i) {
        const auto& cam = problem.camera_rig.intrinsics[i];
        IntrinsicBlock ib;  // Zero-initialized params via = {}
        ib.params[0] = static_cast<double>(cam.fx);
        ib.params[1] = static_cast<double>(cam.fy);
        ib.params[2] = static_cast<double>(cam.cx);
        ib.params[3] = static_cast<double>(cam.cy);

        if (cam.model_type == CameraIntrinsics::ModelType::FISHEYE) {
            // Fisheye: fx, fy, cx, cy, k1, k2, k3, k4 (8 params)
            ib.params[4] = static_cast<double>(cam.params[0]); // k1
            ib.params[5] = static_cast<double>(cam.params[1]); // k2
            ib.params[6] = static_cast<double>(cam.params[2]); // k3
            ib.params[7] = static_cast<double>(cam.params[3]); // k4
            ib.size = FISHEYE_INTR_SIZE;
        } else {
            // Pinhole radtan: fx, fy, cx, cy, k1, k2, p1, p2, k3 (9 params)
            ib.params[4] = static_cast<double>(cam.params[0]); // k1
            ib.params[5] = static_cast<double>(cam.params[1]); // k2
            ib.params[6] = static_cast<double>(cam.params[2]); // p1
            ib.params[7] = static_cast<double>(cam.params[3]); // p2
            ib.params[8] = static_cast<double>(cam.params[4]); // k3
            ib.size = PINHOLE_INTR_SIZE;
        }

        intr_blocks.push_back(ib);
        cam_index[cam.camera_id] = i;
    }

    // Camera extrinsic blocks indexed by camera_id
    std::unordered_map<uint8_t, size_t> extr_index;
    std::vector<PoseBlock> extr_blocks;
    extr_blocks.reserve(problem.camera_rig.extrinsics.size());

    for (size_t i = 0; i < problem.camera_rig.extrinsics.size(); ++i) {
        const auto& ext = problem.camera_rig.extrinsics[i];
        PoseBlock pb;
        PoseSE3_To_qt(ext.T_vehicle_to_camera, pb.q, pb.t);
        extr_blocks.push_back(pb);
        extr_index[ext.camera_id] = i;
    }

    // Landmark blocks indexed by landmark_id
    std::unordered_map<uint64_t, size_t> lm_index;
    std::vector<LandmarkBlock> lm_blocks;
    lm_blocks.reserve(problem.landmarks.landmarks.size());

    for (size_t i = 0; i < problem.landmarks.landmarks.size(); ++i) {
        const auto& lm = problem.landmarks.landmarks[i];
        LandmarkBlock lb;
        lb.xyz[0] = static_cast<double>(lm.x);
        lb.xyz[1] = static_cast<double>(lm.y);
        lb.xyz[2] = static_cast<double>(lm.z);
        lm_blocks.push_back(lb);
        lm_index[lm.id] = i;
    }

    // Count observations per landmark for min_observations_per_landmark filtering
    std::unordered_map<uint64_t, uint16_t> landmark_obs_count;
    for (const auto& obs : problem.observations) {
        ++landmark_obs_count[obs.landmark_id];
    }

    // Build set of under-observed landmarks to exclude
    std::unordered_set<uint64_t> underobserved_landmarks;
    if (params.min_observations_per_landmark > 1) {
        for (const auto& [lm_id, count] : landmark_obs_count) {
            if (count < params.min_observations_per_landmark) {
                underobserved_landmarks.insert(lm_id);
            }
        }
        if (params.verbose && !underobserved_landmarks.empty()) {
            std::cout << "Filtered " << underobserved_landmarks.size()
                      << " landmarks with fewer than " << params.min_observations_per_landmark
                      << " observations\n";
        }
    }

    // Loss function — owned by unique_ptr to prevent leaks on early return.
    // Ceres Problem is configured with DO_NOT_TAKE_OWNERSHIP so the loss
    // can be safely shared across all residual blocks and across passes.
    std::unique_ptr<ceres::LossFunction> loss;
    switch (params.loss_type) {
        case BundleAdjustmentParams::LossType::NONE:
            break;  // nullptr = no robust loss (squared error)
        case BundleAdjustmentParams::LossType::HUBER:
            loss = std::make_unique<ceres::HuberLoss>(static_cast<double>(params.loss_scale));
            break;
        case BundleAdjustmentParams::LossType::CAUCHY:
            loss = std::make_unique<ceres::CauchyLoss>(static_cast<double>(params.loss_scale));
            break;
        case BundleAdjustmentParams::LossType::SOFT_L1:
            loss = std::make_unique<ceres::SoftLOneLoss>(static_cast<double>(params.loss_scale));
            break;
    }

    // Solver options (ONCE, before rejection loop)
    ceres::Solver::Options options;
    options.max_num_iterations = params.max_iterations;
    options.num_threads = 8;
    options.function_tolerance = static_cast<double>(params.convergence_threshold);
    options.gradient_tolerance = static_cast<double>(params.convergence_threshold) * 0.01;
    options.parameter_tolerance = static_cast<double>(params.convergence_threshold);
    options.minimizer_progress_to_stdout = params.verbose;

    // Sparse solver selection: SPARSE_SCHUR exploits BA's block-sparse structure
    // (camera/pose params vs landmarks), giving O(n) per iteration instead of O(n^3).
    if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::SUITE_SPARSE)) {
        options.linear_solver_type = ceres::SPARSE_SCHUR;
    } else if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::EIGEN_SPARSE)) {
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
    } else {
        options.linear_solver_type = ceres::DENSE_QR;
    }

    // Save initial intrinsic values for parameter bounds (before any optimization modifies them)
    std::vector<IntrinsicBlock> initial_intr_blocks = intr_blocks;

    
    // Outer rejection loop: solve -> evaluate -> reject -> re-solve
    // When max_outlier_rejection_passes == 0, this executes exactly once
    // (pass 0 only), preserving legacy behavior.
    
    const int total_passes = 1 + static_cast<int>(params.max_outlier_rejection_passes);

    std::unordered_set<size_t> pruned_indices;     // Pre-solve pruning (pass 0 only)
    std::unordered_set<size_t> rejected_indices;   // Post-solve outlier rejection (accumulates)
    // Track which pass rejected each observation (indexed by obs_idx)
    std::unordered_map<size_t, uint8_t> rejection_pass_map;

    size_t pruned_behind_camera = 0;
    size_t pruned_high_error = 0;

    // Multi-pass tracking: accumulate stats across all passes
    double accumulated_time_sec = 0.0;
    uint16_t accumulated_iterations = 0;
    float pass0_initial_cost = 0.f;

    ceres::Solver::Summary summary;  // Last pass summary used for result extraction

    result.rejected_per_pass.clear();
    result.total_rejected = 0;

    for (int pass = 0; pass < total_passes; ++pass) {

        // Create a new Ceres problem each pass (parameter blocks persist in our arrays)
        ceres::Problem::Options problem_options;
        problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
        ceres::Problem ceres_problem(problem_options);

        
        // Phase B: Add residual blocks
        
        size_t residuals_added = 0;

        // Pre-solve pruning only on pass 0
        if (pass == 0) {
            pruned_behind_camera = 0;
            pruned_high_error = 0;
        }

        for (size_t obs_idx = 0; obs_idx < problem.observations.size(); ++obs_idx) {
            // Skip observations pruned in pass 0
            if (pruned_indices.count(obs_idx)) continue;
            // Skip observations rejected in previous passes
            if (rejected_indices.count(obs_idx)) continue;

            const auto& obs = problem.observations[obs_idx];

            // Skip observations for under-observed landmarks
            if (underobserved_landmarks.count(obs.landmark_id)) continue;

            auto pit = pose_index.find(obs.timestamp_ns);
            auto cit = cam_index.find(obs.camera_id);
            auto eit = extr_index.find(obs.camera_id);
            auto lit = lm_index.find(obs.landmark_id);

            if (pit == pose_index.end() || cit == cam_index.end() ||
                eit == extr_index.end() || lit == lm_index.end()) {
                continue; // skip unmatched observations
            }

            PoseBlock& pose = pose_blocks[pit->second];
            PoseBlock& extr = extr_blocks[eit->second];
            IntrinsicBlock& intr = intr_blocks[cit->second];
            LandmarkBlock& lm = lm_blocks[lit->second];

            // --- Phase B.1: Observation Pruning (pass 0 only) ---
            // Uses prune_error_threshold_px (default 50 px) — NOT max_reproj_error_px
            if (pass == 0 && (params.prune_behind_camera || params.prune_high_error)) {
                Eigen::Vector2d uv_pred;
                const CameraIntrinsics& cam_intr = problem.camera_rig.intrinsics[cit->second];
                double err = computeReprojError(pose, extr, intr, lm,
                    cam_intr.model_type,
                    static_cast<double>(obs.keypoint.x),
                    static_cast<double>(obs.keypoint.y), uv_pred);

                // Check 1: Prune points behind the camera (negative Z)
                if (params.prune_behind_camera && err < 0.0) {
                    ++pruned_behind_camera;
                    pruned_indices.insert(obs_idx);
                    continue;
                }

                // Check 2: Prune observations with high initial reprojection error
                if (params.prune_high_error && err >= 0.0 &&
                    static_cast<float>(err) > params.prune_error_threshold_px) {
                    ++pruned_high_error;
                    pruned_indices.insert(obs_idx);
                    continue;
                }
            }

            // Select cost function based on camera model
            const CameraIntrinsics& cam_intr = problem.camera_rig.intrinsics[cit->second];

            Eigen::Vector2d uv(static_cast<double>(obs.keypoint.x),
                               static_cast<double>(obs.keypoint.y));

            ceres::CostFunction* cost = nullptr;
            switch (cam_intr.model_type) {
                case CameraIntrinsics::ModelType::PINHOLE:
                    cost = ReprojectionCostPinhole::Create(uv);
                    break;
                case CameraIntrinsics::ModelType::FISHEYE:
                    cost = ReprojectionCostFisheye::Create(uv);
                    break;
                default:
                    return Status::Error("Unsupported camera model type (only PINHOLE and FISHEYE are supported)");
            }

            ceres_problem.AddResidualBlock(cost, loss.get(),
                                           pose.q, pose.t,
                                           extr.q, extr.t,
                                           intr.params,
                                           lm.xyz);
            ++residuals_added;
        }

        if (residuals_added == 0) {
            return Status::Error("No residuals added — check observations match poses/cameras/landmarks");
        }

        if (params.verbose) {
            std::cout << "Pass " << pass << " Phase B: Observation processing complete\n";
            std::cout << "  Total observations: " << problem.observations.size() << "\n";
            std::cout << "  Residuals added: " << residuals_added << "\n";
            if (pass == 0 && params.prune_behind_camera) {
                std::cout << "  Pruned (behind camera): " << pruned_behind_camera << "\n";
            }
            if (pass == 0 && params.prune_high_error) {
                std::cout << "  Pruned (high error > " << params.prune_error_threshold_px << " px): " << pruned_high_error << "\n";
            }
            if (pass > 0) {
                std::cout << "  Previously rejected: " << rejected_indices.size() << "\n";
            }
        }

        
        // Phase C: Constraints
        

        // Quaternion manifold for all pose quaternion blocks
        // New manifold each pass (owned by the Problem)
#if CERES_VERSION_MAJOR > 2 || (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
        auto* q_manifold = new ceres::EigenQuaternionManifold();
        for (auto& pb : pose_blocks) {
            if (ceres_problem.HasParameterBlock(pb.q)) {
                ceres_problem.SetManifold(pb.q, q_manifold);
            }
        }
        for (auto& eb : extr_blocks) {
            if (ceres_problem.HasParameterBlock(eb.q)) {
                ceres_problem.SetManifold(eb.q, q_manifold);
            }
        }
#else
        auto* q_param = new ceres::EigenQuaternionParameterization();
        for (auto& pb : pose_blocks) {
            if (ceres_problem.HasParameterBlock(pb.q)) {
                ceres_problem.SetParameterization(pb.q, q_param);
            }
        }
        for (auto& eb : extr_blocks) {
            if (ceres_problem.HasParameterBlock(eb.q)) {
                ceres_problem.SetParameterization(eb.q, q_param);
            }
        }
#endif

        // Set extrinsics constant (I1)
        if (!problem.optimize_extrinsics) {
            for (auto& eb : extr_blocks) {
                if (ceres_problem.HasParameterBlock(eb.q)) {
                    ceres_problem.SetParameterBlockConstant(eb.q);
                }
                if (ceres_problem.HasParameterBlock(eb.t)) {
                    ceres_problem.SetParameterBlockConstant(eb.t);
                }
            }
        }

        // When optimizing extrinsics, fix the reference camera to resolve
        // the 6-DOF extrinsic gauge ambiguity (T_delta absorbed into poses).
        // See Rule 2 of the 4-camera implementation spec.
        if (problem.optimize_extrinsics) {
            uint8_t ref_id = problem.camera_rig.reference_camera_id;
            auto rit = extr_index.find(ref_id);
            if (rit != extr_index.end()) {
                if (ceres_problem.HasParameterBlock(extr_blocks[rit->second].q)) {
                    ceres_problem.SetParameterBlockConstant(extr_blocks[rit->second].q);
                }
                if (ceres_problem.HasParameterBlock(extr_blocks[rit->second].t)) {
                    ceres_problem.SetParameterBlockConstant(extr_blocks[rit->second].t);
                }
            } else if (extr_blocks.size() > 1) {
                // Multi-camera rig requires a valid reference camera for gauge fixing (Rule 2)
                return Status::Error("Extrinsic gauge fix failed: reference_camera_id="
                    + std::to_string(ref_id) + " not found in extrinsics. "
                    "Cannot optimize multi-camera extrinsics without a fixed reference camera (Rule 2).");
            }
            // Single camera: gauge is resolved by the fixed first pose; no reference camera needed.
        }

        // Intrinsics constraint: SubsetManifold for selective index freezing
        // Pinhole: indices 0-3 = [fx, fy, cx, cy], indices 4-8 = [k1, k2, p1, p2, k3] (size 9)
        // Fisheye: indices 0-3 = [fx, fy, cx, cy], indices 4-7 = [k1, k2, k3, k4]     (size 8)
        for (auto& ib : intr_blocks) {
            if (!ceres_problem.HasParameterBlock(ib.params)) continue;

            if (!problem.optimize_intrinsics && !problem.optimize_distortion) {
                // Both flags false: entire block constant (I1 baseline behavior)
                ceres_problem.SetParameterBlockConstant(ib.params);
            } else if (!problem.optimize_intrinsics || !problem.optimize_distortion) {
                // One flag true, one false: use SubsetManifold to freeze a subset
                std::vector<int> constant_indices;
                if (!problem.optimize_intrinsics) {
                    // Freeze fx, fy, cx, cy (indices 0-3)
                    constant_indices = {0, 1, 2, 3};
                } else {
                    // Freeze distortion indices (model-dependent range)
                    for (int j = 4; j < ib.size; ++j)
                        constant_indices.push_back(j);
                }
#if CERES_VERSION_MAJOR > 2 || (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
                ceres_problem.SetManifold(ib.params,
                    new ceres::SubsetManifold(ib.size, constant_indices));
#else
                ceres_problem.SetParameterization(ib.params,
                    new ceres::SubsetParameterization(ib.size, constant_indices));
#endif
            }
            // else: both flags true, all parameters free, no manifold needed
        }

        // Parameter bounds for intrinsics when free
        // Use initial values (before optimization) for bound computation
        for (size_t idx = 0; idx < intr_blocks.size(); ++idx) {
            auto& ib = intr_blocks[idx];
            if (!ceres_problem.HasParameterBlock(ib.params)) continue;

            const auto& init_ib = initial_intr_blocks[idx];

            if (problem.optimize_intrinsics) {
                // fx, fy: 50%-200% of initial value
                ceres_problem.SetParameterLowerBound(ib.params, 0, init_ib.params[0] * 0.5);
                ceres_problem.SetParameterUpperBound(ib.params, 0, init_ib.params[0] * 2.0);
                ceres_problem.SetParameterLowerBound(ib.params, 1, init_ib.params[1] * 0.5);
                ceres_problem.SetParameterUpperBound(ib.params, 1, init_ib.params[1] * 2.0);
                // cx, cy: initial +/- 100 pixels
                ceres_problem.SetParameterLowerBound(ib.params, 2, init_ib.params[2] - 100.0);
                ceres_problem.SetParameterUpperBound(ib.params, 2, init_ib.params[2] + 100.0);
                ceres_problem.SetParameterLowerBound(ib.params, 3, init_ib.params[3] - 100.0);
                ceres_problem.SetParameterUpperBound(ib.params, 3, init_ib.params[3] + 100.0);
            }
            if (problem.optimize_distortion) {
                // All distortion coefficients: [-1.0, +1.0]
                for (int j = 4; j < ib.size; ++j) {
                    ceres_problem.SetParameterLowerBound(ib.params, j, -1.0);
                    ceres_problem.SetParameterUpperBound(ib.params, j,  1.0);
                }
            }
        }

        // Set landmarks constant if not optimizing
        if (!problem.optimize_landmarks) {
            for (auto& lb : lm_blocks) {
                if (ceres_problem.HasParameterBlock(lb.xyz)) {
                    ceres_problem.SetParameterBlockConstant(lb.xyz);
                }
            }
        }

        // Gauge fix: first pose (both q and t constant).
        // Validate that the first pose actually participates in the problem.
        if (!pose_blocks.empty() && problem.optimize_poses) {
            bool gauge_applied = false;
            for (size_t gi = 0; gi < pose_blocks.size(); ++gi) {
                if (ceres_problem.HasParameterBlock(pose_blocks[gi].q)) {
                    ceres_problem.SetParameterBlockConstant(pose_blocks[gi].q);
                    ceres_problem.SetParameterBlockConstant(pose_blocks[gi].t);
                    gauge_applied = true;
                    if (gi > 0 && params.verbose) {
                        std::cout << "  Gauge fix: first " << gi << " pose(s) have no residual blocks, "
                                  << "fixed pose[" << gi << "] instead\n";
                    }
                    break;
                }
            }
            if (!gauge_applied && params.verbose) {
                std::cout << "  WARNING: No pose could be fixed for gauge — "
                          << "Hessian may be rank-deficient\n";
            }
        }

        // Set all poses constant if not optimizing
        if (!problem.optimize_poses) {
            for (auto& pb : pose_blocks) {
                if (ceres_problem.HasParameterBlock(pb.q)) {
                    ceres_problem.SetParameterBlockConstant(pb.q);
                }
                if (ceres_problem.HasParameterBlock(pb.t)) {
                    ceres_problem.SetParameterBlockConstant(pb.t);
                }
            }
        }

        
        // Phase D: Solve
        
        ceres::Solve(options, &ceres_problem, &summary);

        // Accumulate multi-pass statistics
        accumulated_time_sec += summary.total_time_in_seconds;
        accumulated_iterations += static_cast<uint16_t>(summary.iterations.size());
        if (pass == 0) {
            pass0_initial_cost = static_cast<float>(summary.initial_cost);
        }

        if (params.verbose) {
            std::cout << summary.FullReport() << std::endl;
        }

        
        // Outlier rejection evaluation (between passes, not on last pass)
        
        if (pass < total_passes - 1) {
            // Evaluate residuals for all active observations
            std::vector<float> active_errors;
            active_errors.reserve(problem.observations.size());

            // Collect (obs_idx, error) pairs for active observations
            std::vector<std::pair<size_t, float>> obs_errors;
            obs_errors.reserve(problem.observations.size());

            for (size_t obs_idx = 0; obs_idx < problem.observations.size(); ++obs_idx) {
                if (pruned_indices.count(obs_idx)) continue;
                if (rejected_indices.count(obs_idx)) continue;

                const auto& obs = problem.observations[obs_idx];
                auto pit = pose_index.find(obs.timestamp_ns);
                auto cit = cam_index.find(obs.camera_id);
                auto eit = extr_index.find(obs.camera_id);
                auto lit = lm_index.find(obs.landmark_id);

                if (pit == pose_index.end() || cit == cam_index.end() ||
                    eit == extr_index.end() || lit == lm_index.end()) {
                    continue;
                }

                const CameraIntrinsics& cam_intr = problem.camera_rig.intrinsics[cit->second];
                Eigen::Vector2d uv_pred;
                double err = computeReprojError(
                    pose_blocks[pit->second], extr_blocks[eit->second],
                    intr_blocks[cit->second], lm_blocks[lit->second],
                    cam_intr.model_type,
                    static_cast<double>(obs.keypoint.x),
                    static_cast<double>(obs.keypoint.y), uv_pred);

                if (err < 0.0) err = 1e6;  // behind camera = extreme outlier
                float err_f = static_cast<float>(err);
                active_errors.push_back(err_f);
                obs_errors.push_back({obs_idx, err_f});
            }

            // Compute rejection threshold
            float threshold = 0.f;
            if (params.outlier_threshold_px > 0.f) {
                // Fixed threshold mode
                threshold = params.outlier_threshold_px;
            } else {
                // Adaptive threshold: k * median
                std::vector<float> sorted_errors = active_errors;
                std::sort(sorted_errors.begin(), sorted_errors.end());
                float median = 0.f;
                if (!sorted_errors.empty()) {
                    size_t mid = sorted_errors.size() / 2;
                    median = (sorted_errors.size() % 2 == 0)
                        ? (sorted_errors[mid - 1] + sorted_errors[mid]) / 2.f
                        : sorted_errors[mid];
                }
                threshold = params.outlier_multiplier * median;
            }

            // Reject observations above threshold
            uint32_t new_rejections = 0;
            uint8_t pass_number = static_cast<uint8_t>(pass + 1);  // 1-indexed
            for (const auto& [obs_idx, err] : obs_errors) {
                if (err > threshold) {
                    rejected_indices.insert(obs_idx);
                    rejection_pass_map[obs_idx] = pass_number;
                    ++new_rejections;
                }
            }

            result.rejected_per_pass.push_back(new_rejections);
            result.total_rejected += new_rejections;

            if (params.verbose) {
                std::cout << "Pass " << pass << " rejection: threshold=" << threshold
                          << " px, rejected=" << new_rejections
                          << ", total_rejected=" << result.total_rejected << "\n";
            }

            // Early termination: no new rejections means the solution is stable
            if (new_rejections == 0) {
                result.outlier_rejection_passes_used = static_cast<uint8_t>(pass + 1);
                break;
            }
        }

        result.outlier_rejection_passes_used = static_cast<uint8_t>(pass + 1);
    } // end outer rejection loop

    // Phase E: Extract results (ONCE, after all passes)
    
    result.success = (summary.termination_type == ceres::CONVERGENCE ||
                      summary.termination_type == ceres::NO_CONVERGENCE);
    result.converged = (summary.termination_type == ceres::CONVERGENCE);
    result.iterations_used = static_cast<uint16_t>(summary.iterations.size());
    result.total_iterations = accumulated_iterations;
    result.initial_cost = pass0_initial_cost;  // True initial cost from first pass
    result.final_cost = static_cast<float>(summary.final_cost);
    result.message = summary.message;

    // Termination type string
    switch (summary.termination_type) {
        case ceres::CONVERGENCE:    result.termination_type = "CONVERGENCE"; break;
        case ceres::NO_CONVERGENCE: result.termination_type = "NO_CONVERGENCE"; break;
        case ceres::FAILURE:        result.termination_type = "FAILURE"; break;
        case ceres::USER_FAILURE:   result.termination_type = "USER_FAILURE"; break;
        default:                    result.termination_type = "UNKNOWN"; break;
    }

    // Linear solver type string
    result.linear_solver_type = ceres::LinearSolverTypeToString(
        summary.linear_solver_type_used);

    // Total solver time across ALL passes
    result.total_time_ms = accumulated_time_sec * 1000.0;

    // Per-iteration convergence curve (last pass only — for multi-pass,
    // earlier passes have already converged before rejection)
    result.iteration_curve.clear();
    result.iteration_curve.reserve(summary.iterations.size());
    for (const auto& iter : summary.iterations) {
        IterationRecord rec;
        rec.iteration = iter.iteration;
        rec.cost = iter.cost;
        rec.gradient_norm = iter.gradient_max_norm;
        rec.step_norm = iter.step_norm;
        rec.relative_decrease = iter.relative_decrease;
        result.iteration_curve.push_back(rec);
    }

    // Optimized poses
    result.optimized_poses.poses_buffer.resize(problem.vehicle_poses.poses_buffer.size());
    for (size_t i = 0; i < pose_blocks.size(); ++i) {
        auto& vp = result.optimized_poses.poses_buffer[i];
        vp.timestamp_ns = problem.vehicle_poses.poses_buffer[i].timestamp_ns;
        vp.pose = qt_To_PoseSE3(pose_blocks[i].q, pose_blocks[i].t);
    }

    // Optimized landmarks
    result.optimized_landmarks.landmarks.resize(problem.landmarks.landmarks.size());
    for (size_t i = 0; i < lm_blocks.size(); ++i) {
        auto& lm = result.optimized_landmarks.landmarks[i];
        lm.id = problem.landmarks.landmarks[i].id;
        lm.x = static_cast<float>(lm_blocks[i].xyz[0]);
        lm.y = static_cast<float>(lm_blocks[i].xyz[1]);
        lm.z = static_cast<float>(lm_blocks[i].xyz[2]);
    }

    // Optimized camera rig: copy input (preserves extrinsics, image dims, model type),
    // then overwrite intrinsics/distortion from solver blocks
    result.optimized_camera_rig = problem.camera_rig;
    for (const auto& [camera_id, idx] : cam_index) {
        const IntrinsicBlock& ib = intr_blocks[idx];
        // Find matching CameraIntrinsics in result by camera_id
        for (auto& ci : result.optimized_camera_rig.intrinsics) {
            if (ci.camera_id == camera_id) {
                ci.fx = static_cast<float>(ib.params[0]);
                ci.fy = static_cast<float>(ib.params[1]);
                ci.cx = static_cast<float>(ib.params[2]);
                ci.cy = static_cast<float>(ib.params[3]);
                // Distortion params: copy only model-relevant count
                int num_dist = ib.size - 4; // 4 for fisheye, 5 for pinhole
                for (int j = 0; j < num_dist; ++j) {
                    ci.params[j] = static_cast<float>(ib.params[4 + j]);
                }
                break;
            }
        }
    }

    // Writeback optimized extrinsics from solver blocks
    for (const auto& [camera_id, idx] : extr_index) {
        const PoseBlock& eb = extr_blocks[idx];
        for (auto& ce : result.optimized_camera_rig.extrinsics) {
            if (ce.camera_id == camera_id) {
                ce.T_vehicle_to_camera = qt_To_PoseSE3(eb.q, eb.t);
                break;
            }
        }
    }

    // Compute per-observation reprojection errors.
    // Includes ALL observations: active, rejected, AND pruned (each marked accordingly).
    result.per_observation.clear();
    result.per_observation.reserve(problem.observations.size());
    std::vector<float> all_errors;       // Only active (non-rejected, non-pruned) for statistics
    all_errors.reserve(problem.observations.size());
    float max_err = 0.f;
    uint32_t num_inliers = 0;
    uint32_t behind_camera_count = 0;

    for (size_t obs_idx = 0; obs_idx < problem.observations.size(); ++obs_idx) {
        const auto& obs = problem.observations[obs_idx];

        auto pit = pose_index.find(obs.timestamp_ns);
        auto cit = cam_index.find(obs.camera_id);
        auto eit = extr_index.find(obs.camera_id);
        auto lit = lm_index.find(obs.landmark_id);

        if (pit == pose_index.end() || cit == cam_index.end() ||
            eit == extr_index.end() || lit == lm_index.end()) {
            continue;
        }

        // Skip under-observed landmarks (not in the problem at all)
        if (underobserved_landmarks.count(obs.landmark_id)) continue;

        bool is_pruned = pruned_indices.count(obs_idx) > 0;
        bool is_rejected = rejected_indices.count(obs_idx) > 0;

        const CameraIntrinsics& cam_intr = problem.camera_rig.intrinsics[cit->second];
        Eigen::Vector2d uv_pred;
        double err_d = computeReprojError(
            pose_blocks[pit->second], extr_blocks[eit->second],
            intr_blocks[cit->second], lm_blocks[lit->second],
            cam_intr.model_type,
            static_cast<double>(obs.keypoint.x),
            static_cast<double>(obs.keypoint.y), uv_pred);

        bool is_behind = (err_d < 0.0);
        if (is_behind) ++behind_camera_count;

        // For behind-camera points, report a large sentinel error (not 0)
        float err = is_behind ? 1e4f : static_cast<float>(err_d);

        uint8_t rej_pass = 0;
        auto rp_it = rejection_pass_map.find(obs_idx);
        if (rp_it != rejection_pass_map.end()) {
            rej_pass = rp_it->second;
        }

        PerObservationResult por;
        por.timestamp_ns = obs.timestamp_ns;
        por.camera_id = obs.camera_id;
        por.landmark_id = obs.landmark_id;
        por.u_meas = obs.keypoint.x;
        por.v_meas = obs.keypoint.y;
        por.u_hat = is_behind ? 0.f : static_cast<float>(uv_pred.x());
        por.v_hat = is_behind ? 0.f : static_cast<float>(uv_pred.y());
        por.err_px = err;
        por.pruned = is_pruned;
        por.behind_camera = is_behind;
        // Inlier = not pruned, not rejected, not behind camera, and error below threshold
        por.inlier = !is_pruned && !is_rejected && !is_behind
                     && (err <= params.max_reproj_error_px);
        por.rejection_pass = rej_pass;
        result.per_observation.push_back(por);

        // Only include active (non-rejected, non-pruned, not behind camera) in summary stats
        if (!is_rejected && !is_pruned && !is_behind) {
            all_errors.push_back(err);
            if (err > max_err) max_err = err;
            if (por.inlier) ++num_inliers;
        }
    }

    result.num_observations = static_cast<uint32_t>(all_errors.size());
    result.num_inliers = num_inliers;
    result.num_outliers = result.num_observations - num_inliers;
    result.max_reproj_error_px = max_err;
    result.num_pruned = static_cast<uint32_t>(pruned_indices.size());
    result.num_behind_camera = behind_camera_count;

    if (!all_errors.empty()) {
        double n = static_cast<double>(all_errors.size());
        double sum = std::accumulate(all_errors.begin(), all_errors.end(), 0.0);
        double mean = sum / n;
        result.avg_reproj_error_px = static_cast<float>(mean);

        // RMSE = sqrt(mean(err^2))
        double sum_sq = 0.0;
        for (float e : all_errors) sum_sq += static_cast<double>(e) * e;
        result.rmse_reproj_error_px = static_cast<float>(std::sqrt(sum_sq / n));

        // Sample standard deviation (Bessel's correction: n-1)
        double sum_dev_sq = 0.0;
        for (float e : all_errors) {
            double d = static_cast<double>(e) - mean;
            sum_dev_sq += d * d;
        }
        double denom = (all_errors.size() > 1) ? (n - 1.0) : 1.0;
        result.std_reproj_error_px = static_cast<float>(std::sqrt(sum_dev_sq / denom));

        // Median and P95 (requires sorted copy)
        std::vector<float> sorted_errors = all_errors;
        std::sort(sorted_errors.begin(), sorted_errors.end());

        size_t median_idx = sorted_errors.size() / 2;
        if (sorted_errors.size() % 2 == 0) {
            result.median_reproj_error_px = (sorted_errors[median_idx - 1] + sorted_errors[median_idx]) / 2.0f;
        } else {
            result.median_reproj_error_px = sorted_errors[median_idx];
        }

        size_t p95_idx = static_cast<size_t>(std::ceil(0.95 * sorted_errors.size())) - 1;
        if (p95_idx >= sorted_errors.size()) p95_idx = sorted_errors.size() - 1;
        result.p95_reproj_error_px = sorted_errors[p95_idx];
    } else {
        result.avg_reproj_error_px = 0.f;
        result.median_reproj_error_px = 0.f;
        result.p95_reproj_error_px = 0.f;
        result.rmse_reproj_error_px = 0.f;
        result.std_reproj_error_px = 0.f;
    }

    return Status::OK();
}

} // namespace ba
