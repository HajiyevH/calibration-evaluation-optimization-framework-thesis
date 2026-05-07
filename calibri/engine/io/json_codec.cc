#include "io/json_codec.h"
#include "io/contracts_bridge.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <vector>

using json = nlohmann::json;

namespace ba {

namespace {

struct ErrorStats {
    float mean = 0.f;
    float median = 0.f;
    float p95 = 0.f;
};

/// Compute mean, median, and 95th-percentile from an error vector.
/// The input is copied and sorted internally.
ErrorStats computeErrorStats(const std::vector<float>& errors) {
    ErrorStats stats;
    if (errors.empty()) return stats;

    std::vector<float> sorted = errors;
    std::sort(sorted.begin(), sorted.end());
    size_t n = sorted.size();

    stats.mean = std::accumulate(sorted.begin(), sorted.end(), 0.f)
                 / static_cast<float>(n);
    stats.median = (n % 2 == 0)
        ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2.f
        : sorted[n / 2];
    size_t p95_idx = static_cast<size_t>(std::ceil(0.95 * n)) - 1;
    stats.p95 = sorted[std::min(p95_idx, n - 1)];
    return stats;
}

} // anonymous namespace

// --- Loading ---

Status loadProblemJson(const std::string& path,
                       BundleAdjustmentProblem& problem,
                       BundleAdjustmentParams& params) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        return Status::Error("Cannot open file: " + path);
    }

    json j;
    try {
        ifs >> j;
    } catch (const json::parse_error& e) {
        return Status::Error(std::string("JSON parse error: ") + e.what());
    } catch (const json::type_error& e) {
        return Status::Error(std::string("JSON type error: ") + e.what());
    }

    // --- Cameras ---
    if (j.contains("cameras")) {
        for (const auto& cam_j : j["cameras"]) {
            CameraIntrinsics ci;
            ci.camera_id = cam_j.value("id", 0);

            std::string model = cam_j.value("model", "Pinhole");
            if (model == "Pinhole") ci.model_type = CameraIntrinsics::ModelType::PINHOLE;
            else if (model == "Fisheye") ci.model_type = CameraIntrinsics::ModelType::FISHEYE;
            else return Status::Error("Unsupported camera model: " + model + " (only Pinhole and Fisheye are supported)");

            ci.image_width = cam_j.value("width", 640);
            ci.image_height = cam_j.value("height", 480);

            if (ci.image_width == 0 || ci.image_height == 0) {
                return Status::Error("Camera id=" + std::to_string(ci.camera_id)
                    + " has invalid image dimensions ("
                    + std::to_string(ci.image_width) + "x"
                    + std::to_string(ci.image_height) + ")");
            }

            if (!cam_j.contains("intrinsics")) {
                return Status::Error("Camera id=" + std::to_string(ci.camera_id)
                    + " missing required 'intrinsics' block (fx, fy, cx, cy)");
            }

            if (cam_j.contains("intrinsics")) {
                const auto& intr = cam_j["intrinsics"];
                ci.fx = intr.value("fx", 300.f);
                ci.fy = intr.value("fy", 300.f);
                ci.cx = intr.value("cx", 320.f);
                ci.cy = intr.value("cy", 240.f);
            }

            if (cam_j.contains("distortion")) {
                const auto& dist = cam_j["distortion"];

                if (ci.model_type == CameraIntrinsics::ModelType::FISHEYE) {
                    // Fisheye: 4 radial distortion coefficients
                    ci.params[0] = dist.value("k1", 0.f);
                    ci.params[1] = dist.value("k2", 0.f);
                    ci.params[2] = dist.value("k3", 0.f);
                    ci.params[3] = dist.value("k4", 0.f);
                } else {
                    // Pinhole: radial-tangential distortion
                    ci.params[0] = dist.value("k1", 0.f);
                    ci.params[1] = dist.value("k2", 0.f);
                    ci.params[2] = dist.value("p1", 0.f);
                    ci.params[3] = dist.value("p2", 0.f);
                    ci.params[4] = dist.value("k3", 0.f);
                }
            }

            problem.camera_rig.intrinsics.push_back(ci);

            // Extrinsics
            CameraExtrinsics ce;
            ce.camera_id = ci.camera_id;
            // Support both "extrinsics": { "T_vehicle_to_camera": {q,t} }
            // and legacy "extrinsic": {q,t}
            const nlohmann::json* ext_ptr = nullptr;
            if (cam_j.contains("extrinsics") && cam_j["extrinsics"].contains("T_vehicle_to_camera")) {
                ext_ptr = &cam_j["extrinsics"]["T_vehicle_to_camera"];
            } else if (cam_j.contains("extrinsic")) {
                ext_ptr = &cam_j["extrinsic"];
            }
            if (ext_ptr) {
                const auto& ext = *ext_ptr;

                // JSON quaternion convention: [w, x, y, z]
                Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
                if (ext.contains("q") && ext["q"].size() == 4) {
                    q = Eigen::Quaterniond(
                        ext["q"][0].get<double>(),  // w
                        ext["q"][1].get<double>(),  // x
                        ext["q"][2].get<double>(),  // y
                        ext["q"][3].get<double>()).normalized(); // z; normalize (Rule 5)
                }

                Eigen::Vector3d t = Eigen::Vector3d::Zero();
                if (ext.contains("t") && ext["t"].size() == 3) {
                    t = Eigen::Vector3d(
                        ext["t"][0].get<double>(),
                        ext["t"][1].get<double>(),
                        ext["t"][2].get<double>());
                }

                ce.T_vehicle_to_camera = qt_To_PoseSE3(q, t);
            }
            problem.camera_rig.extrinsics.push_back(ce);
        }
    }

    // --- Poses ---
    if (j.contains("poses")) {
        for (const auto& pose_j : j["poses"]) {
            VehiclePose vp;
            vp.timestamp_ns = pose_j.value("timestamp_ns", uint64_t(0));

            // JSON quaternion convention: [w, x, y, z]
            Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
            if (pose_j.contains("q") && pose_j["q"].size() == 4) {
                q = Eigen::Quaterniond(
                    pose_j["q"][0].get<double>(),  // w
                    pose_j["q"][1].get<double>(),  // x
                    pose_j["q"][2].get<double>(),  // y
                    pose_j["q"][3].get<double>()).normalized(); // z; normalize (Rule 5)
            }

            Eigen::Vector3d t = Eigen::Vector3d::Zero();
            if (pose_j.contains("t") && pose_j["t"].size() == 3) {
                t = Eigen::Vector3d(
                    pose_j["t"][0].get<double>(),
                    pose_j["t"][1].get<double>(),
                    pose_j["t"][2].get<double>());
            }

            vp.pose = qt_To_PoseSE3(q, t);
            problem.vehicle_poses.poses_buffer.push_back(vp);
        }
    }

    // --- Landmarks ---
    if (j.contains("landmarks")) {
        for (const auto& lm_j : j["landmarks"]) {
            Landmark lm;
            lm.id = lm_j.value("id", uint64_t(0));
            if (lm_j.contains("position") && lm_j["position"].size() == 3) {
                lm.x = lm_j["position"][0].get<float>();
                lm.y = lm_j["position"][1].get<float>();
                lm.z = lm_j["position"][2].get<float>();
            } else {
                // Legacy format: separate X, Y, Z keys
                lm.x = lm_j.value("X", 0.f);
                lm.y = lm_j.value("Y", 0.f);
                lm.z = lm_j.value("Z", 0.f);
            }
            problem.landmarks.landmarks.push_back(lm);
        }
    }

    // --- Observations ---
    if (j.contains("observations")) {
        for (const auto& obs_j : j["observations"]) {
            Observation2D3D obs;
            obs.landmark_id = obs_j.value("landmark_id", uint64_t(0));
            obs.camera_id = obs_j.value("camera_id", uint8_t(0));
            obs.timestamp_ns = obs_j.value("timestamp_ns", uint64_t(0));
            obs.keypoint.x = obs_j.value("u", 0.f);
            obs.keypoint.y = obs_j.value("v", 0.f);
            problem.observations.push_back(obs);
        }
    }

    // --- Flags ---
    if (j.contains("flags")) {
        const auto& flags = j["flags"];
        problem.optimize_intrinsics = flags.value("opt_intrinsics", false);
        problem.optimize_distortion = flags.value("opt_distortion", false);
        problem.optimize_extrinsics = flags.value("opt_extrinsics", false);
        problem.optimize_poses = flags.value("opt_poses", true);
        problem.optimize_landmarks = flags.value("opt_landmarks", true);
    }

    // --- Camera rig config ---
    if (j.contains("camera_rig")) {
        const auto& rig = j["camera_rig"];
        problem.camera_rig.reference_camera_id =
            rig.value("reference_camera_id", uint8_t(0));
    }

    // --- Robust loss ---
    if (j.contains("robust")) {
        const auto& robust = j["robust"];
        std::string type = robust.value("type", "Huber");

        if (type == "None") {
            params.loss_type = BundleAdjustmentParams::LossType::NONE;
        } else if (type == "Huber") {
            params.loss_type = BundleAdjustmentParams::LossType::HUBER;
        } else if (type == "Cauchy") {
            params.loss_type = BundleAdjustmentParams::LossType::CAUCHY;
        } else if (type == "SoftL1") {
            params.loss_type = BundleAdjustmentParams::LossType::SOFT_L1;
        } else {
            params.loss_type = BundleAdjustmentParams::LossType::HUBER;  // Default
        }

        params.loss_scale = robust.value("scale", 1.0f);
    }

    // --- Solver params ---
    if (j.contains("solver")) {
        const auto& solver = j["solver"];
        params.max_iterations = solver.value("max_iterations", 100);
        params.convergence_threshold = solver.value("convergence_threshold", 1e-6f);
        params.prune_error_threshold_px = solver.value("prune_error_threshold_px", 50.0f);
        params.max_reproj_error_px = solver.value("max_reproj_error_px", 2.0f);
    }

    // --- Outlier rejection ---
    if (j.contains("outlier_rejection")) {
        const auto& rej = j["outlier_rejection"];
        params.max_outlier_rejection_passes = rej.value("max_passes", uint8_t(0));
        params.outlier_threshold_px = rej.value("threshold_px", 0.f);
        params.outlier_multiplier = rej.value("multiplier", 3.0f);
    }

    return Status::OK();
}

// --- Writing ---

Status writeResultJson(const std::string& path,
                       const BundleAdjustmentResult& result,
                       const BundleAdjustmentProblem& original_problem,
                       const BundleAdjustmentParams& params) {
    json j;

    // --- Summary ---
    // Use statistics computed by the solver (excludes pruned observations)
    // Note: per_observation contains ALL observations (including pruned),
    // but summary statistics only reflect non-pruned observations
    float cost_reduction = (result.initial_cost > 0.f)
        ? (1.f - result.final_cost / result.initial_cost) : 0.f;

    j["summary"] = {
        {"success", result.success},
        {"converged", result.converged},
        {"iterations", result.iterations_used},
        {"total_iterations", result.total_iterations},
        {"initial_cost", result.initial_cost},
        {"final_cost", result.final_cost},
        {"cost_reduction_ratio", cost_reduction},
        {"total_time_ms", result.total_time_ms},
        {"termination_type", result.termination_type},
        {"linear_solver_type", result.linear_solver_type},
        {"mean_reproj_px", result.avg_reproj_error_px},
        {"median_reproj_px", result.median_reproj_error_px},
        {"p95_reproj_px", result.p95_reproj_error_px},
        {"rmse_reproj_px", result.rmse_reproj_error_px},
        {"std_reproj_px", result.std_reproj_error_px},
        {"max_reproj_px", result.max_reproj_error_px},
        {"num_observations", result.num_observations},
        {"num_inliers", result.num_inliers},
        {"num_outliers", result.num_outliers},
        {"num_pruned", result.num_pruned},
        {"num_behind_camera", result.num_behind_camera},
        {"termination_message", result.message},
        {"outlier_rejection_passes_used", result.outlier_rejection_passes_used},
        {"total_rejected", result.total_rejected},
        {"rejected_per_pass", result.rejected_per_pass}
    };

    // --- Solver configuration used ---
    std::string loss_str;
    switch (params.loss_type) {
        case BundleAdjustmentParams::LossType::NONE:   loss_str = "None"; break;
        case BundleAdjustmentParams::LossType::HUBER:   loss_str = "Huber"; break;
        case BundleAdjustmentParams::LossType::CAUCHY:  loss_str = "Cauchy"; break;
        case BundleAdjustmentParams::LossType::SOFT_L1: loss_str = "SoftL1"; break;
    }
    j["solver_config"] = {
        {"max_iterations", params.max_iterations},
        {"convergence_threshold", params.convergence_threshold},
        {"loss_type", loss_str},
        {"loss_scale", params.loss_scale},
        {"max_reproj_error_px", params.max_reproj_error_px},
        {"prune_error_threshold_px", params.prune_error_threshold_px},
        {"opt_poses", original_problem.optimize_poses},
        {"opt_landmarks", original_problem.optimize_landmarks},
        {"opt_intrinsics", original_problem.optimize_intrinsics},
        {"opt_distortion", original_problem.optimize_distortion},
        {"opt_extrinsics", original_problem.optimize_extrinsics},
        {"outlier_rejection", {
            {"max_passes", params.max_outlier_rejection_passes},
            {"threshold_px", params.outlier_threshold_px},
            {"multiplier", params.outlier_multiplier}
        }}
    };

    // --- Per-iteration convergence curve ---
    json iter_arr = json::array();
    for (const auto& rec : result.iteration_curve) {
        iter_arr.push_back({
            {"iteration", rec.iteration},
            {"cost", rec.cost},
            {"gradient_norm", rec.gradient_norm},
            {"step_norm", rec.step_norm},
            {"relative_decrease", rec.relative_decrease}
        });
    }
    j["convergence_curve"] = iter_arr;

    // --- Per-frame statistics (active observations only — excludes pruned/rejected/behind-camera) ---
    std::unordered_map<uint64_t, std::vector<float>> frame_errors;
    for (const auto& po : result.per_observation) {
        if (!po.pruned && !po.behind_camera && po.rejection_pass == 0) {
            frame_errors[po.timestamp_ns].push_back(po.err_px);
        }
    }

    json per_frame_arr = json::array();
    for (const auto& [ts, errs] : frame_errors) {
        ErrorStats fs = computeErrorStats(errs);
        per_frame_arr.push_back({
            {"timestamp_ns", ts},
            {"mean_px", fs.mean},
            {"median_px", fs.median},
            {"p95_px", fs.p95}
        });
    }
    j["per_frame"] = per_frame_arr;

    // --- Per-camera statistics (active observations only — excludes pruned/rejected/behind-camera) ---
    std::unordered_map<uint8_t, std::vector<float>> cam_errors;
    for (const auto& po : result.per_observation) {
        if (!po.pruned && !po.behind_camera && po.rejection_pass == 0) {
            cam_errors[po.camera_id].push_back(po.err_px);
        }
    }

    json per_cam_arr = json::array();
    for (const auto& [cid, errs] : cam_errors) {
        ErrorStats cs = computeErrorStats(errs);
        per_cam_arr.push_back({
            {"camera_id", cid},
            {"mean_px", cs.mean},
            {"median_px", cs.median},
            {"p95_px", cs.p95}
        });
    }
    j["per_camera"] = per_cam_arr;

    // --- Final params ---
    json final_params;

    // Poses
    json poses_arr = json::array();
    for (const auto& vp : result.optimized_poses.poses_buffer) {
        Eigen::Quaterniond q;
        Eigen::Vector3d t;
        PoseSE3_To_qt(vp.pose, q, t);
        // JSON convention: [w, x, y, z]
        poses_arr.push_back({
            {"timestamp_ns", vp.timestamp_ns},
            {"q", {q.w(), q.x(), q.y(), q.z()}},
            {"t", {t.x(), t.y(), t.z()}}
        });
    }
    final_params["poses"] = poses_arr;

    // Landmarks
    json lm_arr = json::array();
    for (const auto& lm : result.optimized_landmarks.landmarks) {
        lm_arr.push_back({
            {"id", lm.id},
            {"X", lm.x},
            {"Y", lm.y},
            {"Z", lm.z}
        });
    }
    final_params["landmarks"] = lm_arr;

    // Cameras
    json cam_arr = json::array();
    for (size_t i = 0; i < result.optimized_camera_rig.intrinsics.size(); ++i) {
        const auto& ci = result.optimized_camera_rig.intrinsics[i];

        // Build intrinsic array with model-dependent size:
        //   Pinhole+RadTan: 9 params [fx, fy, cx, cy, k1, k2, p1, p2, k3]
        //   Fisheye:        8 params [fx, fy, cx, cy, k1, k2, k3, k4]
        json intr_arr = json::array();
        intr_arr.push_back(static_cast<double>(ci.fx));
        intr_arr.push_back(static_cast<double>(ci.fy));
        intr_arr.push_back(static_cast<double>(ci.cx));
        intr_arr.push_back(static_cast<double>(ci.cy));

        int num_dist = (ci.model_type == CameraIntrinsics::ModelType::FISHEYE) ? 4 : 5;
        for (int k = 0; k < num_dist; ++k) {
            intr_arr.push_back(static_cast<double>(ci.params[k]));
        }

        json cam_entry = {{"id", ci.camera_id}, {"intr", intr_arr}};

        if (i < result.optimized_camera_rig.extrinsics.size()) {
            const auto& ce = result.optimized_camera_rig.extrinsics[i];
            Eigen::Quaterniond eq;
            Eigen::Vector3d et;
            PoseSE3_To_qt(ce.T_vehicle_to_camera, eq, et);
            cam_entry["extrinsic"] = {
                {"q", {eq.w(), eq.x(), eq.y(), eq.z()}},
                {"t", {et.x(), et.y(), et.z()}}
            };
        }
        cam_arr.push_back(cam_entry);
    }
    final_params["cameras"] = cam_arr;
    j["final_params"] = final_params;

    // --- Residuals ---
    json res_arr = json::array();
    for (const auto& po : result.per_observation) {
        res_arr.push_back({
            {"timestamp_ns", po.timestamp_ns},
            {"camera_id", po.camera_id},
            {"landmark_id", po.landmark_id},
            {"u", po.u_meas},
            {"v", po.v_meas},
            {"u_hat", po.u_hat},
            {"v_hat", po.v_hat},
            {"err", po.err_px},
            {"inlier", po.inlier},
            {"pruned", po.pruned},
            {"behind_camera", po.behind_camera},
            {"rejection_pass", po.rejection_pass}
        });
    }
    j["residuals"] = res_arr;

    // Write to file using streaming serializer to avoid materializing the
    // entire JSON string in memory (can exceed stack size for large datasets
    // with 100K+ observations).
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        return Status::Error("Cannot open output file: " + path);
    }

    // nlohmann::json's operator<< streams without building a single string
    ofs << std::setw(2) << j;

    if (!ofs.good()) {
        return Status::Error("Failed to write result JSON to: " + path);
    }
    return Status::OK();
}

} // namespace ba
