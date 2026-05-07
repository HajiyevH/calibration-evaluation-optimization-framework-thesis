#include "engine/contracts/types.h"
#include "engine/contracts/status.h"
#include "engine/io/json_codec.h"
#include "engine/solver/bundle_adjuster.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <optional>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ba_cli <problem.json> [result.json] [OPTIONS]" << std::endl;
        std::cerr << "Options:" << std::endl;
        std::cerr << "  --max-iterations <N>     Override maximum solver iterations" << std::endl;
        std::cerr << "  --loss-type <TYPE>       Override robust loss (Huber|Cauchy|None|SoftL1)" << std::endl;
        std::cerr << "  --loss-scale <X>         Override robust loss scale parameter" << std::endl;
        std::cerr << "  --outlier-passes <N>     Iterative outlier rejection passes (0=disabled)" << std::endl;
        std::cerr << "  --outlier-threshold <px> Fixed outlier threshold (0=adaptive k*median)" << std::endl;
        return 1;
    }

    std::string problem_path = argv[1];
    std::string result_path = (argc >= 3) ? argv[2] : "result.json";

    // Parse optional command-line overrides
    std::optional<int> max_iter_override;
    std::optional<std::string> loss_type_override;
    std::optional<float> loss_scale_override;
    std::optional<int> outlier_passes_override;
    std::optional<float> outlier_threshold_override;

    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--max-iterations" && i + 1 < argc) {
            max_iter_override = std::stoi(argv[++i]);
        } else if (arg == "--loss-type" && i + 1 < argc) {
            loss_type_override = argv[++i];
        } else if (arg == "--loss-scale" && i + 1 < argc) {
            loss_scale_override = std::stof(argv[++i]);
        } else if (arg == "--outlier-passes" && i + 1 < argc) {
            outlier_passes_override = std::stoi(argv[++i]);
        } else if (arg == "--outlier-threshold" && i + 1 < argc) {
            outlier_threshold_override = std::stof(argv[++i]);
        } else if (arg.rfind("--", 0) == 0) {
            std::cerr << "Warning: Unknown option '" << arg << "', ignoring" << std::endl;
        }
    }

    // Load problem
    ba::BundleAdjustmentProblem problem;
    ba::BundleAdjustmentParams params;

    std::cout << "Loading problem from: " << problem_path << std::endl;
    ba::Status load_status = ba::loadProblemJson(problem_path, problem, params);
    if (!load_status) {
        std::cerr << "Error loading problem: " << load_status.message << std::endl;
        return 1;
    }

    // Apply command-line overrides
    if (max_iter_override.has_value()) {
        params.max_iterations = max_iter_override.value();
        std::cout << "Override: max_iterations = " << params.max_iterations << std::endl;
    }
    if (loss_type_override.has_value()) {
        std::string type = loss_type_override.value();
        if (type == "Huber") {
            params.loss_type = ba::BundleAdjustmentParams::LossType::HUBER;
        } else if (type == "Cauchy") {
            params.loss_type = ba::BundleAdjustmentParams::LossType::CAUCHY;
        } else if (type == "None") {
            params.loss_type = ba::BundleAdjustmentParams::LossType::NONE;
        } else if (type == "SoftL1") {
            params.loss_type = ba::BundleAdjustmentParams::LossType::SOFT_L1;
        } else {
            std::cerr << "Warning: Unknown loss type '" << type << "', ignoring" << std::endl;
        }
        std::cout << "Override: loss_type = " << type << std::endl;
    }
    if (loss_scale_override.has_value()) {
        params.loss_scale = loss_scale_override.value();
        std::cout << "Override: loss_scale = " << params.loss_scale << std::endl;
    }
    if (outlier_passes_override.has_value()) {
        int val = outlier_passes_override.value();
        if (val < 0 || val > 255) {
            std::cerr << "Error: --outlier-passes must be in [0, 255], got " << val << std::endl;
            return 1;
        }
        params.max_outlier_rejection_passes = static_cast<uint8_t>(val);
        std::cout << "Override: outlier_passes = " << static_cast<int>(params.max_outlier_rejection_passes) << std::endl;
    }
    if (outlier_threshold_override.has_value()) {
        params.outlier_threshold_px = outlier_threshold_override.value();
        std::cout << "Override: outlier_threshold = " << params.outlier_threshold_px << " px" << std::endl;
    }
    // Print summary
    std::cout << "Problem summary:" << std::endl;
    std::cout << "  Cameras:      " << problem.camera_rig.intrinsics.size() << std::endl;
    std::cout << "  Poses:        " << problem.vehicle_poses.poses_buffer.size() << std::endl;
    std::cout << "  Landmarks:    " << problem.landmarks.landmarks.size() << std::endl;
    std::cout << "  Observations: " << problem.observations.size() << std::endl;
    std::cout << "  Opt poses:    " << (problem.optimize_poses ? "yes" : "no") << std::endl;
    std::cout << "  Opt landmarks:" << (problem.optimize_landmarks ? "yes" : "no") << std::endl;
    std::cout << "  Opt intrinsics:" << (problem.optimize_intrinsics ? "yes" : "no") << std::endl;
    std::cout << "  Opt distortion:" << (problem.optimize_distortion ? "yes" : "no") << std::endl;
    std::cout << "  Opt extrinsics:" << (problem.optimize_extrinsics ? "yes" : "no") << std::endl;

    // Solve
    params.verbose = true;
    ba::BundleAdjustmentResult result;
    ba::BundleAdjuster adjuster;

    std::cout << "\nSolving..." << std::endl;
    ba::Status solve_status = adjuster.solve(problem, params, result);
    if (!solve_status) {
        std::cerr << "Solver error: " << solve_status.message << std::endl;
        return 1;
    }

    // Print result
    std::cout << "\nResult:" << std::endl;
    std::cout << "  Success:      " << (result.success ? "yes" : "no") << std::endl;
    std::cout << "  Converged:    " << (result.converged ? "yes" : "no") << std::endl;
    std::cout << "  Termination:  " << result.termination_type << std::endl;
    std::cout << "  Solver:       " << result.linear_solver_type << std::endl;
    std::cout << "  Iterations:   " << result.total_iterations << " (last pass: " << result.iterations_used << ")" << std::endl;
    std::cout << "  Time:         " << std::fixed << std::setprecision(1) << result.total_time_ms << " ms" << std::endl;
    std::cout << "  Initial cost: " << result.initial_cost << std::endl;
    std::cout << "  Final cost:   " << result.final_cost << std::endl;
    float cost_red = (result.initial_cost > 0.f) ? (1.f - result.final_cost / result.initial_cost) * 100.f : 0.f;
    std::cout << "  Cost reduced: " << std::fixed << std::setprecision(1) << cost_red << "%" << std::endl;
    std::cout << "  Mean reproj:  " << result.avg_reproj_error_px << " px" << std::endl;
    std::cout << "  RMSE reproj:  " << result.rmse_reproj_error_px << " px" << std::endl;
    std::cout << "  Max reproj:   " << result.max_reproj_error_px << " px" << std::endl;
    std::cout << "  Inliers:      " << result.num_inliers << "/" << result.num_observations << std::endl;
    if (result.num_pruned > 0) {
        std::cout << "  Pruned:       " << result.num_pruned << std::endl;
    }
    if (result.num_behind_camera > 0) {
        std::cout << "  Behind camera:" << result.num_behind_camera << std::endl;
    }
    if (params.max_outlier_rejection_passes > 0) {
        std::cout << "  Rejection passes: " << static_cast<int>(result.outlier_rejection_passes_used) << std::endl;
        std::cout << "  Total rejected:   " << result.total_rejected << std::endl;
        for (size_t i = 0; i < result.rejected_per_pass.size(); ++i) {
            std::cout << "    Pass " << (i + 1) << ": " << result.rejected_per_pass[i] << " rejected" << std::endl;
        }
    }

    // Write result
    std::cout << "\nWriting result to: " << result_path << std::endl;
    ba::Status write_status = ba::writeResultJson(result_path, result, problem, params);
    if (!write_status) {
        std::cerr << "Error writing result: " << write_status.message << std::endl;
        return 1;
    }

    std::cout << "Done." << std::endl;
    return 0;
}
