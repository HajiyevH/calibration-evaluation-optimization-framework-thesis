#define _USE_MATH_DEFINES
#include <cmath>
#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "engine/contracts/types.h"
#include "engine/contracts/status.h"
#include "engine/solver/bundle_adjuster.h"
#include "engine/io/contracts_bridge.h"

using namespace ba;

// Empty observations should return an error status
TEST(SolverEdgeCases, EmptyObservationsReturnsError) {
    BundleAdjustmentProblem problem;
    BundleAdjustmentParams params;
    BundleAdjustmentResult result;

    // Add a camera, pose, and landmark but no observations
    CameraIntrinsics ci;
    ci.camera_id = 0;
    ci.fx = 500.f; ci.fy = 500.f;
    ci.cx = 320.f; ci.cy = 240.f;
    problem.camera_rig.intrinsics.push_back(ci);

    CameraExtrinsics ce;
    ce.camera_id = 0;
    problem.camera_rig.extrinsics.push_back(ce);

    VehiclePose vp;
    vp.timestamp_ns = 1000000000ULL;
    problem.vehicle_poses.poses_buffer.push_back(vp);

    Landmark lm;
    lm.id = 1; lm.x = 1.f; lm.y = 0.f; lm.z = 10.f;
    problem.landmarks.landmarks.push_back(lm);

    // No observations added
    BundleAdjuster adjuster;
    Status s = adjuster.solve(problem, params, result);
    EXPECT_FALSE(s.ok) << "Should fail with no observations";
}

// Observations referencing non-existent poses/cameras/landmarks should be skipped
TEST(SolverEdgeCases, UnmatchedObservationsSkipped) {
    BundleAdjustmentProblem problem;
    BundleAdjustmentParams params;
    BundleAdjustmentResult result;

    // Add camera and extrinsic
    CameraIntrinsics ci;
    ci.camera_id = 0;
    ci.fx = 500.f; ci.fy = 500.f;
    ci.cx = 320.f; ci.cy = 240.f;
    problem.camera_rig.intrinsics.push_back(ci);

    CameraExtrinsics ce;
    ce.camera_id = 0;
    problem.camera_rig.extrinsics.push_back(ce);

    // Add a pose
    VehiclePose vp;
    vp.timestamp_ns = 1000000000ULL;
    problem.vehicle_poses.poses_buffer.push_back(vp);

    // Add a landmark
    Landmark lm;
    lm.id = 1; lm.x = 0.f; lm.y = 0.f; lm.z = 10.f;
    problem.landmarks.landmarks.push_back(lm);

    // Add observation referencing non-existent camera_id=5
    Observation2D3D obs;
    obs.landmark_id = 1;
    obs.camera_id = 5; // does not exist
    obs.timestamp_ns = 1000000000ULL;
    obs.keypoint.x = 320.f;
    obs.keypoint.y = 240.f;
    problem.observations.push_back(obs);

    BundleAdjuster adjuster;
    Status s = adjuster.solve(problem, params, result);
    // Should fail because all observations were skipped -> 0 residuals
    EXPECT_FALSE(s.ok) << "Should fail when all observations are unmatched";
}

// Single observation with nothing optimized: verify the solver evaluates without error.
// Poses and landmarks are both fixed, so the solver only evaluates the residual.
TEST(SolverEdgeCases, SingleObservationEvaluates) {
    BundleAdjustmentProblem problem;
    BundleAdjustmentParams params;
    params.max_iterations = 10;
    params.loss_type = BundleAdjustmentParams::LossType::NONE;
    params.min_observations_per_landmark = 1;
    BundleAdjustmentResult result;

    // Camera
    CameraIntrinsics ci;
    ci.camera_id = 0;
    ci.fx = 500.f; ci.fy = 500.f;
    ci.cx = 320.f; ci.cy = 240.f;
    problem.camera_rig.intrinsics.push_back(ci);

    CameraExtrinsics ce;
    ce.camera_id = 0;
    problem.camera_rig.extrinsics.push_back(ce);

    // Pose at origin (identity)
    VehiclePose vp;
    vp.timestamp_ns = 1000000000ULL;
    problem.vehicle_poses.poses_buffer.push_back(vp);

    // Landmark at (0, 0, 10) - on optical axis
    Landmark lm;
    lm.id = 1;
    lm.x = 0.f; lm.y = 0.f; lm.z = 10.f;
    problem.landmarks.landmarks.push_back(lm);

    // Observation at principal point (correct projection of on-axis landmark)
    Observation2D3D obs;
    obs.landmark_id = 1;
    obs.camera_id = 0;
    obs.timestamp_ns = 1000000000ULL;
    obs.keypoint.x = 320.f;
    obs.keypoint.y = 240.f;
    problem.observations.push_back(obs);

    problem.optimize_poses = false;
    problem.optimize_landmarks = false;

    BundleAdjuster adjuster;
    Status s = adjuster.solve(problem, params, result);
    ASSERT_TRUE(s.ok) << s.message;
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.per_observation.size(), 1u);
    EXPECT_LT(result.per_observation[0].err_px, 1.0f);
}

// Test behind-camera pruning.
// With pruning enabled, the solver should skip the behind-camera observation
// in Phase B and successfully solve with only the valid (front) observation.
TEST(SolverEdgeCases, BehindCameraPruning) {
    BundleAdjustmentProblem problem;
    BundleAdjustmentParams params;
    params.max_iterations = 10;
    params.prune_behind_camera = true;
    params.min_observations_per_landmark = 1;
    params.verbose = false;
    BundleAdjustmentResult result;

    // Camera at origin
    CameraIntrinsics ci;
    ci.camera_id = 0;
    ci.fx = 500.f; ci.fy = 500.f;
    ci.cx = 320.f; ci.cy = 240.f;
    problem.camera_rig.intrinsics.push_back(ci);

    CameraExtrinsics ce;
    ce.camera_id = 0;
    problem.camera_rig.extrinsics.push_back(ce);

    // Pose at origin
    VehiclePose vp;
    vp.timestamp_ns = 1000000000ULL;
    problem.vehicle_poses.poses_buffer.push_back(vp);

    // Landmark BEHIND camera (negative Z in camera frame)
    Landmark lm_behind;
    lm_behind.id = 1;
    lm_behind.x = 0.f;
    lm_behind.y = 0.f;
    lm_behind.z = -5.f;
    problem.landmarks.landmarks.push_back(lm_behind);

    // Landmark IN FRONT of camera
    Landmark lm_front;
    lm_front.id = 2;
    lm_front.x = 0.f;
    lm_front.y = 0.f;
    lm_front.z = 10.f;
    problem.landmarks.landmarks.push_back(lm_front);

    // Observation for behind-camera landmark (should be pruned in Phase B)
    Observation2D3D obs_behind;
    obs_behind.landmark_id = 1;
    obs_behind.camera_id = 0;
    obs_behind.timestamp_ns = 1000000000ULL;
    obs_behind.keypoint.x = 320.f;
    obs_behind.keypoint.y = 240.f;
    problem.observations.push_back(obs_behind);

    // Observation for front landmark (should NOT be pruned)
    Observation2D3D obs_front;
    obs_front.landmark_id = 2;
    obs_front.camera_id = 0;
    obs_front.timestamp_ns = 1000000000ULL;
    obs_front.keypoint.x = 320.f;
    obs_front.keypoint.y = 240.f;
    problem.observations.push_back(obs_front);

    problem.optimize_poses = false;
    problem.optimize_landmarks = false;

    BundleAdjuster adjuster;
    Status s = adjuster.solve(problem, params, result);
    ASSERT_TRUE(s.ok) << s.message;
    EXPECT_TRUE(result.success);

    // per_observation contains ALL observations (including pruned).
    // num_observations counts only non-pruned observations used in optimization.
    EXPECT_EQ(result.per_observation.size(), 2u) << "Both observations in output";
    EXPECT_EQ(result.num_observations, 1u) << "Only non-pruned observations counted";

    // Verify which observation was pruned and which was kept
    bool found_pruned = false, found_active = false;
    for (const auto& por : result.per_observation) {
        if (por.landmark_id == 1) {
            EXPECT_TRUE(por.pruned) << "Behind-camera obs should be pruned";
            found_pruned = true;
        } else if (por.landmark_id == 2) {
            EXPECT_FALSE(por.pruned) << "Front obs should not be pruned";
            EXPECT_LT(por.err_px, 1.0f) << "Front obs has low error";
            found_active = true;
        }
    }
    EXPECT_TRUE(found_pruned) << "Behind-camera landmark (id=1) should be in output";
    EXPECT_TRUE(found_active) << "Front landmark (id=2) should be in output";
}

// Test high-error pruning.
// With high-error pruning enabled, observations with large initial reprojection
// error should be excluded from Phase B (not added as residual blocks).
TEST(SolverEdgeCases, HighErrorPruning) {
    BundleAdjustmentProblem problem;
    BundleAdjustmentParams params;
    params.max_iterations = 10;
    params.prune_high_error = true;
    params.prune_error_threshold_px = 5.0f;  // Pruning threshold (separate from inlier)
    params.min_observations_per_landmark = 1;
    params.verbose = false;
    BundleAdjustmentResult result;

    // Camera
    CameraIntrinsics ci;
    ci.camera_id = 0;
    ci.fx = 500.f; ci.fy = 500.f;
    ci.cx = 320.f; ci.cy = 240.f;
    problem.camera_rig.intrinsics.push_back(ci);

    CameraExtrinsics ce;
    ce.camera_id = 0;
    problem.camera_rig.extrinsics.push_back(ce);

    // Pose at origin
    VehiclePose vp;
    vp.timestamp_ns = 1000000000ULL;
    problem.vehicle_poses.poses_buffer.push_back(vp);

    // Landmark 1: at optical axis (projects to principal point)
    Landmark lm1;
    lm1.id = 1;
    lm1.x = 0.f; lm1.y = 0.f; lm1.z = 10.f;
    problem.landmarks.landmarks.push_back(lm1);

    // Landmark 2: offset from optical axis
    Landmark lm2;
    lm2.id = 2;
    lm2.x = 1.f; lm2.y = 0.f; lm2.z = 10.f;
    problem.landmarks.landmarks.push_back(lm2);

    // Observation 1: Good match (low error)
    Observation2D3D obs1;
    obs1.landmark_id = 1;
    obs1.camera_id = 0;
    obs1.timestamp_ns = 1000000000ULL;
    obs1.keypoint.x = 320.f;
    obs1.keypoint.y = 240.f;
    problem.observations.push_back(obs1);

    // Observation 2: Bad match (high error > 5px)
    // Landmark 2 at (1, 0, 10) projects to ~(370, 240), but we claim (500, 240)
    // -> error ~130 pixels, well above the 5px threshold
    Observation2D3D obs2;
    obs2.landmark_id = 2;
    obs2.camera_id = 0;
    obs2.timestamp_ns = 1000000000ULL;
    obs2.keypoint.x = 500.f;
    obs2.keypoint.y = 240.f;
    problem.observations.push_back(obs2);

    problem.optimize_poses = false;
    problem.optimize_landmarks = false;

    BundleAdjuster adjuster;
    Status s = adjuster.solve(problem, params, result);
    ASSERT_TRUE(s.ok) << s.message;
    EXPECT_TRUE(result.success);

    // The high-error observation was pruned in Phase B.
    // Pruned observations still appear in per_observation (marked with pruned=true),
    // but only non-pruned observations are counted in num_observations and stats.
    EXPECT_EQ(result.num_observations, 1u) << "Only non-pruned observations counted in stats";
    EXPECT_EQ(result.per_observation.size(), 2u) << "Both observations in output (pruned is marked)";
    EXPECT_EQ(result.num_pruned, 1u) << "One observation should be pruned";

    // Find the non-pruned observation and verify it's the good-match landmark
    const PerObservationResult* good_obs = nullptr;
    for (const auto& po : result.per_observation) {
        if (!po.pruned) { good_obs = &po; break; }
    }
    ASSERT_NE(good_obs, nullptr) << "Should have at least one non-pruned observation";
    EXPECT_EQ(good_obs->landmark_id, 1u)
        << "The non-pruned observation should be the good-match landmark (id=1)";
    EXPECT_LT(good_obs->err_px, 5.0f) << "Kept observation should have low error";

    // The mean error should only reflect the good observation, not the pruned outlier
    EXPECT_LT(result.avg_reproj_error_px, 5.0f) << "Mean should exclude pruned outlier";
}

