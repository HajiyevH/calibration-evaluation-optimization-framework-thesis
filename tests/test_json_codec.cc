#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
#include <cstdio>
#include "engine/io/json_codec.h"
#include "engine/io/contracts_bridge.h"
#include "engine/contracts/types.h"

using namespace ba;
namespace fs = std::filesystem;

namespace {

// RAII guard: deletes the temp file on destruction, even if a test assertion fails.
struct TempFileGuard {
    std::string path;
    ~TempFileGuard() { fs::remove(path); }
};

// Write a string to a temp file and return the path
std::string writeTempJson(const std::string& content, const std::string& name) {
    fs::path dir = fs::temp_directory_path() / "ba_test";
    fs::create_directories(dir);
    fs::path p = dir / name;
    std::ofstream ofs(p);
    ofs << content;
    ofs.close();
    return p.string();
}

const char* MINIMAL_PROBLEM = R"({
  "cameras": [
    {
      "id": 0,
      "model": "Pinhole",
      "width": 1280,
      "height": 800,
      "intrinsics": { "fx": 950.0, "fy": 950.0, "cx": 640.0, "cy": 400.0 },
      "distortion": { "k1": -0.1, "k2": 0.01, "p1": 0.0, "p2": 0.0, "k3": 0.0 },
      "extrinsic": { "q": [1,0,0,0], "t": [0,0,0], "fixed": true }
    }
  ],
  "poses": [
    { "timestamp_ns": 1000000000, "q": [1,0,0,0], "t": [0,0,0] },
    { "timestamp_ns": 2000000000, "q": [0.9999,0,0,0.014], "t": [0.5,0.0,0.01] }
  ],
  "landmarks": [
    { "id": 1, "X": 5.5, "Y": -1.2, "Z": 12.3 },
    { "id": 2, "X": 4.1, "Y": 0.7, "Z": 18.8 }
  ],
  "observations": [
    { "landmark_id": 1, "camera_id": 0, "timestamp_ns": 1000000000, "u": 812.3, "v": 512.1 },
    { "landmark_id": 2, "camera_id": 0, "timestamp_ns": 2000000000, "u": 805.8, "v": 510.9 }
  ],
  "flags": { "opt_intrinsics": false, "opt_extrinsics": false, "opt_poses": true, "opt_landmarks": true },
  "robust": { "type": "Huber", "scale": 1.0 }
})";

} // anonymous namespace

// Parse a valid problem.json
TEST(JsonCodec, ParseMinimalProblem) {
    std::string path = writeTempJson(MINIMAL_PROBLEM, "test_minimal.json");
    TempFileGuard guard{path};

    BundleAdjustmentProblem problem;
    BundleAdjustmentParams params;
    Status s = loadProblemJson(path, problem, params);

    ASSERT_TRUE(s.ok) << s.message;

    EXPECT_EQ(problem.camera_rig.intrinsics.size(), 1u);
    EXPECT_EQ(problem.camera_rig.extrinsics.size(), 1u);
    EXPECT_EQ(problem.vehicle_poses.poses_buffer.size(), 2u);
    EXPECT_EQ(problem.landmarks.landmarks.size(), 2u);
    EXPECT_EQ(problem.observations.size(), 2u);

    // Check camera intrinsics
    EXPECT_FLOAT_EQ(problem.camera_rig.intrinsics[0].fx, 950.f);
    EXPECT_FLOAT_EQ(problem.camera_rig.intrinsics[0].params[0], -0.1f);  // k1

    // Check flags
    EXPECT_FALSE(problem.optimize_intrinsics);
    EXPECT_FALSE(problem.optimize_extrinsics);
    EXPECT_TRUE(problem.optimize_poses);
    EXPECT_TRUE(problem.optimize_landmarks);

    // Check robust params
    EXPECT_EQ(params.loss_type, BundleAdjustmentParams::LossType::HUBER);
    EXPECT_FLOAT_EQ(params.loss_scale, 1.0f);

    // Check observations
    EXPECT_FLOAT_EQ(problem.observations[0].keypoint.x, 812.3f);
    EXPECT_FLOAT_EQ(problem.observations[0].keypoint.y, 512.1f);

    // Check landmarks
    EXPECT_EQ(problem.landmarks.landmarks[0].id, 1u);
    EXPECT_FLOAT_EQ(problem.landmarks.landmarks[0].x, 5.5f);
}

// Parse + serialize round-trip
TEST(JsonCodec, ParseSerializeRoundTrip) {
    std::string path = writeTempJson(MINIMAL_PROBLEM, "test_roundtrip_in.json");
    TempFileGuard guard_in{path};

    BundleAdjustmentProblem problem;
    BundleAdjustmentParams params;
    Status s = loadProblemJson(path, problem, params);
    ASSERT_TRUE(s.ok) << s.message;

    // Build a mock result
    BundleAdjustmentResult result;
    result.success = true;
    result.iterations_used = 10;
    result.initial_cost = 100.f;
    result.final_cost = 5.f;
    result.num_observations = 2;
    result.num_inliers = 2;
    result.num_outliers = 0;
    result.avg_reproj_error_px = 0.5f;
    result.max_reproj_error_px = 0.8f;
    result.optimized_poses = problem.vehicle_poses;
    result.optimized_landmarks = problem.landmarks;
    result.optimized_camera_rig = problem.camera_rig;

    // Add per-observation results
    for (const auto& obs : problem.observations) {
        PerObservationResult por;
        por.timestamp_ns = obs.timestamp_ns;
        por.camera_id = obs.camera_id;
        por.landmark_id = obs.landmark_id;
        por.u_meas = obs.keypoint.x;
        por.v_meas = obs.keypoint.y;
        por.u_hat = obs.keypoint.x + 0.3f;
        por.v_hat = obs.keypoint.y - 0.2f;
        por.err_px = 0.36f;
        por.inlier = true;
        result.per_observation.push_back(por);
    }

    std::string out_path = (fs::temp_directory_path() / "ba_test" / "test_roundtrip_out.json").string();
    TempFileGuard guard_out{out_path};
    Status ws = writeResultJson(out_path, result, problem);
    ASSERT_TRUE(ws.ok) << ws.message;

    // Read result back and check key fields
    nlohmann::json j;
    {
        std::ifstream ifs(out_path);
        ASSERT_TRUE(ifs.is_open());
        ifs >> j;
    } // ifstream closed here before fs::remove

    EXPECT_TRUE(j["summary"]["success"].get<bool>());
    EXPECT_EQ(j["summary"]["iterations"].get<int>(), 10);
    EXPECT_FLOAT_EQ(j["summary"]["initial_cost"].get<float>(), 100.f);
    EXPECT_EQ(j["residuals"].size(), 2u);
    EXPECT_EQ(j["final_params"]["poses"].size(), 2u);
    EXPECT_EQ(j["final_params"]["landmarks"].size(), 2u);
}

// Missing required camera fields (intrinsics, q, t) should return an error.
// A camera without intrinsics or extrinsics is a user mistake, not an optional field.
TEST(JsonCodec, MissingRequiredFieldsReturnsError) {
    const char* incomplete = R"({
      "cameras": [{ "id": 0, "model": "Pinhole" }],
      "poses": [{ "timestamp_ns": 1000 }],
      "landmarks": [{ "id": 1 }],
      "observations": [{ "landmark_id": 1, "camera_id": 0, "timestamp_ns": 1000 }]
    })";

    std::string path = writeTempJson(incomplete, "test_missing_required.json");
    TempFileGuard guard{path};
    BundleAdjustmentProblem problem;
    BundleAdjustmentParams params;
    Status s = loadProblemJson(path, problem, params);

    EXPECT_FALSE(s.ok);
}

// Quaternion convention: JSON [w,x,y,z] should map correctly
TEST(JsonCodec, QuaternionConventionFromJson) {
    // 90-degree rotation about Z: w=cos(45), x=0, y=0, z=sin(45)
    const char* json_str = R"({
      "cameras": [{
        "id": 0,
        "intrinsics": { "fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0 }
      }],
      "poses": [{
        "timestamp_ns": 1000,
        "q": [0.7071068, 0, 0, 0.7071068],
        "t": [1.0, 2.0, 3.0]
      }],
      "landmarks": [],
      "observations": []
    })";

    std::string path = writeTempJson(json_str, "test_quat_convention.json");
    TempFileGuard guard{path};
    BundleAdjustmentProblem problem;
    BundleAdjustmentParams params;
    Status s = loadProblemJson(path, problem, params);
    ASSERT_TRUE(s.ok) << s.message;

    Eigen::Quaterniond q;
    Eigen::Vector3d t;
    PoseSE3_To_qt(problem.vehicle_poses.poses_buffer[0].pose, q, t);

    // Should be approximately a 90-degree rotation about Z
    EXPECT_NEAR(q.w(), 0.7071068, 1e-4);
    EXPECT_NEAR(q.z(), 0.7071068, 1e-4);
    EXPECT_NEAR(q.x(), 0.0, 1e-4);
    EXPECT_NEAR(q.y(), 0.0, 1e-4);
    EXPECT_NEAR(t.x(), 1.0, 1e-4);
    EXPECT_NEAR(t.y(), 2.0, 1e-4);
    EXPECT_NEAR(t.z(), 3.0, 1e-4);
}

// Multi-camera rig: two cameras with different intrinsics and extrinsics
TEST(JsonCodec, ParseMultiCameraRig) {
    const char* json_str = R"({
      "cameras": [
        {
          "id": 0,
          "model": "Pinhole",
          "width": 1920,
          "height": 1080,
          "intrinsics": { "fx": 1200.0, "fy": 1200.0, "cx": 960.0, "cy": 540.0 },
          "distortion": { "k1": -0.05, "k2": 0.002, "p1": 0.001, "p2": -0.001, "k3": 0.0 },
          "extrinsic": { "q": [1,0,0,0], "t": [0,0,0] }
        },
        {
          "id": 1,
          "model": "Pinhole",
          "width": 1280,
          "height": 800,
          "intrinsics": { "fx": 850.0, "fy": 855.0, "cx": 640.0, "cy": 400.0 },
          "distortion": { "k1": -0.12, "k2": 0.015, "p1": 0.0, "p2": 0.0, "k3": 0.001 },
          "extrinsic": { "q": [0.7071068, 0, 0.7071068, 0], "t": [0.5, 0.0, -0.1] }
        }
      ],
      "poses": [
        { "timestamp_ns": 1000, "q": [1,0,0,0], "t": [0,0,0] }
      ],
      "landmarks": [
        { "id": 1, "X": 5.0, "Y": 0.0, "Z": 10.0 }
      ],
      "observations": [
        { "landmark_id": 1, "camera_id": 0, "timestamp_ns": 1000, "u": 500.0, "v": 300.0 },
        { "landmark_id": 1, "camera_id": 1, "timestamp_ns": 1000, "u": 600.0, "v": 350.0 }
      ],
      "flags": { "opt_poses": true, "opt_landmarks": true }
    })";

    std::string path = writeTempJson(json_str, "test_multi_camera.json");
    TempFileGuard guard{path};
    BundleAdjustmentProblem problem;
    BundleAdjustmentParams params;
    Status s = loadProblemJson(path, problem, params);
    ASSERT_TRUE(s.ok) << s.message;

    // Two cameras parsed
    ASSERT_EQ(problem.camera_rig.intrinsics.size(), 2u);
    ASSERT_EQ(problem.camera_rig.extrinsics.size(), 2u);

    // Camera 0 intrinsics
    const auto& ci0 = problem.camera_rig.intrinsics[0];
    EXPECT_EQ(ci0.camera_id, 0);
    EXPECT_FLOAT_EQ(ci0.fx, 1200.f);
    EXPECT_FLOAT_EQ(ci0.fy, 1200.f);
    EXPECT_FLOAT_EQ(ci0.cx, 960.f);
    EXPECT_FLOAT_EQ(ci0.cy, 540.f);
    EXPECT_EQ(ci0.image_width, 1920);
    EXPECT_EQ(ci0.image_height, 1080);
    EXPECT_FLOAT_EQ(ci0.params[0], -0.05f);  // k1
    EXPECT_FLOAT_EQ(ci0.params[1], 0.002f);  // k2
    EXPECT_FLOAT_EQ(ci0.params[2], 0.001f);  // p1
    EXPECT_FLOAT_EQ(ci0.params[3], -0.001f); // p2
    EXPECT_FLOAT_EQ(ci0.params[4], 0.0f);    // k3

    // Camera 1 intrinsics
    const auto& ci1 = problem.camera_rig.intrinsics[1];
    EXPECT_EQ(ci1.camera_id, 1);
    EXPECT_FLOAT_EQ(ci1.fx, 850.f);
    EXPECT_FLOAT_EQ(ci1.fy, 855.f);
    EXPECT_FLOAT_EQ(ci1.cx, 640.f);
    EXPECT_FLOAT_EQ(ci1.cy, 400.f);
    EXPECT_EQ(ci1.image_width, 1280);
    EXPECT_EQ(ci1.image_height, 800);
    EXPECT_FLOAT_EQ(ci1.params[0], -0.12f);  // k1
    EXPECT_FLOAT_EQ(ci1.params[1], 0.015f);  // k2
    EXPECT_FLOAT_EQ(ci1.params[4], 0.001f);  // k3

    // Camera 0 extrinsics: identity rotation, zero translation
    const auto& ce0 = problem.camera_rig.extrinsics[0];
    EXPECT_EQ(ce0.camera_id, 0);
    Eigen::Quaterniond q0;
    Eigen::Vector3d t0;
    PoseSE3_To_qt(ce0.T_vehicle_to_camera, q0, t0);
    EXPECT_NEAR(q0.w(), 1.0, 1e-6);
    EXPECT_NEAR(q0.x(), 0.0, 1e-6);
    EXPECT_NEAR(q0.y(), 0.0, 1e-6);
    EXPECT_NEAR(q0.z(), 0.0, 1e-6);
    EXPECT_NEAR(t0.x(), 0.0, 1e-6);
    EXPECT_NEAR(t0.y(), 0.0, 1e-6);
    EXPECT_NEAR(t0.z(), 0.0, 1e-6);

    // Camera 1 extrinsics: 90-deg rotation about Y, translated [0.5, 0, -0.1]
    // JSON q = [w=0.7071068, x=0, y=0.7071068, z=0]
    const auto& ce1 = problem.camera_rig.extrinsics[1];
    EXPECT_EQ(ce1.camera_id, 1);
    Eigen::Quaterniond q1;
    Eigen::Vector3d t1;
    PoseSE3_To_qt(ce1.T_vehicle_to_camera, q1, t1);
    EXPECT_NEAR(q1.w(), 0.7071068, 1e-4);
    EXPECT_NEAR(q1.x(), 0.0, 1e-4);
    EXPECT_NEAR(q1.y(), 0.7071068, 1e-4);
    EXPECT_NEAR(q1.z(), 0.0, 1e-4);
    EXPECT_NEAR(t1.x(), 0.5, 1e-6);
    EXPECT_NEAR(t1.y(), 0.0, 1e-6);
    EXPECT_NEAR(t1.z(), -0.1, 1e-6);

    // Observations reference both cameras
    ASSERT_EQ(problem.observations.size(), 2u);
    EXPECT_EQ(problem.observations[0].camera_id, 0);
    EXPECT_EQ(problem.observations[1].camera_id, 1);
}

