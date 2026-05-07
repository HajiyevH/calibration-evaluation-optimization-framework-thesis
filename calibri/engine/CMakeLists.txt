add_library(ba_engine STATIC
    cameras/pinhole_radtan.cc
    io/contracts_bridge.cc
    io/json_codec.cc
    solver/bundle_adjuster.cc
)

target_include_directories(ba_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(ba_engine PUBLIC
    Ceres::ceres
    Eigen3::Eigen
    opencv_core
    opencv_calib3d
    nlohmann_json::nlohmann_json
)
