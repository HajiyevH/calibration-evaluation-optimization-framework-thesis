cmake_minimum_required(VERSION 3.20)
project(ThesisBundleAdjustment LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies (using vcpkg)
find_package(Ceres CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED COMPONENTS core calib3d)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)

add_subdirectory(calibri/engine)
add_subdirectory(calibri/app)
add_subdirectory(tests)
