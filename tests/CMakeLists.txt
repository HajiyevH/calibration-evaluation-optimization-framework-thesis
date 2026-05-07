enable_testing()

add_executable(ba_tests
    test_pinhole_radtan.cc
    test_pinhole_fisheye.cc
    test_contracts_bridge.cc
    test_json_codec.cc
    test_reprojection_cost.cc
    test_e2e_synthetic.cc
    test_solver_edge_cases.cc
)

target_link_libraries(ba_tests PRIVATE
    ba_engine
    GTest::gtest
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(ba_tests)
