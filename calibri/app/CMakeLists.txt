add_executable(ba_cli main.cc)

target_link_libraries(ba_cli PRIVATE ba_engine)

# Increase stack size to 8MB for large datasets (168K+ observations).
# MSVC default is 1MB which can overflow during JSON serialization.
if(MSVC)
    target_link_options(ba_cli PRIVATE /STACK:8388608)
endif()
