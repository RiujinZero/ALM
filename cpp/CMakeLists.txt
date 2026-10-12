cmake_minimum_required(VERSION 3.20)
project(alm_native LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(alm_native SHARED src/c_api.cpp)
target_include_directories(alm_native PUBLIC include)

if(MSVC)
  # /fp:precise and no /fp:fast: the backoff chain must land on exactly the
  # same IEEE-754 bits as the Python reference, so neither reassociation nor
  # FMA contraction is acceptable here.
  target_compile_options(alm_native PRIVATE
    /O2 /Oi /Ot /GL /fp:precise /fp:except- /EHsc /W4 /permissive-)
  target_link_options(alm_native PRIVATE /LTCG)
else()
  target_compile_options(alm_native PRIVATE
    -O3 -fno-fast-math -ffp-contract=off -Wall -Wextra -msse2)
endif()

set_target_properties(alm_native PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# A separate DLL from alm_native on purpose: alm_native stays loaded inside
# long-running processes (the byte-cortex 12h run), which locks the file on
# Windows. Building vision into the same target would fail to link -- or
# worse, require killing a live experiment -- every time this module changes.
add_library(alm_vision SHARED src/vision_c_api.cpp)
target_include_directories(alm_vision PUBLIC include)

# /arch:AVX2 (unlike alm_native, which stays SSE2-baseline-portable for its
# bit-exact contract): activations() is O(live_assemblies * synapses) and
# live_assemblies grows to tens of thousands over a long run, so the
# gather-multiply-accumulate there is the dominant cost at scale. This
# module's contract is already numerically-close, not bit-exact (see
# vision.hpp and cpp/README.md), so the reassociation from AVX2 gather +
# float32 accumulation is acceptable -- it stays within the float32 noise
# floor already measured by tests/test_native_vision_matches_python.py.
if(MSVC)
  target_compile_options(alm_vision PRIVATE
    /O2 /Oi /Ot /GL /fp:precise /fp:except- /EHsc /W4 /permissive- /arch:AVX2)
  target_link_options(alm_vision PRIVATE /LTCG)
else()
  target_compile_options(alm_vision PRIVATE
    -O3 -fno-fast-math -ffp-contract=off -Wall -Wextra -mavx2)
endif()

set_target_properties(alm_vision PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

include(CTest)
if(BUILD_TESTING)
  add_executable(alm_row_arena_test tests/row_arena_test.cpp)
  target_include_directories(alm_row_arena_test PRIVATE include)
  add_test(NAME alm_row_arena_test COMMAND alm_row_arena_test)
endif()

