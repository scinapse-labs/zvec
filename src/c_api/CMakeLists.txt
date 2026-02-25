# Copyright 2025-present the zvec project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# C API library source files
set(ZVEC_C_API_SOURCES
    c_api.cc
)

# C API library header files
set(ZVEC_C_API_HEADERS
    ${PROJECT_SOURCE_DIR}/src/include/zvec/c_api.h
)

# Create shared library
add_library(zvec_c_api SHARED
    ${ZVEC_C_API_SOURCES}
    ${ZVEC_C_API_HEADERS}
)

# Set library properties
set_target_properties(zvec_c_api PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME "zvec_c_api"
)

find_package(Threads REQUIRED)

# Link dependencies - Add force_load flag to ensure static initialization is executed
if(APPLE)
    target_link_libraries(zvec_c_api
        PRIVATE
            "-Wl,-force_load" "$<TARGET_FILE:zvec_core>"
            zvec_db
            Threads::Threads
    )
else()
    target_link_libraries(zvec_c_api
        PRIVATE
            "-Wl,--whole-archive" zvec_core "-Wl,--no-whole-archive"
            zvec_db
            Threads::Threads
    )
endif()

# Include directories
target_include_directories(zvec_c_api
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)

# Compile options
target_compile_options(zvec_c_api PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wpedantic>
)

# Installation rules
install(TARGETS zvec_c_api
    EXPORT zvecTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(FILES ${PROJECT_SOURCE_DIR}/src/include/zvec/c_api.h
    DESTINATION include/zvec
)

# Create static library version (optional)
if(BUILD_STATIC_LIBS)
    add_library(zvec_c_api_static STATIC
        ${ZVEC_C_API_SOURCES}
        ${ZVEC_C_API_HEADERS}
    )
    
    set_target_properties(zvec_c_api_static PROPERTIES
        OUTPUT_NAME "zvec_c_api"
    )
    
    # Static library also adds force_load flag
    if(APPLE)
        target_link_libraries(zvec_c_api_static
            PRIVATE
                "-Wl,-force_load" "$<TARGET_FILE:zvec_core>"
                zvec_db
                Threads::Threads
        )
    else()
        target_link_libraries(zvec_c_api_static
            PRIVATE
                "-Wl,--whole-archive" zvec_core "-Wl,--no-whole-archive"
                zvec_db
                Threads::Threads
        )
    endif()
    
    target_include_directories(zvec_c_api_static
        PUBLIC
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${PROJECT_SOURCE_DIR}/src
    )
    
    install(TARGETS zvec_c_api_static
        EXPORT zvecTargets
        ARCHIVE DESTINATION lib
        INCLUDES DESTINATION include
    )
endif()