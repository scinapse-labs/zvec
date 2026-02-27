include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

find_package(Threads REQUIRED)

if(UNIX AND NOT APPLE)
    find_library(LIB_RT NAMES rt)
else()
    set(LIB_RT "")
endif()

git_version(GIT_SRCS_VER ${CMAKE_CURRENT_SOURCE_DIR})
file(GLOB_RECURSE ALL_SRCS *.cc *.c *.h)

set(EXTRA_LIBS ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})

if(UNIX AND NOT APPLE)
    list(APPEND EXTRA_LIBS ${LIB_RT})
endif()

#set(CMAKE_CXX_FLAGS "-march=sapphirerapids")
#set(CMAKE_C_FLAGS "-march=sapphirerapids")

set(MATH_MARCH_FLAG "-march=sapphirerapids")

# Apply custom march flags to math directories if specified
if(MATH_MARCH_FLAG)
    file(GLOB_RECURSE MATH_FILES 
        ${CMAKE_CURRENT_SOURCE_DIR}/math/*.cc 
        ${CMAKE_CURRENT_SOURCE_DIR}/math/*.c 
        ${CMAKE_CURRENT_SOURCE_DIR}/math_batch/*.cc 
        ${CMAKE_CURRENT_SOURCE_DIR}/math_batch/*.c
    )
    
    foreach(MATH_FILE ${MATH_FILES})
        set_source_files_properties(
            ${MATH_FILE}
            PROPERTIES
            COMPILE_FLAGS "${MATH_MARCH_FLAG}"
        )
    endforeach()
endif()

cc_library(
    NAME zvec_ailego STATIC STRICT PACKED
    SRCS    ${ALL_SRCS}
    LIBS    ${EXTRA_LIBS}
            Arrow::arrow_static
            Arrow::parquet_static
    VERSION "${GIT_SRCS_VER}"
)
