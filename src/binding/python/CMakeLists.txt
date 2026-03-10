include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

set(SRC_LISTS
        binding.cc
        model/python_collection.cc
        model/python_doc.cc
        model/param/python_param.cc
        model/schema/python_schema.cc
        model/common/python_config.cc
        typing/python_type.cc
)

pybind11_add_module(_zvec ${SRC_LISTS})


if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(_zvec PRIVATE
            # Link all zvec internal objects (includes all algorithms)
            "$<TARGET_OBJECTS:zvec_ailego_objects>"
            "$<TARGET_OBJECTS:zvec_core_objects>"
            "$<TARGET_OBJECTS:zvec_db_objects>"
            # Link third-party static libraries
            Arrow::arrow_static
            Arrow::arrow_compute
            Arrow::arrow_dataset
            Arrow::arrow_acero
            Arrow::parquet_static
            rocksdb
            glog
            libprotobuf
            antlr4
            roaring
            sparsehash
            magic_enum
    )
    target_link_options(_zvec PRIVATE
            "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map"
    )
elseif(APPLE)
    # Link all zvec internal objects (includes all algorithms)
  set(ZVEC_PYTHON_STATIC_DEPS
        "$<TARGET_OBJECTS:zvec_ailego_objects>"
        "$<TARGET_OBJECTS:zvec_core_objects>"
        "$<TARGET_OBJECTS:zvec_db_objects>"
    )
    
    target_link_libraries(_zvec PRIVATE
            ${ZVEC_PYTHON_STATIC_DEPS}
            # Link third-party static libraries
            Arrow::arrow_static
            Arrow::arrow_compute
            Arrow::arrow_dataset
            Arrow::arrow_acero
            Arrow::parquet_static
            rocksdb
            glog
            libprotobuf
            antlr4
            roaring
            sparsehash
            magic_enum
    )
    target_link_options(_zvec PRIVATE
            "-Wl,-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/exports.mac"
    )
endif()

target_include_directories(_zvec PRIVATE ${PYBIND11_INCLUDE_DIR} ${PROJECT_ROOT_DIR}/src ${PROJECT_ROOT_DIR}/src/binding/python/include)
