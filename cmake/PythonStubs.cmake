# CMake function to generate/update type stubs for Python bindings
# Add this to your main CMakeLists.txt or include it as a separate file

# Function to generate Python type stubs
function(generate_python_stubs)
    # Arguments
    set(oneValueArgs TARGET BINDINGS_SOURCE OUTPUT_FILE)
    set(multiValueArgs DEPENDENCIES)
    cmake_parse_arguments(STUBS "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Default values
    if(NOT STUBS_OUTPUT_FILE)
        set(STUBS_OUTPUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/qforte/qforte.pyi")
    endif()

    if(NOT STUBS_BINDINGS_SOURCE)
        set(STUBS_BINDINGS_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/qforte/bindings.cc")
    endif()

    # Find Python
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    # Path to the stub generation script
    set(STUB_GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify_pyi.py")

    # Custom target to generate stubs
    add_custom_target(${STUBS_TARGET}_stubs
        COMMAND ${Python3_EXECUTABLE} ${STUB_GENERATOR}
        DEPENDS ${STUBS_BINDINGS_SOURCE} ${STUBS_DEPENDENCIES}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Verifying Python type stubs for ${STUBS_TARGET}"
        VERBATIM
    )

    # Make the main target depend on stub generation (optional)
    if(TARGET ${STUBS_TARGET})
        add_dependencies(${STUBS_TARGET} ${STUBS_TARGET}_stubs)
    endif()

    # Custom command to update stubs when bindings change
    add_custom_command(
        OUTPUT ${STUBS_OUTPUT_FILE}.timestamp
        COMMAND ${Python3_EXECUTABLE} ${STUB_GENERATOR}
        COMMAND ${CMAKE_COMMAND} -E touch ${STUBS_OUTPUT_FILE}.timestamp
        DEPENDS ${STUBS_BINDINGS_SOURCE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Checking type stubs are up to date"
        VERBATIM
    )

    # Add a timestamp dependency to ensure stubs are checked
    add_custom_target(${STUBS_TARGET}_stubs_check
        DEPENDS ${STUBS_OUTPUT_FILE}.timestamp
    )

endfunction()

# Function to add stub generation to pybind11 modules
function(add_pybind11_stubs TARGET)
    # Add stub generation for pybind11 module
    generate_python_stubs(
        TARGET ${TARGET}
        BINDINGS_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/qforte/bindings.cc"
        OUTPUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/qforte/qforte.pyi"
    )
endfunction()
