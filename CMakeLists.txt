include("${CODAL_UTILS_LOCATION}")

# codal-tflite api
RECURSIVE_FIND_DIR(INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/libraries/codal-tflite/inc" "*.h")

# tensorflow lite
list(APPEND INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/libraries/codal-tflite/")

# third party - it's a bit weird so have specifics
list(APPEND INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/libraries/codal-tflite/third_party/")
list(APPEND INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/libraries/codal-tflite/third_party/ruy")
list(APPEND INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/libraries/codal-tflite/third_party/flatbuffers/include")
list(APPEND INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/libraries/codal-tflite/third_party/gemmlowp/")
list(APPEND INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/libraries/codal-tflite/third_party/kissfft/")

# In the case of flatbuffers we need to remove this directory or its string.h file with clash with stdlib <string.h>
list(FILTER INCLUDE_DIRS EXCLUDE REGEX "${PROJECT_SOURCE_DIR}/libraries/codal-tflite/third_party/flatbuffers/include/flatbuffers")

RECURSIVE_FIND_FILE(SOURCE_FILES "${PROJECT_SOURCE_DIR}/libraries/codal-tflite/" "*.cpp")
RECURSIVE_FIND_FILE(CC_FILES "${PROJECT_SOURCE_DIR}/libraries/codal-tflite/" "*.cc")
RECURSIVE_FIND_FILE(S_FILES "${PROJECT_SOURCE_DIR}/libraries/codal-tflite/" "*.s")
RECURSIVE_FIND_FILE(C_FILES "${PROJECT_SOURCE_DIR}/libraries/codal-tflite/" "*.c")
list(APPEND SOURCE_FILES ${CC_FILES})
list(APPEND SOURCE_FILES ${S_FILES})
list(APPEND SOURCE_FILES ${C_FILES})

# add them
include_directories(${INCLUDE_DIRS})

add_library(codal-tflite
    ${SOURCE_FILES}
)

target_link_libraries(
    codal-tflite
    codal-core
)

target_include_directories(codal-tflite PUBLIC ${INCLUDE_DIRS})