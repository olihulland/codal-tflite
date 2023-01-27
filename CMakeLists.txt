include("${CODAL_UTILS_LOCATION}")

RECURSIVE_FIND_DIR(INCLUDE_DIRS "./inc" "*.h")
list(APPEND INCLUDE_DIRS "./third_party/")
list(APPEND INCLUDE_DIRS "./third_party/ruy")
list(APPEND INCLUDE_DIRS "./third_party/flatbuffers/include")
list(APPEND INCLUDE_DIRS "./third_party/gemmlowp/")
RECURSIVE_FIND_DIR(TENSORFLOW_DIRS "./inc" "*.h")
list(APPEND INCLUDE_DIRS TENSORFLOW_DIRS)

# In the case of flatbuffers we need to remove this directory or its string.h file with clash with stdlib <string.h>
list(FILTER INCLUDE_DIRS EXCLUDE REGEX "./third_party/flatbuffers/include/flatbuffers")

RECURSIVE_FIND_FILE(SOURCE_FILES "./" "*.cpp")
RECURSIVE_FIND_FILE(CC_FILES "./" "*.cc")
RECURSIVE_FIND_FILE(S_FILES "./" "*.s")
RECURSIVE_FIND_FILE(C_FILES "./" "*.c")
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