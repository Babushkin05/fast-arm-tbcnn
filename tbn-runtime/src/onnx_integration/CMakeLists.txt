# ONNX Integration Sources

# Use simple parser for now (without ONNX library dependency)
# When ONNX library is available, switch to onnx_parser.cpp
set(ONNX_SOURCES
    onnx_parser_simple.cpp
)

# Add sources to parent library
# These will be compiled as part of the main tbn library
# target_sources(tbn PRIVATE ${ONNX_SOURCES})