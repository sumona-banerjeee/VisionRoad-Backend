# onnx_to_trt.py - Convert ONNX model to TensorRT engine (FP32 for accuracy)
import tensorrt as trt
import os

ONNX_FILE = "models/pothole-detector.onnx"
ENGINE_FILE = "models/pothole-detector.engine"

print(f"TensorRT version: {trt.__version__}")
print(f"Converting {ONNX_FILE} to {ENGINE_FILE}...")
print("Using FP32 precision for maximum accuracy")

# Create builder and network
logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

# Parse ONNX model
print("Parsing ONNX model...")
with open(ONNX_FILE, 'rb') as model:
    if not parser.parse(model.read()):
        for error in range(parser.num_errors):
            print(f"Error {error}: {parser.get_error(error)}")
        raise RuntimeError("Failed to parse ONNX model")

print("ONNX model parsed successfully!")

# Create builder config
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB workspace

# DO NOT enable FP16 - keep FP32 for accuracy
# If you want FP16 later: config.set_flag(trt.BuilderFlag.FP16)
print("Using FP32 precision (no FP16)")

# Build engine
print("Building TensorRT engine (this may take a few minutes)...")
serialized_engine = builder.build_serialized_network(network, config)

if serialized_engine is None:
    raise RuntimeError("Failed to build TensorRT engine")

# Save engine
print(f"Saving engine to {ENGINE_FILE}...")
with open(ENGINE_FILE, 'wb') as f:
    f.write(serialized_engine)

print(f"TensorRT engine exported successfully to: {ENGINE_FILE}")
print(f"Engine size: {os.path.getsize(ENGINE_FILE) / (1024*1024):.2f} MB")
