import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
import os
import sys


# Suppress some TensorRT warning messages, can be changed to INFO or VERBOSE
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    # Check if we are using the new API (TensorRT 10+)
    use_new_api = not hasattr(engine, "num_bindings")
    num_tensors = engine.num_io_tensors if use_new_api else engine.num_bindings

    for i in range(num_tensors):
        if use_new_api:
            name = engine.get_tensor_name(i)
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        else:
            name = engine.get_binding_name(i)
            shape = engine.get_binding_shape(i)
            dtype = trt.nptype(engine.get_binding_dtype(i))
            is_input = engine.binding_is_input(i)
        
        # Handle dynamic batch (if there is a -1 dimension)
        # Simple handling here: if -1, temporarily set to 1 or actual batch size
        dims = list(shape)
        if len(dims) > 0 and dims[0] == -1:
            dims[0] = 1
            shape = tuple(dims)
            
        size = trt.volume(shape)
        # trt.nptype might return None in some versions, need safe handling
        if dtype is None:
            dtype = np.float32

        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))

        if is_input:
            inputs.append({"host": host_mem, "device": device_mem, "shape": shape, "name": name})
            print(f"Input:  {name} {shape}")
        else:
            outputs.append({"host": host_mem, "device": device_mem, "shape": shape, "name": name})
            print(f"Output: {name} {shape}")

    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream):
    # Copy input data Host -> Device
    for inp in inputs:
        cuda.memcpy_htod_async(inp["device"], inp["host"], stream)

    # Execute inference
    if hasattr(context, "execute_async_v2"):
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    else:
        # TensorRT 10+ fallback to v3
        for inp in inputs:
            context.set_tensor_address(inp["name"], int(inp["device"]))
        for out in outputs:
            context.set_tensor_address(out["name"], int(out["device"]))
        context.execute_async_v3(stream_handle=stream.handle)

    # Copy output data Device -> Host
    for out in outputs:
        cuda.memcpy_dtoh_async(out["host"], out["device"], stream)

    stream.synchronize()
    return [out["host"] for out in outputs]


def benchmark(context, bindings, inputs, outputs, stream, warmup=20, runs=100):
    print(f"Starting benchmark (warmup={warmup}, runs={runs})...")
    for _ in range(warmup):
        do_inference(context, bindings, inputs, outputs, stream)

    times = []
    for _ in range(runs):
        start = time.time()
        do_inference(context, bindings, inputs, outputs, stream)
        times.append((time.time() - start) * 1000)
    
    avg = np.mean(times)
    fps = 1000.0 / avg
    return {"avg_ms": avg, "min_ms": np.min(times), "max_ms": np.max(times), "fps": fps}


def main():
    # Please ensure the path is correct
    engine_path = "model.trt"

    engine = None

    # 1. Load Engine
    if not os.path.exists(engine_path):
        print(f"Error: Engine file not found at {engine_path}")
        sys.exit(1)

    print(f"Loading existing engine from {engine_path}...")
    runtime = trt.Runtime(TRT_LOGGER)
    try:
        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
    except Exception as e:
        print(f"Error loading engine: {e}")
        sys.exit(1)

    # 2. Check if Engine was successfully created
    if engine is None:
        print("CRITICAL ERROR: Engine is None. Exiting.")
        sys.exit(1)

    print("Engine loaded successfully.")

    # 3. Create execution context
    context = engine.create_execution_context()
    if context is None:
        print("Error: Failed to create execution context.")
        sys.exit(1)

    try:
        inputs, outputs, bindings, stream = allocate_buffers(engine)
    except Exception as e:
        print(f"Error allocating buffers: {e}")
        # Common error is API incompatibility due to TensorRT version differences
        return

    # 4. Fill Dummy Data
    # Use the actually allocated shape to prevent reshape errors
    input_shape = inputs[0]["shape"]
    dummy_input = np.random.randn(*input_shape).astype(np.float32).ravel()
    np.copyto(inputs[0]["host"], dummy_input) # Safer copy method

    # 5. Benchmark
    stats = benchmark(context, bindings, inputs, outputs, stream)
    print(f"\nAvg latency: {stats['avg_ms']:.2f} ms, FPS: {stats['fps']:.2f}")

    # 6. Inference Test
    out = do_inference(context, bindings, inputs, outputs, stream)[0]
    print("Output sample:", out[:10], "...")


if __name__ == "__main__":
    main()
