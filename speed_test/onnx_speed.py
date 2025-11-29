import os, glob, sys, random
from time import perf_counter

import torch
import numpy as np
import onnxruntime as ort
from PIL import Image


# onnx inference
def inference_onnx_model(model_path, img_path, input_shape=(1, 3, 224, 224)):
    h, w = input_shape[2], input_shape[3]
    img = Image.open(img_path).resize((w, h)).convert('L' if input_shape[1] == 1 else 'RGB')
    img = np.array(img).reshape(input_shape).astype(np.float32) / 255.0

    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    # output_name = session.get_outputs()[0].name
    # Run the model
    output = session.run(None, {input_name: img})
    conf = np.max(np.exp(output[0]), axis=2).squeeze()
    output = np.argmax(output[0], axis=2).squeeze()
    return output, conf


def test_onnx_model_speed(model_path, input_shape, warm_up=10, test=100, force_cpu=True):
    options = ort.SessionOptions()
    # options.enable_profiling = True
    options.intra_op_num_threads = 4
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    provider_options = ['CPUExecutionProvider'] if force_cpu else ort.get_available_providers()
    print(f'providers: {provider_options}')
    session = ort.InferenceSession(model_path, options, providers=provider_options)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_data = {
        input_name: np.random.randn(*input_shape).astype(np.float32)
    }

    times = []
    total_iterations = warm_up + test
    for i in range(total_iterations):

        if i < warm_up:
            session.run(None, input_data)
            continue

        start = perf_counter()
        session.run(None, input_data)
        end = perf_counter()
        times.append(end - start)

    times = np.array(times)
    times = np.sort(times)[int(test * 0.2):int(test * 0.8)]
    avg_time = np.mean(times)
    print(f'Min time: {min(times)*1000:.2f} ms')
    print(f'Max time: {max(times)*1000:.2f} ms')
    print(f'\33[32mAvg time: {avg_time * 1000:.2f} ms\33[0m')


if __name__ == '__main__':
    model_path = 'model.onnx'
    input_shape = (1, 3, 224, 224)

    test_onnx_model_speed(model_path, input_shape, warm_up=20, test=100)

    image_path = glob.glob('data/*.jpg')
    # shuffle(image_path)
    image_path = random.choice(image_path)
    print("image_path: ", image_path)
    output, conf = inference_onnx_model(model_path, image_path, input_shape)
    print("conf: ", conf)
    print("output: ", output)
