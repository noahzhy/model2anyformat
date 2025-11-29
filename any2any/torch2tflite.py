import os, sys, glob
from pathlib import Path
from typing import Any, Tuple, Dict, Union

import yaml
import torch
import numpy as np
import ai_edge_torch
from PIL import Image
from ai_edge_quantizer import quantizer, recipe
from ai_edge_quantizer.utils import tfl_interpreter_utils

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def convert_to_tflite(
    torch_model: torch.nn.Module,
    save_path: str,
    quant: bool = True,
    quant_type: str = "static",
    calibration_path: str = "",
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
) -> str:
    """Convert a PyTorch model to TFLite format and save it.

    Args:
        torch_model: The PyTorch model to convert.
        save_path: The path to save the TFLite model.
        quant: Whether to apply quantization.
        quant_type: The type of quantization to apply. Options are "static" or "dynamic".
    Returns:
        The path to the saved TFLite model.
    """
    edge_model = ai_edge_torch.convert(
        torch_model.eval(),
        (torch.randn(input_shape),),
    )
    edge_model.export(save_path)

    if quant:
        qt = quantizer.Quantizer(save_path)
        if quant_type == "static":
            qt.load_quantization_recipe(recipe.static_wi8_ai8())
            calibration_result = (
                qt.calibrate(_get_calibration_data(
                    save_path,
                    calibration_path,
                    input_shape,
                )) if qt.need_calibration else None
            )
            qt = qt.quantize(calibration_result)
        elif quant_type == "dynamic":
            qt.load_quantization_recipe(recipe.dynamic_wi8_afp32())
            qt = qt.quantize()
        else:
            raise ValueError(f"Unsupported quantization type: {quant_type}")

        qt.export_model(f"{Path(save_path).stem}_quant.tflite")


def _get_calibration_data(
    tflite_model_path: str,
    root_dir: str = "",
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
) -> dict[str, list[dict[str, Any]]]:
    """Generate random dummy calibration data.

    The calibration data is a list of dictionaries, each of which contains an
    input data for a single calibration step. The key is the input tensor name
    and the value is the input tensor value.

    Args:
      num_samples: Number of samples to generate.

    Returns:
      A list of calibration data.
    """
    def _get_signature_detail(tflite_model_path: str) -> str:
        interpreter = tfl_interpreter_utils.create_tfl_interpreter(tflite_model_path)
        signature = interpreter.get_signature_list()
        # {'serving_default': {'inputs': ['args_0'], 'outputs': ['output_0']}}
        return signature['serving_default']['inputs'][0]

    N, C, H, W = input_shape

    calibration_samples = []
    for img in glob.glob(f"{root_dir}/*.jpg"):
        pil_image = Image.open(img).convert("L") if C == 1 else Image.open(img).convert("RGB")
        pil_image = pil_image.resize((W, H))
        data = np.array(pil_image) / 255.0
        calibration_samples.append(
            {
                str(_get_signature_detail(tflite_model_path)):
                data.reshape(input_shape).astype(np.float32)
            }
        )
    calibration_data = {
        tfl_interpreter_utils.DEFAULT_SIGNATURE_KEY: calibration_samples,
    }
    return calibration_data


if __name__ == "__main__":
    model = "YOUR_PYTORCH_MODEL_HERE"   # Replace with your PyTorch model
    input_shape = (1, 3, 224, 224)      # Replace with your model's input shape

    convert_to_tflite(
        model,
        save_path='model.tflite',
        quant=True,
        quant_type="static",
        calibration_path="data",
        input_shape=input_shape,
    )
