import torch
import torch.nn as nn

import onnx
import data, utils
from train import device, NUM_CLASSES
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import onnxruntime as ort
import numpy as np



model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
model.classifier = nn.Sequential(
                                    nn.Dropout(p = 0.2, inplace = True),
                                    nn.Linear(1280, NUM_CLASSES),
                                    # nn.Softmax()
                                )
model = utils.load_model(model, "save_model/best_model.pth").to(device)

PATH = "save_model/food_cpu.onnx"
# onnx inference
utils.onnx_inference(model, PATH, "cpu")
onnx_model = onnx.load(PATH)
onnx_check = onnx.checker.check_model(onnx_model)

# print(onnx_check)
x, y = data.test_datasets[0][0], data.test_datasets[0][1]
ort_sess = ort.InferenceSession(PATH)
outputs = ort_sess.run(None, {'input': x.unsqueeze(dim = 0).numpy()})

# Result
classes = data.train_datasets.classes
predicted, actual = classes[outputs[0][0].argmax(0)], classes[y]
print(f'Predicted: "{predicted}", Actual: "{actual}"')