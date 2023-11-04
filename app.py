import gradio as gr
import torch
import torch.nn as nn
import onnx
import data, utils
from typing import Tuple, Dict
from train import NUM_CLASSES
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torchvision.transforms as T
import onnxruntime as ort
import numpy as np
from timeit import default_timer as timer
from pathlib import Path

PATH = "save_model/food_cpu.onnx"


model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
model.classifier = nn.Sequential(
                                    nn.Dropout(p = 0.2, inplace = True),
                                    nn.Linear(1280, NUM_CLASSES),
                                    # nn.Softmax()
                                )
model = utils.load_model(model, "save_model/best_model.pth")

utils.onnx_inference(model, PATH, "cpu")
onnx_model = onnx.load(PATH)
onnx_check = onnx.checker.check_model(onnx_model)

classes = data.train_datasets.classes

trn = T.ToPILImage()

def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and return prediction and time take."""
    # Start the timer
    start_time = timer()

    # transform the target image and add a batch dimension
    img = data.transform(img).unsqueeze(dim = 0)

    # inference using onnx
    ort_sess = ort.InferenceSession(PATH)
    outputs = ort_sess.run(None, {'input': img.numpy()})

    predicted = classes[outputs[0][0].argmax(0)]

    # print("\n", outputs[0][0], "\n")
    outputs = np.array(torch.softmax(torch.from_numpy(outputs[0]), dim = 1))
    pred_labels_and_prob = {classes[i]: float(outputs[0][i]) for i in range(len(classes))}
    # print(f'Predicted: "{predicted}"')

    # Calculate the predicion time
    pred_time = round(timer() - start_time, 5)

    # Return the prediction dictionary and prediction time
    return pred_labels_and_prob, pred_time


image = trn(data.test_datasets[3][0])

exp_dir = "./example_data/"
test_data_paths = list(Path(exp_dir).glob("*.jpg"))
# print(test_data_paths)
example_list = [[str(filepath)] for filepath in test_data_paths]
# print(example_list)


# pred_dict, pred_time = predict(img = image)
# print(f"Predicted label and probability: {pred_dict}")
# print(f"Prediction time: {pred_time}")

title = "FoodVision 🍕🥩🍣"
description = "An EfficientNetB2 feature extractor computer vision model to classify images of food in 101 different classes."
article = "Created at [09. PyTorch Model Deployment](https://www.learnpytorch.io/09_pytorch_model_deployment/)."


# Create the Gradio demo
demo = gr.Interface(fn=predict, # mapping function from input to output
                    inputs=gr.Image(type="pil"), # what are the inputs?
                    outputs=[gr.Label(num_top_classes=101, label="Predictions"), # what are the outputs?
                             gr.Number(label="Prediction time (s)")], # our fn has two outputs, therefore we have two outputs
                    examples=example_list, 
                    title=title,
                    description=description,
                    article=article)

# Launch the demo!
# demo.launch(debug=False, # print errors locally?
#             share=True) # generate a publically shareable URL?
demo.launch(share=True)