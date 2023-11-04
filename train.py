import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchmetrics.classification import MulticlassAccuracy

import data, engine, utils


# if __name__ == "__main__":
    # HYPERPARAMETERS
SEED = 64
NUM_EPOCH = 50
LEARNIGN_RATE = 4e-6 # 3e-4, 4e-5, 7e-6, 5e-7, 3e-9

NUM_CLASSES = 101
# CUDA_LAUNCH_BLOCKING=1

device = torch.device("cuda:3" if torch.cuda.is_available() else 'cpu')
# print(device)


if __name__ == "__main__":

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    model.classifier = nn.Sequential(
                                        nn.Dropout(p = 0.2, inplace = True),
                                        nn.Linear(1280, NUM_CLASSES),
                                        # nn.Softmax()
                                    )
    model = model.to(device)
    # print(model)


    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    accuracy_fn = MulticlassAccuracy(num_classes = NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNIGN_RATE)


    # print(f"lr: {scheduler.optimizer.param_groups[0]['lr']}")

    train_losses, test_losses, train_accs, test_accs, train_model = engine.train(model, data.train_dataloader, data.test_dataloader,
                                                                                    optimizer, loss_fn, accuracy_fn, NUM_EPOCH, device)

    utils.save_model(model = train_model, target_dir = "./save_model", model_name = f"train_model_{LEARNIGN_RATE}.pth")

    utils.plot_graph(train_losses = train_losses, test_losses = test_losses, train_accs = train_accs, 
                        test_accs = test_accs, fig_name = f"plots/cnn_train_Loss_and_accuracy_plot_{LEARNIGN_RATE}.jpg")