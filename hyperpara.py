import torch
import torch.nn as nn
import optuna
from optuna.trial import TrialState
from torch import optim
import engine, data, utils
from train import device, LEARNIGN_RATE, NUM_EPOCH, NUM_CLASSES
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchmetrics.classification import MulticlassAccuracy

import wandb

# WanDB login

API_KEY = "881252af31786a1cf813449b9b4124955f54703e"

wandb.login(key=API_KEY)
print("[LOG]: Login Succesfull.")




def objective(trial, n_trials=100):
    """Objective function to be optimized by Optuna.
    Hyperparameters chosen to be optimized: optimizer, learning rate,
    dropout values, number of convolutional layers, number of filters of
    convolutional layers, number of neurons of fully connected layers.
    Inputs:
        - trial (optuna.trial._trial.Trial): Optuna trial
    Returns:
        - accuracy(torch.Tensor): The test accuracy. Parameter to be maximized.
    """

    lr = trial.suggest_float("lr", LEARNIGN_RATE*1e-2, LEARNIGN_RATE, log=True)                                 # Learning rates
    n_epochs = trial.suggest_int('n_estimators', NUM_EPOCH//2, NUM_EPOCH)

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr) #torch.optim.Adam(model.parameters(), lr = lr)# getattr(optim, optimizer_name)(model.parameters(), lr=lr)


    loss_fn = torch.nn.CrossEntropyLoss()
    accuracy_fn = MulticlassAccuracy(num_classes = NUM_CLASSES).to(device)

    wandb.init(
                    # set the wandb project where this run will be logged
                    project="food-app",
                    
                    # track hyperparameters and run metadata
                    config={
                    "optimizer": trial.params["optimizer"],
                    "architecture": "Efficientnet B0",
                    "dataset": "Food101",
                    "epochs": trial.params["n_estimators"],
                    }
                )

    # Training of the model
    best_loss = 100
    patience = 5
    early_stop = 0
    for epoch in range(n_epochs):

        train_model, train_loss, train_acc = engine.train_loop(model = model, dataloader = data.train_dataloader, 
                                                        loss_fn = loss_fn, optimizer = optimizer, 
                                                        accuracy_fn = accuracy_fn, device = device)

        val_loss, val_acc = engine.validation(model = model, dataloader = data.valid_dataloader, loss_fn = loss_fn,
                                        accuracy_fn = accuracy_fn, log_images=(epoch==(wandb.config.epochs-1)), device = device)
        
        
        # For pruning (stops trial early if not promising)
        trial.report(val_acc, epoch)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_acc < best_loss:
            early_stop = 0
            best_loss = val_acc
            utils.save_model(model = train_model, target_dir = "./save_model", model_name = f"best_model.pth")

        else:
            early_stop += 1

        if early_stop == patience:
            break

    wandb.log({"Train Loss": train_loss, "Train Accuracy": train_acc, "Validation Loss": val_loss, "Validation Accuracy": val_acc})

    wandb.finish()

    return val_acc


model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
model.classifier = nn.Sequential(
                                    nn.Dropout(p = 0.2, inplace = True),
                                    nn.Linear(1280, NUM_CLASSES),
                                    # nn.Softmax()
                                )

# isExist = os.path.exists("save_model/train_model_4e-06.pth")
# print(isExist)
model = utils.load_model(model, "save_model/train_model_4e-06.pth").to(device)
# print(model)

# Create an Optuna study to maximize test accuracy
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=11)

# Find number of pruned and completed trials
pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

# Display the study statistics
print("\nStudy statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

trial = study.best_trial
print("Best trial:")
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))