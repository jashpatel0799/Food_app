import torch
import wandb
from tqdm.auto import tqdm

API_KEY = "881252af31786a1cf813449b9b4124955f54703e"


def train_loop(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer, accuracy_fn, device: torch.device):

    """
    Function for Model Training

    Args:
        model: A pytorch model you want to train
        dataloader: A dataloader for intance for model training
        loss_fn: A loss function for calculate model loss
        accuracy_fn: A Accuracy function to check how model is accurate
        device: A device on which model run i.e.: "cuda" or "cpu"

    Return:
        list of train loss and accuracy and also model weights

    Example usage:
        train_loop(model = mymodel, dataloader = train_dataloader, loss_fn = loss_fn, 
                    accuracy_fn = accuracy_fn, device = device)
    """

    train_loss, train_acc = 0, 0 

    model.train()

    for batch, (x_train, y_train) in enumerate(dataloader):
        x_train, y_train = x_train.to(device), y_train.to(device)

        # 1. Forwrad Pass
        logits = model(x_train)

        # 2. Loss
        loss = loss_fn(logits, y_train)

        # 3. Gradzero step
        optimizer.zero_grad()

        # 4. Backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        acc = accuracy_fn(torch.argmax(logits, dim = 1), y_train)

        train_acc += acc
        train_loss += loss

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return model, train_loss, train_acc


def test_loop(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module,
              accuracy_fn, device: torch.device):

    """
    A Funtion to test the model after traininig

    Args:
        model: A model which you want to test on test intance
        dataloader: A dataloader intance on which you test model
        loss_fn: A loss function to calculate the model loss
        accuracy_fn : A accuracy function to check model accuracy on dataloader intance
        device: A device on whic you want to run model i.e.: "cuda" or "cpu"

    Return:
        A list of test loss and Accuracy

    Example Usage:
        test_loop(model = mymodel, dataloader = test_datloader, loss_fn = loss_fn,
                  accuracy_fn = accuracy+fn, device = device)
    """

    test_loss, test_acc = 0, 0

    model.eval()
    with torch.inference_mode():
        for x_test, y_test in dataloader:
            x_test, y_test = x_test.to(device), y_test.to(device)

            # 1. Forward Pass
            logits = model(x_test)

            # 2. Loss
            test_loss += loss_fn(logits, y_test)

            test_acc += accuracy_fn(torch.argmax(logits, dim = 1), y_test)

        test_acc /= len(dataloader)
        test_loss /= len(dataloader)

    return test_loss, test_acc


def log_image_table(images, predicted, labels, probs):
    "Log a wandb.Table with (img, pred, target, scores)"
    wandb.login(key=API_KEY)
    print("[LOG]: Login Succesfull.")
    # 🐝 Create a wandb Table to log images, labels and predictions to
    table = wandb.Table(columns=["image", "pred", "target"]+[f"score_{i}" for i in range(10)])
    for img, pred, targ, prob in zip(images.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")):
        table.add_data(wandb.Image(img[0].numpy()*255), pred, targ, *prob.numpy())
    wandb.log({"predictions_table":table}, commit=False)



def validation(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module,
               accuracy_fn, log_images: bool, device: torch.device, batch_idx=0):
    """
    A Function for hyperparameter tuning

    Args:
        model: A model the you want tune its hyperparameter
        dataloder: Adataloader intance for hyperparameter tuning
        loss_fn: A loss funtion to calcualte model loss
        Accuracy_fn: a accuracy function to calcultae accuracy for model perforamnce
        device: A device on which model run i.e.: "cuda" or "cpu"

    Return:
        A list of accuracy and loss

    Example usage:
        validation(model = mymodel, dataloader = valid_dataloader, loss_fn = loss_fn,
                   accuracy_fn = accuracy_fn, device = device)
    """

    val_loss, val_acc = 0, 0

    model.eval()
    with torch.inference_mode():
        for i, (x_val, y_val) in enumerate(dataloader):
            x_val, y_val = x_val.to(device), y_val.to(device)

            logits = model(x_val)

            val_loss += loss_fn(logits, y_val)

            val_acc += accuracy_fn(torch.argmax(logits, dim = 1), y_val)

            # Log one batch of images to the dashboard, always same batch_idx.
            if i==batch_idx and log_images:
                log_image_table(x_val, torch.max(logits.data, 1)[0], y_val, logits.softmax(dim=1))

        val_loss /= len(dataloader)
        val_acc /= len(dataloader)

    return val_loss, val_acc

def train(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader, test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module, accuracy_fn, epochs: int, device: torch.device):

    """
    A function to train and test the pytorch model

    Args:
        model: A model to train and test
        train_dataloader: A dataloader intance for train model
        test_dataloader: A dataloader intance for test model
        optimizer: A optimizer funtion to optimize the model
        loss_fn: A loss function to calculate model loss
        accuracy_fn: An accuracy  to calculate model performance
        epochs: number of iteration to run the loop
        device: A device on which model run i.e.: "cuda" or "cpu"

    Return:
        train model, List of train and test losses and accuracy

    Example usage:
        train(model = mymodel, train_dataloader = train_dataloader, test_dataloader = test_dataloader, optimizer = optimizer,
              loss_fn = loss_fn, acuuracy_fn = accuracy_fn, epochs = epochs, device = device)
    """

    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    for epoch in tqdm(range(epochs)):

        print(f"\nEpoch: {epoch+1}")

        train_model, train_loss, train_acc = train_loop(model = model, dataloader = train_dataloader, 
                                                        loss_fn = loss_fn, optimizer = optimizer, 
                                                        accuracy_fn = accuracy_fn, device = device)

        test_loss, test_acc = test_loop(model = model, dataloader = test_dataloader, loss_fn = loss_fn,
                                        accuracy_fn = accuracy_fn, device = device)

        print(f"Train Loss: {train_loss:.5f} | Test Loss: {test_loss:.5f} || Train Accuracy: {train_acc:.5f} | Test Accuracy: {test_acc:.5f}")

        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())
        train_accs.append(train_acc.item())
        test_accs.append(test_acc.item())

    return train_losses, test_losses, train_accs, test_accs, train_model 