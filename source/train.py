from __future__ import print_function # future proof
import argparse
import os
import time

import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import train_test_split

# pytorch
import torch
from torch.optim import Adam
import torch.utils.data
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

# import model
from source.model import QuantileModel


def model_fn(model_dir):
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QuantileModel(model_info['in_tabular_features'],
                          model_info['out_quantiles'])

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    return model.to(device)


# Load the training data from a csv file
def _get_train_loader(batch_size, data_dir):
    print("Get data loader.")

    # read in csv file - with FVC in first column, then rest of features
    train_data = pd.read_csv(filepath_or_buffer=os.path.join(data_dir, "pp_train.csv"), header=None, names=None)
    print(train_data.head())

    # labels are first column
    train_y = torch.from_numpy(train_data[[0]].values).float().squeeze()
    # features are the rest apart from fvc
    train_x = torch.from_numpy(train_data.drop([0], axis=1).values).float()

    X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.33, random_state=42)

    # create datasets
    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    val_ds = torch.utils.data.TensorDataset(X_val, y_val)

    dataloaders = {
        'train': DataLoader(train_ds, batch_size=batch_size,
                            shuffle=True, num_workers=2),
        'val': DataLoader(val_ds, batch_size=batch_size,
                          shuffle=False, num_workers=2)
    }
    return dataloaders


# Provided train function
def train(model, train_loader, epochs, optimizer, lr_scheduler, device, quantiles, model_dir):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    optimizer    - The optimizer to use during training.
    lr_scheduler - The learning rate optimiser
    device       - Where the model and data should be loaded (gpu or cpu).
    quantiles    - For quantile regression
    model_dir    - model directory
    """

    for epoch in range(epochs):
        start_time = time.time()
        itr = 1
        model.train()
        train_losses =[]
        for batch in train_loader['train']:
            batch_x, batch_y = batch
            inputs = batch_x.float().to(device)
            targets = batch_y.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                preds = model(inputs)
                # use the quantile loss for backpropagation in the train dataset
                loss = quantile_loss(preds, targets, quantiles)
                train_losses.append(loss.tolist())
                loss.backward()
                optimizer.step()
            if itr % 10 == 0:
                print(f"[TRAIN] Epoch #{epoch+1} Iteration #{itr} quantile loss: {loss}")
            itr += 1
        model.eval()
        all_preds = []
        all_targets = []
        itr = 1
        for batch in train_loader['val']:
            batch_x, batch_y = batch
            inputs = batch_x.float().to(device)
            targets = batch_y.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(False):
                preds = model(inputs)
                all_preds.extend(preds.detach().cpu().numpy().tolist())
                all_targets.extend(targets.numpy().tolist())
        all_preds =torch.FloatTensor(all_preds)
        all_targets =torch.FloatTensor(all_targets)
        # metric loss used for loss calculation of validation ds
        val_metric_loss = metric_loss(device, all_preds, all_targets)
        val_metric_loss = torch.mean(val_metric_loss).tolist()
        # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        lr_scheduler.step()
        elapsed_time = time.time() - start_time
        print(f"Epoch #{epoch+1}","Training loss : {0:.4f}".format(np.mean(train_losses)),
              "Validation LLL : {0:.4f}".format(val_metric_loss),
              f"Time taken : {elapsed_time * 1000} milliseconds")
        torch.save(copy.deepcopy(model.state_dict()), model_dir+'model.pth')


def quantile_loss(preds, target, quantiles):
    #assert not target.requires_grad
    assert len(preds) == len(target)
    losses = []
    for i, q in enumerate(quantiles):
        errors = target - preds[:, i]
        losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
    loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
    return loss


def metric_loss(device, pred_fvc,true_fvc):
    #Implementation of the metric in pytorch
    sigma = pred_fvc[:, 2] - pred_fvc[:, 0]
    true_fvc=torch.reshape(true_fvc,pred_fvc[:,1].shape)
    sigma_clipped=torch.clamp(sigma,min=70)
    delta=torch.clamp(torch.abs(pred_fvc[:,1]-true_fvc),max=1000)
    metric=torch.div(-torch.sqrt(torch.tensor([2.0]).to(device))*delta,sigma_clipped)-torch.log(torch.sqrt(torch.tensor([2.0]).to(device))*sigma_clipped)
    return metric


def save_model_params(model_dir):
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'in_tabular_features': args.in_tabular_features,
            'quantiles': args.quantiles
        }
        torch.save(model_info, f)


def main(arguments):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set the seed for generating random numbers
    torch.manual_seed(arguments.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(arguments.seed)

    # get train loader
    train_loader = _get_train_loader(arguments.batch_size, arguments.data_dir) # data_dir from above..


    ## Build model
    model = QuantileModel(arguments.in_tabular_features, len(args.quantiles)).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(params, lr=arguments.lr)
    # to optimise the lr
    lr_scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    save_model_params(arguments.model_dir)

    # Trains the model (given line of code, which calls the above training function)
    train(model, train_loader, arguments.epochs, optimizer, lr_scheduler, device, arguments.quantiles, arguments.model_dir)


if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job

    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
   #parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    #parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str)
    parser.add_argument('--data-dir', type=str)

    # Training Parameters, given
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=150, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=3e-3, metavar='LR',
                        help='learning rate (default: 0.003)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # Model parameters
    parser.add_argument('--in_tabular_features', type=int, default=9, metavar='IF',
                        help='number of input features')
    parser.add_argument('--quantiles', type=str, default='0.2,0.5,0.8', help='Quantiles', required=True)

    args = parser.parse_args()
    args.quantiles = [float(item) for item in args.quantiles.split(',')]
    main(args)
