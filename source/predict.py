import argparse
import os
import sys
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from io import StringIO
from six import BytesIO

# import model
from model import QuantileModel

# accepts and returns numpy data
CONTENT_TYPE = 'application/x-npy'


def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QuantileModel(model_info['in_tabular_features'], len(model_info['quantiles']))

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # prep for testing
    model.to(device).eval()

    print("Done loading model.")
    return model


def input_fn(serialized_input_data, content_type):
    print('Deserializing the input data.')
    if content_type == CONTENT_TYPE:
        stream = BytesIO(serialized_input_data)
        return np.load(stream)
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)


def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    if accept == CONTENT_TYPE:
        buffer = BytesIO()
        np.save(buffer, prediction_output)
        return buffer.getvalue(), accept
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)


def predict_fn(input_data, model):
    print('Predicting class labels for the input data...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Process input_data so that it is ready to be sent to our model
    # convert data to numpy array then to Tensor
    data = torch.from_numpy(input_data)
    data = data.to(device)

    # Put model into evaluation mode
    model.eval()

    # Compute the result of applying the model to the input data.
    out = model(data)
    # The variable `result` should be a numpy array; a single value 0-1
    result = out.cpu().detach().numpy()

    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str)
    parser.add_argument('--data-dir', type=str)
    args = parser.parse_args()
    model = model_fn(args.model_dir)
    input_data = pd.read_csv(
        filepath_or_buffer=os.path.join(args.data_dir, "pp_test.csv"),
        header=None, names=None)
    results_df = pd.read_csv(
        filepath_or_buffer=os.path.join(args.data_dir, "results.csv"),
        header=0, names=None)
    predictions_df = pd.DataFrame(predict_fn(input_data, model))
    results_df['FVC'] = predictions_df[1]
    results_df['Confidence'] = predictions_df[2] - predictions_df[0]
    results_df.Weeks = results_df.Weeks.astype('str')
    results_df['Patient_Week'] = results_df['Patient'] + '_' + results_df['Weeks']
    results_df = results_df.drop(columns=['Patient', 'Weeks'])
    results_df_cols = list(results_df.columns)
    results_df_cols[0], results_df_cols[1], results_df_cols[2] = results_df_cols[2], results_df_cols[0], results_df_cols[1]
    results_df = results_df[results_df_cols]
    results_df.to_csv('data/submission.csv', index=False)
