import os
import torch
from datetime import datetime


def load_model(params, model, optimizer):
    """
    Load the saved model under the the specified path passed in params.
    Returns dirname of a saved weights.
    """
    if 'load_weights' in params:
        checkpoint = torch.load(params['load_weights'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch'] + 1
        total_loss = checkpoint['loss']
        model_weights_path = os.path.dirname(params['load_weights'])
        print('Loaded', model_weights_path)
    else:
        model_weights_path = os.path.join(os.getcwd(), 'weights', datetime.now().strftime("%d_%m_%Y_%H_%M"))
        os.makedirs(model_weights_path, exist_ok=True)
        epoch = 1
        total_loss = 0.

    return model_weights_path, epoch, total_loss
