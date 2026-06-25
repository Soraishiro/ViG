import torch

def load_trusted_checkpoint(path, map_location='cpu'):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

def extract_checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            return checkpoint['state_dict']
        if 'model' in checkpoint:
            return checkpoint['model']
    return checkpoint
