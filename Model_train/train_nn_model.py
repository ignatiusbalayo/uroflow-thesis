import os
import torch
import joblib
import numpy as np
import torch.nn as nn
from Dataset.devices import Device
from models.cnn_model import UroflowCNN
from models.mlp_model import UroflowMLP
from Dataset.data_loader import UrflowDataLoader
from Dataset.dataset import UroflowDataset_v2
from Transforms.base_transform import LFT, MelSpectrogram_gpu
from configs.path_configs import SEED, DATA_PATH, BATCH_SIZE, NO_BINS
from utils.read_normalization_params import read_norm_data, Transform_keys


class _SubsetDataset:
    """Builds data split wrapper"""
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        return self.dataset[self.indices[index]]
    
def _split_indices(n, val_ratio=0.2, seed= SEED):
    """Generates train and val splits"""

    indices = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    split = int(n * (1 - val_ratio))
    return indices[:split].tolist(), indices[split:].tolist()


def train_one(model, dataloader, optimizer, criterion, x_mean, x_std, y_mean, y_std, device):
    """Run one training epoch, returns mean batch loss"""

    model.train()
    total_loss, n_batches = 0.0, 0

    for x, y in dataloader:
        x = (x - x_mean) / x_std
        y = (y - y_mean) / y_std

        x = torch.tensor(x, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.float32).to(device)

        optimizer.zero_grad()
        loss = criterion(model(x).squeeze(1), y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def validate_one(model, dataloader, criterion, x_mean, x_std, y_mean, y_std, device):
    """Runs one validation pass; returns mean batch loss"""
    model.eval()
    total_loss, n_batches = 0.0, 0

    with torch.no_grad():
        for x, y in dataloader:
            x = (x - x_mean) / x_std
            y = (y - y_mean) / y_std

            x = torch.tensor(x, dtype= torch.float32).to(device)
            y = torch.tensor(y, dtype=torch.float32).to(device)

            loss = criterion(model(x).squeeze(1), y)
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)


def train_nn(epochs=50, lr=1e-3, val_ratio=0.2):
    torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss()

    model_configs = {
        'mlp_mel_1d': {
            'transform_key': Transform_keys.Mel_1d,
            'enable_2d': False,
            'model_cls': UroflowMLP,
            'model_kwargs': {}
        },
        'mlp_lft': {
            'transform_key': Transform_keys.LFT,
            'model_cls': UroflowMLP,
            'model_kwargs': {}
        },
        'cnn': {
            'transform_key': Transform_keys.Mel_2d,
            'enable_2d': True,
            'model_cls': UroflowCNN,
            'model_kwargs': {}
        }
    }

    global_state = {}

    for device in Device.get_devices():
        dataset = UroflowDataset_v2(DATA_PATH, device)
        train_idx, val_idx = _split_indices(len(dataset), val_ratio)

        train_subset = _SubsetDataset(dataset, train_idx)
        val_subset = _SubsetDataset(dataset, val_idx)

        device_state = {}
        for model_name, cfg in model_configs.items():
            if model_name == 'mlp_lft':
                transofrm = LFT(no_bins=NO_BINS)
            else:
                transofrm = MelSpectrogram_gpu(sr=dataset.get_device_rate(), enable_2d=cfg['enable_2d'])
            
            x_mean, x_std, y_mean, y_std = read_norm_data(device, cfg['transform_key'])
            train_loader = UrflowDataLoader(train_subset, batch_size=BATCH_SIZE, transform=transofrm, shuffle=True, permutate=True)
            val_loader = UrflowDataLoader(val_subset, batch_size=BATCH_SIZE, transform=transofrm, shuffle=False, permutate=False)

            model = cfg['model_cls'](**cfg['model_kwargs']).to(torch_device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            best_val_loss = float('inf')
            best_weight = None

            for epoch in range(epochs):
                train_loss = train_one(model, train_loader, optimizer, criterion, x_mean, x_std, y_mean, y_std, torch_device)
                val_loss = validate_one(model, val_loader, criterion, x_mean, x_std, y_mean, y_std, torch_device)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weight = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                
                print(f'[{device.value}][{model_name}] epoch {epoch + 1} / {epochs} train- {train_loss:.4f} val= {val_loss:.4f}')
            
            model.load_state_dict(best_weight)
            device_state[model_name] = {'model': model.cpu(), 'val_loss': best_val_loss}
        
        global_state[device.value] = device_state
    return global_state


if __name__ == '__main__':
    os.system
    state = train_nn(epochs=50)
    torch.save(state, 'nn_model_state.pt')




    




