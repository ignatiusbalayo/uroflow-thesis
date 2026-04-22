from abc import ABC, abstractmethod
import numpy as np
import librosa
import torch
import torchaudio


class Transform(ABC):

    @abstractmethod 
    def forward(self, x):
        raise NotImplementedError('Child classes must implement this method')
    
    def __call__(self, x):
        return self.forward(x)
    
    def fit_transform(self, x):
        return self.forward(x)
    

class LFT(Transform):
    """Applies a linear fourier transformation to the data"""

    def __init__(self, no_bins, axis=1):
        self.no_bins = no_bins
        self.axis = axis

    def forward(self, x):
        x_fft = np.fft.rfft(x, axis=self.axis)
        x_mag = np.abs(x_fft)

        num_freqs = x_mag.shape[self.axis]
        bin_edges = np.linspace(0, num_freqs, self.no_bins + 1, dtype='int')

        x_binned = np.zeros((x.shape[0], self.no_bins), dtype=np.float32)
        for i in range(self.no_bins):
            start, end = bin_edges[i], bin_edges[i+1]
            bineed_slice = x_mag[:, start:end]
            x_binned[:, i] = bineed_slice.sum(axis=1) if not bineed_slice.size == 0 else 0
        return x_binned
    

class MelSpectrogram(Transform):
    """Computes the melspectrogram transformation of a given input"""

    def __init__(self, sr, n_fft=1024, n_mels=20, power=2.0, enable_2d=False, d_scale_factor=32):
        self.n_fft = n_fft
        self.power = power
        self.n_mels = n_mels
        self.enable_2d = enable_2d
        self.d_scale_factor = d_scale_factor
        self.sr = sr
    
    def forward(self, x):
        features = []

        for window in x:
            if window.ndim > 1:
                window = window.mean(axis=-1)
            mel_spec = librosa.feature.melspectrogram(
                y=window,
                sr = self.sr,
                n_fft=self.n_fft,
                hop_length= self.n_fft // self.d_scale_factor,
                n_mels= self.n_mels,
                power=self.power
            )
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)
            mel_features = log_mel if self.enable_2d else log_mel.sum(axis=1)
            features.append(mel_features)
        return np.stack(features)
    

class MelSpectrogram_gpu(MelSpectrogram):
    def forward(self, x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=self.n_fft,
            hop_length=self.n_fft // self.d_scale_factor,
            n_mels=self.n_mels,
            power=self.power,
        ).to(device)

        amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80.0).to(device)

        features = []
        for window in x:
            if window.ndim > 1:
                window = window.mean(axis=-1)

            tensor = torch.from_numpy(window.astype(np.float32)).to(device)
            mel_spec = mel_transform(tensor)
            log_mel = amplitude_to_db(mel_spec)

            mel_features = log_mel if self.enable_2d else log_mel.sum(dim=1)
            features.append(mel_features.cpu().numpy())

        return np.stack(features)





if __name__ == '__main__':
    from Dataset.dataset import UroflowDataset_v2
    from configs.path_configs import DATA_PATH
    from Dataset.devices import Device
    import os   

    os.system('clear')

    dataset = UroflowDataset_v2(DATA_PATH, Device.PHONE)

    x, _ = dataset[0]
    mel_spec = MelSpectrogram_gpu(dataset.device_rate, enable_2d=False)
    shape = None
    

    x_mel = mel_spec.fit_transform(x)
    x_mean, x_std = x_mel.mean(), x_mel.std()

    print(f'Dimensions of X with 1D mel spectrogram transform: {x_mel.shape}', f'Dims: {x_mel.ndim}', f'X_mean: {x_mean.shape}', f'X_std: {x_std.shape}', sep= ' | ')

    mel_2d = MelSpectrogram_gpu(dataset.device_rate, enable_2d=True)
    x_mel_2d = mel_2d.fit_transform(x)

    x_mean, x_std = x_mel_2d.mean((0, 2), keepdims=True), x_mel_2d.std((0, 2), keepdims=True)
    print(f'Dimension of X with 2d mel spectrogram transform: {x_mel_2d.shape}', f'Dims: {x_mel_2d.ndim}', f'X_mean: {x_mean.shape}', f'X_std: {x_std.shape}', sep=' | ')
        
