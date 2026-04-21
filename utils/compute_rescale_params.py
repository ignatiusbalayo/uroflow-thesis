from Dataset.dataset import UroflowDataset_v2
from Dataset.data_loader import UrflowDataLoader
from configs.path_configs import DATA_PATH
from Transforms.base_transform import LFT, MelSpectrogram


def compute_rescale_configs():
    dataset = UroflowDataset_v2(DATA_PATH)