import random
from abc import abstractmethod
from Dataset.dataset import Dataset
from Transforms.base_transform import Transform



class DataLoader:
    """Composes a batched generator around the dataset"""
    def __init__(self, dataset: Dataset, batch_size:int, seed:int, shuffle=False):
        self.dataset = dataset
        self.bach_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
    
    @abstractmethod
    def __iter__(self, dataset: Dataset):
        raise NotImplementedError('Subclasses must implement this method')
    

class UrflowDataLoader(DataLoader):

    def __init__(self, dataset: Dataset, batch_size:int , seed:int, transform: Transform, shuffle: bool=False, out_dim=20):
        super().__init__(dataset, batch_size, seed, shuffle)
        self.out_dim = out_dim
        self.transfrom = transform



    def __iter__(self):
        """Implements an iter around the Dataset obj"""
        size = len(self.dataset)
        indices = list(range(size))

        if self.shuffle:
            random.shuffle(indices)

        for i in range(0, size, self.bach_size):
            sampled_indices = indices[i : i+ self.bach_size]
            temp = []

            for index in sampled_indices:
                x, y  = self.dataset[index]
                if self.enable_2d:
                    x_transformed = self.transfrom.fit_transform(x)
                temp.append(x_transformed, y)

            yield self._make_contiguous(temp, self.out_dim)

    def _make_contiguous(temp: list, dim: int):
        """Batches the data into contigous memory for faster execution"""
        """Deduce the kind of transform from ndim of the x than providing extra parameters for this """
        pass
