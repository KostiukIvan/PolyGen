from data_utils.transformations import *
import numpy as np
import pandas as pd
from os import listdir
from os.path import join
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

categories = ['02691156']


class VerticesDataset(Dataset):
    def __init__(self,
                root_dir=r'C:\Users\ivank\UJ\Deep learining with multiple tasks\projects\ShapeNetCore_PolyGenSubset\train',
                classes=['04330267'],
                transform=[SortVertices(), NormalizeVertices(), QuantizeVertices(), ToTensor(), ResizeVertices(800), VertexTokenizer(801)],
                split='train',
                train_percentage=0.925):
        """
        Args:
            root_dir (string): Directory with all the ShapeNet data.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        
        file_paths = []
        for class_name in os.listdir(root_dir):
            if not class_name in classes:
                continue
            for augmentation_dir in os.listdir(join(root_dir, class_name)):
                for augmented_mesh_file in os.listdir(join(root_dir, class_name, augmentation_dir)):
                    file_paths.append(join(root_dir, class_name, augmentation_dir, augmented_mesh_file))
        self.X_train, self.X_valid = train_test_split(file_paths, test_size = 1 - train_percentage, shuffle=True)
                

    def __len__(self):
        if self.split not in {'train', 'valid'}:
            raise ValueError('Invalid split. Should be train or valid.')
        return len(self.X_train) if self.split == 'train' else len(self.X_valid)

    def __getitem__(self, idx):
        if self.split not in {'train', 'valid'}:
            raise ValueError('Invalid split. Should be train or valid.')

        dataset = self.X_train if self.split == 'train' else self.X_valid
        vertices = self._load_obj(dataset[idx])
        
        if self.transform:
            for trans in self.transform:
                vertices = trans(vertices)

        return vertices

    def _get_names(self) -> pd.DataFrame:
        filenames = []
        for category in categories:
            for f in listdir(join(self.root_dir, category)):
                if f not in ['.DS_Store']:
                    filenames.append((category, f))
        return pd.DataFrame(filenames, columns=['category', 'filename'])

    def _load_obj(self, filename):
        """Load vertices from .obj wavefront format file."""
        vertices = []
        print(filename)
        with open(filename, 'r') as mesh:
            for line in mesh:
                data = line.split()
                if len(data) > 0 and data[0] == 'v':
                    vertices.append(data[1:])
        return np.array(vertices, dtype=np.float32)