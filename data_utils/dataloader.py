from data_utils.transformations import *
import numpy as np
import pandas as pd
from os import listdir
from os.path import join
from torch.utils.data import Dataset

categories = ['basket', 'chair', 'lamp', 'sofa', 'table']


# TODO: Implement TokenizeVertices
class VerticesDataset(Dataset):
    def __init__(self, root_dir=r'C:\Users\ivank\UJ\Deep learining with multiple tasks\projects\ShapeNetCore_PolyGenSubset\train',
                 classes=['basket'], transform=[SortVertices(), NormalizeVertices(), QuantizeVertices(), ToTensor(), ResizeVertices()],
                 split='train', train_percentage=0.925):
        """
        Args:
            root_dir (string): Directory with all the ShapeNet data.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split

        pc_df = self._get_names()
        pc_df = pc_df[pc_df.category.isin(classes)].reset_index(drop=True)

        self.vertices_names_train = pd.concat([pc_df[pc_df['category'] == c][
                                                :int(train_percentage * len(pc_df[pc_df['category'] == c]))].
                                                reset_index(drop=True) for c in classes])
        self.vertices_names_valid = pd.concat([pc_df[pc_df['category'] == c][
                                                int(train_percentage * len(pc_df[pc_df['category'] == c])):].
                                                reset_index(drop=True) for c in classes])

    def __len__(self):
        if self.split == 'train':
            pc_names = self.vertices_names_train
        elif self.split == 'valid':
            pc_names = self.vertices_names_valid
        else:
            raise ValueError('Invalid split. Should be train or valid.')
        return len(pc_names)

    def __getitem__(self, idx):
        if self.split == 'train':
            pc_names = self.vertices_names_train
        elif self.split == 'valid':
            pc_names = self.vertices_names_valid
        else:
            raise ValueError('Invalid split. Should be train or valid.')

        pc_category, pc_filename = pc_names.iloc[idx].values

        pc_filepath = join(self.root_dir, pc_category, pc_filename)
        vertices = self._load_obj(pc_filepath)

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
        with open(filename, 'r') as mesh:
            for line in mesh:
                data = line.split()
                if len(data) > 0 and data[0] == 'v':
                    vertices.append(data[1:])
        return np.array(vertices, dtype=np.float32)


