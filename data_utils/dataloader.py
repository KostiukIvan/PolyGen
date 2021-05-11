from data_utils.transformations import *
import numpy as np
import os
from os.path import join
from torch.utils.data import Dataset
from torchvision.transforms import Lambda
from sklearn.model_selection import train_test_split


def load_obj(filename):
    """Load vertices from .obj wavefront format file."""
    vertices = []
    with open(filename, 'r') as mesh:
        for line in mesh:
            data = line.split()
            if len(data) > 0 and data[0] == 'v':
                vertices.append(data[1:])
    return np.array(vertices, dtype=np.float32)


class VerticesDataset(Dataset):
    def __init__(self,
                root_dir,
                classes,
                transform=[SortVertices(), NormalizeVertices(), QuantizeVertices(), ToTensor(), ResizeVertices(800), VertexTokenizer(801)],
                split='train',
                train_percentage=0.925):
        """
        Args:
            root_dir (string): Directory with all the ShapeNet data.
            classes (list): list of all data classes that can be used. Use None if all classes should be used.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        
        file_paths = []
        for class_idx, class_name in enumerate(os.listdir(root_dir)):
            if classes is not None and class_name not in classes:
                continue
            for augmentation_dir in os.listdir(join(root_dir, class_name)):
                for augmented_mesh_file in os.listdir(join(root_dir, class_name, augmentation_dir)):
                    file_paths.append((join(root_dir, class_name, augmentation_dir, augmented_mesh_file), class_idx))
        self.X_train, self.X_valid = train_test_split(file_paths, test_size=1 - train_percentage, shuffle=True)
                
    def __len__(self):
        if self.split not in {'train', 'valid'}:
            raise ValueError('Invalid split. Should be train or valid.')
        return len(self.X_train) if self.split == 'train' else len(self.X_valid)

    def __getitem__(self, idx):
        if self.split not in {'train', 'valid'}:
            raise ValueError('Invalid split. Should be train or valid.')

        dataset = self.X_train if self.split == 'train' else self.X_valid
        file_paths, class_idx = dataset[idx]
        vertices = load_obj(file_paths)
        
        if self.transform:
            for trans in self.transform:
                vertices = trans(vertices)

        return vertices, class_idx


class MeshesDataset(Dataset):
    def __init__(self,
                 root_dir, *,
                 transform=[SortVertices(),
                            NormalizeVertices(),
                            QuantizeVertices(),
                            ToTensor(),
                            ResizeVertices(800),
                            VertexTokenizer(801),
                            Lambda(lambda x: torch.flatten(x))]):
        """
        Args:
            root_dir (string): Directory with all the ShapeNet data.
        """
        self.root_dir = root_dir
        self.transform = transform

        self.dataset = []
        for class_idx, obj_name in enumerate(os.listdir(root_dir)):
            obj_path = join(root_dir, obj_name)
            vertices = load_obj(obj_path)
            self.dataset.append((vertices, class_idx))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        vertices, class_idx = self.dataset[idx]

        if self.transform:
            for trans in self.transform:
                vertices = trans(vertices)

        return vertices, class_idx


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    ds = MeshesDataset("../meshes")
    dataloader = DataLoader(ds, batch_size=4, shuffle=True)
    for batch in dataloader:
        print(batch)

    ds_2 = VerticesDataset("../data/shapenet_samples", classes=None)
    dataloader = DataLoader(ds_2, batch_size=4, shuffle=True)
    for batch in dataloader:
        print(batch)
        break
