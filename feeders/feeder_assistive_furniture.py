import numpy as np
import random
import os
import json
from torch.utils.data import Dataset
from feeders import tools


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', data_type='j',
                 aug_method='z', intra_p=0.5, inter_p=0.0, window_size=-1,
                 debug=False, thres=64, uniform=False, partition=False):

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.data_type = data_type
        self.aug_method = aug_method
        self.intra_p = intra_p
        self.inter_p = inter_p
        self.window_size = window_size
        self.p_interval = p_interval
        self.thres = thres
        self.uniform = uniform
        self.partition = partition
        self.load_data()
        if partition:
            self.right_arm = np.array([6, 8, 10]) - 1
            self.left_arm = np.array([5, 7, 9]) - 1
            self.right_leg = np.array([12, 14, 16]) - 1
            self.left_leg = np.array([11, 13, 15]) - 1
            self.head = np.array([2, 1, 4, 3]) - 1
            self.new_idx = np.concatenate((self.right_arm, self.left_arm, self.right_leg, self.left_leg, self.head), axis=-1)
            # except for joint no.21

    def load_data(self):
        # data: N C V T M
        self.data = []
        self.label = []
        for file_name in os.listdir(self.data_path):
            if file_name.endswith('.json'):
                file_path = os.path.join(self.data_path, file_name)
                with open(file_path, 'r') as file:
                    json_file = json.load(file)
                    keypoints = [annotation['keypoints'] for annotation in json_file["annotations"]]
                    skeletons = np.array(keypoints)
                    self.data.append(skeletons[:, 1:, :])
                    self.label.append(json_file["category_id"])

        self.data = np.array(self.data)
        self.data = self.data.transpose(0, 3, 1, 2)
        self.data = np.expand_dims(self.data, axis=4)

        if self.split == 'train':
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)

        if self.split == 'train':
            # intra-instance augmentation
            p = np.random.rand(1)
            if p < self.intra_p:
                if '1' in self.aug_method:
                    data_numpy = tools.shear(data_numpy, p=0.5)
                if '2' in self.aug_method:
                    data_numpy = tools.rotate(data_numpy, p=0.5)
                if '3' in self.aug_method:
                    data_numpy = tools.scale(data_numpy, p=0.5)
                if '4' in self.aug_method:
                    data_numpy = tools.spatial_flip(data_numpy, p=0.5)
                if '6' in self.aug_method:
                    data_numpy = tools.gaussian_noise(data_numpy, p=0.5)
                if '7' in self.aug_method:
                    data_numpy = tools.gaussian_filter(data_numpy, p=0.5)
                if '8' in self.aug_method:
                    data_numpy = tools.drop_axis(data_numpy, p=0.5)
                if '9' in self.aug_method:
                    data_numpy = tools.drop_joint(data_numpy, p=0.5)
            else:
                data_numpy = data_numpy.copy()

        # modality
        data_numpy = data_numpy.copy()

        if self.partition:
            data_numpy = data_numpy[:, :, self.new_idx]

        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
