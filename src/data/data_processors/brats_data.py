from torch.utils.data import Dataset

class BraTS2020Data(Dataset):

    def __init__(self, inputs, segmentations, augmentation=None):

        self.inputs = inputs
        self.segmentations = segmentations
        self.augmentation_operation = augmentation


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        sample = {'data': self.inputs[idx], 'label': self.segmentations[idx]}

        if self.augmentation_operation:
            sample = self.augmentation_operation(sample)

        return sample