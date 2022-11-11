"""
Transform and split dataset
"""
import torchvision as tv
import torchvision.transforms.functional as TF
import torch.utils.data as td
import sklearn.model_selection as skms
import numpy as np
from data.cub_200_2011 import Dataset

RANDOM_SEED = 42
params = {'batch_size': 24, 'num_worker': 8}


def pad(img, fill=0, size_max=500):
    """
    Pad images to the specific size
    Fill up the padded area with the value from the 'fill' parameter
    """

    pad_height = max(0, size_max - img.height)
    pad_width = max(0, size_max - img.width)

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    return TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=fill)


# fill padded area with ImageNet's mean pixel value converted to range[0, 255]
fill = tuple(map(lambda x: int(round(x * 256)), (0.485, 0.456, 0.406)))
# pad imgaes to 500 pixels
max_padding = tv.transforms.Lambda(lambda x: pad(x, fill=fill))

# transform images
transforms_train = tv.transforms.Compose([
    max_padding,
    tv.transforms.RandomOrder([
        tv.transforms.RandomCrop((375, 375)),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.RandomVerticalFlip()
    ]),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transforms_eval = tv.transforms.Compose([
    max_padding,
    tv.transforms.CenterCrop((375, 375)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# separate dataset into three subsets to make sure the proper training and evaluation for model
ds_train = Dataset('D:\\ws\\ai\CUB_200_2011', transform=transforms_train, train=True)
ds_val = Dataset('D:\\ws\\ai\CUB_200_2011', transform=transforms_eval, train=True)
ds_test = Dataset('D:\\ws\\ai\CUB_200_2011', transform=transforms_eval, train=False)

splits = skms.StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=RANDOM_SEED)
idx_train, idx_val = next(splits.split(np.zeros(len(ds_train)), ds_train.targets))

# instantiate the data loaders
train_loader = td.DataLoader(
    dataset=ds_train,
    sampler=td.SubsetRandomSampler(idx_train),
    batch_size=24
)
val_loader = td.DataLoader(
    dataset=ds_val,
    sampler=td.SubsetRandomSampler(idx_val),
    batch_size=24
)
test_loader = td.DataLoader(dataset=ds_test, batch_size=24)
