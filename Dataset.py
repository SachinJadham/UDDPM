import glob

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils
from skimage import transform as skTransform

import matplotlib.pyplot as plt
from PIL import Image

import cv2


# plt.ion()  # interactive mode for plots


# =========================== Image Transformations ==================== #
class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        np_image, cl_id = sample['image'], sample['class']

        # h, w, c = np_image.shape
        # if isinstance(self.output_size, int):
        #     if h > w:
        #         new_h, new_w = self.output_size * h / w, self.output_size
        #     else:
        #         new_h, new_w = self.output_size, self.output_size * w / h
        # else:
        #     new_h, new_w = self.output_size
        #
        # new_h, new_w = int(new_h), int(new_w)
        #
        # img = skTransform.resize(np_image, (new_h, new_w))
        img = skTransform.resize(np_image, (self.output_size, self.output_size))

        return {'image': img, 'class': cl_id}


class ToTensor(object):
    def __call__(self, sample):
        np_image, cl_id = sample['image'], sample['class']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = np_image.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image), 'class': torch.from_numpy(cl_id)}


class RandomResizedCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        np_image, cl_id = sample['image'], sample['class']

        h, w = np_image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        image = np_image[top: top + new_h, left: left + new_w]

        return {'image': image, 'class': cl_id}


class NormalizeImage(object):
    # Normalize PIL image which is in range [0,1] to range [-1,1] along all 3 Dims.
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        np_image, cl_id = sample['image'], sample['class']

        normalized_image = (np_image - self.mean) / self.std  # across all 3 channels

        return {'image': normalized_image, 'class': cl_id}


# =========================== Custom TensorDataset ==================== #
class TensorDataset(Dataset):
    def __init__(self, root_dir, transform_list=None):
        self.data = []  # converted to numpy array below.
        self.classMap = {"0": 0}  # dict for mapping sub-folder names to integers.
        self.transforms = transform_list

        # read all the files in the given path; [image path, class] * x
        file_list = glob.glob(root_dir + "/*")
        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            for img_path in glob.glob(class_path + "/*.png"):
                self.data.append([img_path, class_name])
        self.data = np.array(self.data)  # convert to numpy array for better indexing.
        print(f'Read {len(self.data)} Images and {len(np.unique(self.data[:, 1]))} classes from {root_dir}')

        # Transformation list to be applied to the images in getitem
        self.transforms = self.transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path, class_name = self.data[idx]
        img = Image.open(img_path)  # open image as PIL object  # scales [0,255]->[0,1]
        np_img = np.asarray(img)  # convert to numpy array

        class_id = np.asarray(self.classMap[class_name])

        sample = {'image': np_img, 'class': class_id}

        if self.transforms:
            # transforms.Compose will apply the transformation in the specified order,
            # so you can use it directly
            sample = self.transforms(sample)

        # return sample   # Example on how to index; keys of sample_dictionary -> ['image', 'class'].
        return sample['image']  # only return Images No Labels.


if __name__ == "__main__":
    # ------------------- TensorDataset ------------------ #
    # create a 'TensorDataset'; for easy iteration, indexing and slicing along the First dimension of [data and labels]
    train_ds = TensorDataset(r'/home/i_sjadham77/UUDPM/streetview3k',
                             transform_list=transforms.Compose([Rescale(416),
                                                                RandomResizedCrop(416),
                                                                NormalizeImage(mean=0.5, std=0.5),
                                                                ToTensor()]))

    # # plot a sample image
    # image = train_ds[547]
    # print(image.shape)
    # print(image.dtype)
    # np_image = image.numpy()  # convert tensor to numpy
    # np_image = np_image.transpose((1, 2, 0))  # C x H x W  -> H x W x C
    # np_image = cv2.normalize(np_image, None, alpha=0.001, beta=1, norm_type=cv2.NORM_MINMAX)    # [0, 1] range
    # plt.imshow(np_image)
    # plt.show()

    # ===================== check for data and transformation validity ========================== #
    from tqdm import tqdm
    for i, image in enumerate(tqdm(train_ds)):
        i=i
