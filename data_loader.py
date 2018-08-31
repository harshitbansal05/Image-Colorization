import torch
import torchvision
import torchvision.transforms as transforms

import cv2
import numpy as np 

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors.
tensor_transform = transforms.ToTensor()

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=tensor_transform)

classes = ('plane', 'car', 'bbird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#########################################################################
# Transform the images to CieLAB color space by the use of OpenCV library.
rgb_images = []
numpy_lab_images = []
for image, label in trainset:
    rgb_images.append(image)

for rgb_image in rgb_images:
    numpy_rgb_image = np.transpose(rgb_image.numpy(), (1, 2, 0))
    numpy_lab_image = cv2.cvtColor(numpy_rgb_image, cv2.COLOR_RGB2LAB)
    numpy_lab_images.append(numpy_lab_image)

######################################################################
# Transform the numpy lab images to images of range [0, 1] and further
# convert them to tensors.
lab_images = []
for numpy_lab_image in numpy_lab_images:
    numpy_lab_image[:, :, 0] *= 255 / 100
    numpy_lab_image[:, :, 1] += 128
    numpy_lab_image[:, :, 2] += 128
    numpy_lab_image /= 255
    torch_lab_image = torch.from_numpy(np.transpose(numpy_lab_image, (2, 0, 1)))
    lab_images.append(torch_lab_image)

#######################################################################
# Make a custom CieLAB dataset and a data loader that iterates over the
# custom dataset with shuffling and a batch size of 128.
class CieLABDataset(torch.utils.data.Dataset):
    """CieLab dataset."""    
    def __len__(self):
        return len(lab_images)

    def __getitem__(self, index):
        img = lab_images[index]
        return img

cielab_dataset = CieLABDataset()
cielab_loader = torch.utils.data.DataLoader(cielab_dataset, batch_size=128,
                  shuffle=True, num_workers=2)

