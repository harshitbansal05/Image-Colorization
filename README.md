# Image Colorization
*Gray-scale Image Colorization of low resolution images using a Conditional Deep Convolutional Generative Adversarial Network (DCGAN).*
This is a PyTorch implementation of the Conditional DCGAN(Deep Convolutional Generative Adversarial Networks).

## Prerequisites
* Python 3.6
* PyTorch

## Method
In a traditional GAN, the input of the generator is randomly generated noise data z. However, this approach is not applicable to the automatic colorization problem due to the nature of its inputs. The generator must be modified to accept grayscale images as inputs rather than noise. This problem was addressed by using a variant of GAN called Conditional Generative Adversarial Network. Since no noise is introduced, the input of the generator is treated as zero noise with the grayscale input as a prior. The equations describing the cost function to train a DCGAN are as follows:

![CON-GAN](images/con_gan.png)

The discriminator gets colored images from both generator and original data along with the grayscale input as the condition and tries to tell which pair contains the true colored image. This is illustrated by the figure below:

![C-GAN](images/cgan.png)

## Network Architecture
The architecture of generator is inspired by U-Net: The architecture of the model is symmetric, with n encoding units and n decoding units. The architecture of the generator is shown below:

![U-NET](images/unet.png)

For discriminator, we use similar architecture as the baselines contractive path.

## Setup
1. Clone the source code and run `python3 train.py`. 

* It begins by downloading the Cifar-10 train dataset in the `/data` directory. The Cifar-10 train dataset consists of 50,000 images distributed equally across 10 classes like  *plane*, *bird*, *cat*, *ship*, *truck* to name a few and each image is of resolution *32x32*. The images are PIL images in the range [0, 1]. The color space of the images is changed from RGB to CieLAB. Unlike RGB, CieLAB consists of an illuminance channel which entirely contains the brightness information and two chrominance channels which contain the enitre color information. This prevents any sudden variations in both color and
brightness through small perturbations in intensity values that are experienced through RGB. Finally, a data loader is made over the transformed dataset.

* Finally, training begins. The training is carried for 100 epochs and the generator and discriminator models are saved periodically in directories  `/cifar10_train_generator` and `/cifar10_train_discriminator` respectively. 

2. To evaluate the model on the Cifar-10 test dataset, run `python3 eval.py`. It performs the evaluation on the entire test dataset and prints the mean absolute error(MAE: pixel-wise image distance between the source and the generated images) at the end. 

## Datasets
CIFAR-10 is used as the dataset for training. To train the model on the full dataset, download dataset [here](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz).

## Contributing
Suggestions and pull requests are actively welcome.

## References
1. Image Colorization using GANs. ([Paper](https://arxiv.org/pdf/1803.05400.pdf))
2. A TensorFlow Implementation of Image Colorization. ([Link](https://github.com/ImagingLab/Colorizing-with-GANs))
