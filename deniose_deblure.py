import random
import numpy as np
from . import sol5_utils
#import sol5_utils
from scipy.ndimage.filters import convolve
from tensorflow.keras.layers import Input, Conv2D, Activation, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from scipy import misc
from skimage.color import rgb2gray
import matplotlib.pyplot as plt




def read_image(filename, representation):
    """
        This function reads an image file and
        converts it into a given representation.
        :param filename: image file
        :param representation: (1) - grayscale image
        (2) - RGB image
        :return: image type np.float64
        """
    if representation == 1:
        im_float = misc.imread(filename).astype(np.float64)
        im_float = rgb2gray(im_float)
    else:
        im_float = misc.imread(filename, mode= 'RGB').astype(np.float64)
    im_float /= 255.
    return im_float

def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """
    :param filenames: List of file names
    :param batch_size: Number of images to return from generator at each time
    :param corruption_func: Function of type Im -> Im for applying corruptions
    :param crop_size: A tuple (height, width) specify the crop size of
    patches to extract

    :return Python generator. Outputs random tuples of the form (
    source_batch, target_batch) where each output variable is a Numpy array
    of shape (batch_size, 1, height, width), "target_batch" is made of clean
    images, and "source_batch" is their respective randomly corrupted version
    according to "corruption_func(im)".
    """

    # initializing a dictionary for faster computations
    image_cache = {}

    def read_image_with_cache(filename):
        if filename not in image_cache:
            image_cache[filename] = read_image(filename, 1)
        return image_cache[filename]

    def rand_index(max_index, els):
        return np.random.randint(0, max_index, els)

    while True:
        crop_height, crop_width = crop_size
        source_batch = np.zeros((batch_size, crop_height, crop_width, 1),
                                dtype=np.float64)
        target_batch = np.zeros((batch_size, crop_height, crop_width, 1),
                                dtype=np.float64)

        random_indices = rand_index(len(filenames), batch_size)
        curr_step = 0
        for index in random_indices:
            curr_image = read_image_with_cache(filenames[index])
            curr_image_height = curr_image.shape[0]
            curr_image_width = curr_image.shape[1]

            corrupted_image = corruption_func(curr_image)

            y_rand = rand_index(curr_image_height // crop_height, 1)[0] * crop_height
            x_rand = rand_index(curr_image_width // crop_width, 1)[0] * crop_width
            y_rand = y_rand if y_rand + crop_height <= curr_image_height else y_rand - crop_height
            x_rand = x_rand if x_rand + crop_width <= curr_image_width else x_rand - crop_width

            image_crop = curr_image[y_rand: crop_height + y_rand, x_rand:crop_width + x_rand]
            corrupted_image_crop = corrupted_image[y_rand:crop_height + y_rand, x_rand:crop_width + x_rand]

            target_batch[curr_step, :, :, 0] = image_crop - 0.5
            source_batch[curr_step, :, :, 0] = corrupted_image_crop - 0.5
            curr_step += 1

        yield source_batch, target_batch


def resblock(input_tensor, num_channels):
    """

    :param input_tensor: multidimensional array
    :param num_channels: int
    :return: block of network
    """
    a = input_tensor
    b = Conv2D(num_channels, (3, 3), padding='same')(a)
    c = Activation('relu')(b)
    d = Conv2D(num_channels, (3, 3), padding='same')(c)
    d = Add()([a, d])
    return Activation('relu')(d)


def build_nn_model(height, width, num_channels, num_res_blocks):
    """

    :param height: int
    :param width: int
    :param num_channels: int
    :param num_res_blocks: int
    :return: neural network model to be trained
    """
    a = Input(shape=(height, width, 1))
    b = Conv2D(num_channels, (3, 3), padding='same')(a)
    c = Activation('relu')(b)

    for i in range(num_res_blocks):
        c = resblock(c, num_channels)

    d = Conv2D(1, (3, 3), padding='same')(c)
    d = Add()([a, d])
    return Model(inputs=a, outputs= d)


def train_model(model, images, corruption_func,
                batch_size, steps_per_epoch, num_epochs, num_valid_samples):
    """

    :param model: neural network model to be trained
    :param images: a list of file paths pointing to image files
    :param corruption_func: same as described in section 3
    :param batch_size: int
    :param steps_per_epoch: int
    :param num_epochs: int
    :param num_valid_samples: int
    :return: nothing
    """
    N = len(images)
    split_ratio = int(N * 0.8)
    training_set = images[:split_ratio]
    validation_set = images[split_ratio:]
    crop_size = model.input_shape[1:3]

    training_batch_generator = load_dataset(training_set, batch_size,
                                            corruption_func, crop_size)

    validation_batch_generator = load_dataset(validation_set, batch_size,
                                              corruption_func, crop_size)

    model.compile(loss='mean_squared_error', optimizer=Adam(beta_2=0.9))

    model.fit_generator(training_batch_generator,
                        steps_per_epoch=steps_per_epoch, epochs=num_epochs,
                        validation_data=validation_batch_generator,
                        validation_steps=num_valid_samples / batch_size)

def restore_image(corrupted_image, base_model):
    """

    :param corrupted_image:a grayscale image of shape (height, width)
     and with values in the [0, 1] range
    :param base_model: a neural network trained to restore small patches
    :return: restored image grayscale image of shape (height, width)
     and with values in the [0, 1] range
    """

    height, width = corrupted_image.shape
    a = Input(shape=(height, width, 1))
    b = base_model(a)
    new_model = Model(inputs=a, outputs=b)
    x = (corrupted_image - 0.5)[np.newaxis, :, :,np.newaxis]
    y = new_model.predict(x)[0,:,:,0]
    return (0.5 + y).clip(0,1).astype(np.float64)


def add_gaussian_noise(image, min_sigma, max_sigma):
    """
    :param image: grayscale image
    :param min_sigma: a non-negative scalar value representing
     the minimal variance of the gaussian distribution
    :param max_sigma: a non-negative scalar value larger than or
     equal to min_sigma, representing the maximal
    variance of the gaussian distributio
    :return: grayscale image
    """
    sigma = np.random.uniform(min_sigma,max_sigma)
    corrupted= np.random.normal(0, sigma, image.shape)
    return (((image + corrupted)*255).round() / 255).clip(0,1)

def learn_denoising_model(num_res_blocks=5, quick_mode =False):
    """

    :param num_res_blockes: int
    :param quick: boolien
    :return: fitted model
    """
    if not quick_mode:
        batch_size = 100
        steps_per_epoch = 100
        num_epochs = 5
        num_valid_samples = 1000

    else:
        batch_size = 10
        steps_per_epoch = 3
        num_epochs = 2
        num_valid_samples = 30

    new_model = build_nn_model(24, 24, 48, num_res_blocks)
    images = sol5_utils.images_for_denoising()
    corruption_func = lambda im: add_gaussian_noise(im, 0, 0.2)
    train_model(new_model, images, corruption_func, batch_size,
                steps_per_epoch, num_epochs, num_valid_samples)
    return new_model


def add_motion_blur(image, kernel_size, angle):
    """

    :param image: grayscale image
    :param kernel_size: int
    :param angle: value [0,pi)
    :return: filtered image
    """
    kernel = sol5_utils.motion_blur_kernel(kernel_size, angle)
    return convolve(image, kernel)

def random_motion_blur(image, list_of_kernel_sizes):
    """

    :param image: grayscale image
    :param list_of_kernel_sizes: list of ints
    :return: filtered image
    """
    angle = random.uniform(0, np.pi)
    kernel_size_idx = random.randint(0, len(list_of_kernel_sizes)-1)
    kernel_size = list_of_kernel_sizes[kernel_size_idx]
    return ((add_motion_blur(image, kernel_size, angle)*255).round() / 255 ).clip(0,1)


def learn_deblurring_model(num_res_blocks=5, quick_mode = False):
    """

    :param num_res_blockes:
    :param qick_mode:
    :return:
    """
    if not quick_mode:
        batch_size = 100
        steps_per_epoch = 100
        num_epochs = 10
        num_valid_samples = 1000

    else:
        batch_size = 10
        steps_per_epoch = 3
        num_epochs = 2
        num_valid_samples = 30

    new_model = build_nn_model(16, 16, 32, num_res_blocks)
    images = sol5_utils.images_for_denoising()
    corruption_func = lambda im: random_motion_blur(im, [7])
    train_model(new_model, images, corruption_func, batch_size,
                       steps_per_epoch, num_epochs, num_valid_samples)
    return new_model


	
	
	
