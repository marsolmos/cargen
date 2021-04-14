# example of a wgan for generating handwritten digits
import os
import cv2
import numpy as np
from numpy import expand_dims
from numpy import mean
from numpy import ones
from numpy.random import randn
from numpy.random import randint
import tensorflow as tf
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras import backend
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.constraints import Constraint
from matplotlib import pyplot


# Environment variables
base_dir = "D:/Data Warehouse/thecarconnection/pictures" # Base directory
category = "front" # Category that we want to use for image generation
IMG_WIDTH = 56 # Target width of images when being loaded (in pixels)
IMG_HEIGHT = 56 # Target height of images when being loaded (in pixels)

# clip model weights to a given hypercube
class ClipConstraint(Constraint):
	# set clip value when initialized
	def __init__(self, clip_value):
		self.clip_value = clip_value

	# clip model weights to hypercube
	def __call__(self, weights):
		return backend.clip(weights, -self.clip_value, self.clip_value)

	# get the config
	def get_config(self):
		return {'clip_value': self.clip_value}


# calculate wasserstein loss
def wasserstein_loss(y_true, y_pred):
	return backend.mean(y_true * y_pred)


# define the standalone critic model
def define_critic(in_shape=(28,28,1)):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# weight constraint
	const = ClipConstraint(0.01)
	# define model
	model = Sequential()
	# # downsample to 112x112
	# model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, kernel_constraint=const, input_shape=in_shape))
	# model.add(BatchNormalization())
	# model.add(LeakyReLU(alpha=0.2))
	# # downsample to 56x56
	# model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, kernel_constraint=const, input_shape=in_shape))
	# model.add(BatchNormalization())
	# model.add(LeakyReLU(alpha=0.2))
	# downsample to 28x28
	model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, kernel_constraint=const, input_shape=in_shape))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))
	# downsample to 14x14
	model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, kernel_constraint=const, input_shape=in_shape))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))
	# downsample to 7x7
	model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, kernel_constraint=const))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))
	# scoring, linear activation
	model.add(Flatten())
	model.add(Dense(1))
	# compile model
	opt = RMSprop(lr=0.00005)
	model.compile(loss=wasserstein_loss, optimizer=opt)
	return model


# define the standalone generator model
def define_generator(latent_dim):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# define model
	model = Sequential()
	# foundation for 7x7 image
	n_nodes = 128 * 7 * 7
	model.add(Dense(n_nodes, kernel_initializer=init, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((7, 7, 128)))
	# upsample to 14x14
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 28x28
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 56x56
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))
	# # upsample to 112x112
	# model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
	# model.add(BatchNormalization())
	# model.add(LeakyReLU(alpha=0.2))
	# # upsample to 224x224
	# model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
	# model.add(BatchNormalization())
	# model.add(LeakyReLU(alpha=0.2))
	# output 112x112x1
	model.add(Conv2D(1, (7,7), activation='tanh', padding='same', kernel_initializer=init))
	return model


# define the combined generator and critic model, for updating the generator
def define_gan(generator, critic):
	# make weights in the critic not trainable
	for layer in critic.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(generator)
	# add the critic
	model.add(critic)
	# compile model
	opt = RMSprop(lr=0.00005)
	model.compile(loss=wasserstein_loss, optimizer=opt)
	return model


def load_real_samples(base_dir, category, target_size=(112, 112)):
    '''
    Load real samples from category directory and preprocess them in the way the network expects them

    ## ARGS ##
    base_dir[str]: path to base directory where all images are saved into one folder per category
    categroy[str]: name of the folder with the images belonging to the category which we want to
                   use to train the system
    target_size[tuple]: target size if the images when being loaded.

    return[np.array]: array with images loaded and preprocessed. Expected output shape will be
                      (n_samples, width, heigth, channels). As images will be loaded as grayscale,
                      channels will always be equal to 1.
    '''
    # Categories directory where to find images to generate the dataset
    cat_dir = os.path.join(base_dir, category)
    # Iterate for all images
    imlist = []
    for file in os.listdir(cat_dir):
        img_path = os.path.join(cat_dir, file)
        if os.path.isdir(img_path):
            # Skip directories
            continue
        else:
            # Load image into grayscale
            im = image.load_img(img_path, target_size=target_size, color_mode="grayscale")
            # Convert image into array
            im = image.img_to_array(im)
            # Convert from ints to floats
            im = im.astype('float32')
            # Scale from [0,255] to [-1,1]
            im = (im - 127.5) / 127.5
            # Include into dataset
            imlist.append(im)
    # Generate dataset with all images in numpy.array format
    dataset = np.array(imlist)

    return dataset


# select real samples
def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# select images
	X = dataset[ix]
	# generate class labels, -1 for 'real'
	y = -ones((n_samples, 1))
	return X, y


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = generator.predict(x_input)
	# create class labels with 1.0 for 'fake'
	y = ones((n_samples, 1))
	return X, y


# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, latent_dim, base_dir, n_samples=100):
	# prepare fake examples
	X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
	# scale from [-1,1] to [0,1]
	X = (X + 1) / 2.0
	# plot images
	for i in range(10 * 10):
		# define subplot
		pyplot.subplot(10, 10, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
	# save plot to file
	image_name = 'generated_plot_%04d.png' % (step+1)
	image_path = os.path.join(base_dir, 'images')
	# create directory for images (if necessary)
	if not os.path.isdir(image_path):
		os.makedirs(image_path)
	image_path = os.path.join(image_path, image_name)
	pyplot.savefig(image_path)
	pyplot.close()
	# save the generator model
	model_path = os.path.join(base_dir, 'model_saved.h5')
	g_model.save(model_path)
	print('>Saved: %s and %s' % (image_path, model_path))
	return


# create a line plot of loss for the gan and save to file
def plot_history(d1_hist, d2_hist, g_hist, base_dir):
	# plot history
	pyplot.plot(d1_hist, label='crit_real')
	pyplot.plot(d2_hist, label='crit_fake')
	pyplot.plot(g_hist, label='gen')
	pyplot.legend()
	save_path = os.path.join(base_dir, 'images')
	# create directory for images (if necessary)
	if not os.path.isdir(save_path):
		os.makedirs(save_path)
	save_path = os.path.join(save_path, 'plot_line_plot_loss.png')
	pyplot.savefig(save_path)
	pyplot.close()
	return


# train the generator and critic
def train(g_model, c_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=64, n_critic=5):
	# base directory where to save generator model and generated images
	model_name = "models_" + str(latent_dim) +\
						"_" + str(n_epochs) +\
						"_" + str(n_batch) +\
						"_" + str(n_critic)
	# create directory for images and model saved (if necessary)
	base_dir = os.path.join('models', model_name)
	if not os.path.isdir(base_dir):
		os.makedirs(base_dir)
	# calculate the number of batches per training epoch
	bat_per_epo = int(dataset.shape[0] / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# calculate the size of half a batch of samples
	half_batch = int(n_batch / 2)
	# lists for keeping track of loss
	c1_hist, c2_hist, g_hist = list(), list(), list()
	# manually enumerate epochs
	for i in range(n_steps):
		# update the critic "n_critic times more" than the generator
		c1_tmp, c2_tmp = list(), list()
		for _ in range(n_critic):
			# get randomly selected 'real' samples
			X_real, y_real = generate_real_samples(dataset, half_batch)
			# update critic model weights
			c_loss1 = c_model.train_on_batch(X_real, y_real)
			c1_tmp.append(c_loss1)
			# generate 'fake' examples
			X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			# update critic model weights
			c_loss2 = c_model.train_on_batch(X_fake, y_fake)
			c2_tmp.append(c_loss2)
		# store critic loss
		c1_hist.append(mean(c1_tmp))
		c2_hist.append(mean(c2_tmp))
		# prepare points in latent space as input for the generator
		X_gan = generate_latent_points(latent_dim, n_batch)
		# create inverted labels for the fake samples
		y_gan = -ones((n_batch, 1))
		# update the generator via the critic's error
		g_loss = gan_model.train_on_batch(X_gan, y_gan)
		g_hist.append(g_loss)
		# summarize loss on this batch
		print('>%d, c1=%.3f, c2=%.3f g=%.3f' % (i+1, c1_hist[-1], c2_hist[-1], g_loss))
		# evaluate the model performance every 'epoch'
		if (i+1) % bat_per_epo == 0:
			summarize_performance(i, g_model, latent_dim, base_dir)
	# line plots of loss
	plot_history(c1_hist, c2_hist, g_hist, base_dir)
	return


# define GPU usage for training
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)

# define all grid search parameters
all_n_critic = [1, 2, 3, 4, 5, 6, 7, 8] # Number of times that the critic updates per each update of generator
all_latent_dim = [25, 50, 100, 150, 200] # Size of the latent space
all_n_epochs = [50] # Number of training epochs
all_n_batch = [8, 16, 32, 64, 128, 256] # Size of training batches

# iterate for all posible values
for n_critic in all_n_critic:
	for latent_dim in all_latent_dim:
		for n_epochs in all_n_epochs:
			for n_batch in all_n_batch:
				try:
					# create the critic
					critic = define_critic(in_shape=(IMG_WIDTH, IMG_HEIGHT, 1))
					# create the generator
					generator = define_generator(latent_dim)
					# create the gan
					gan_model = define_gan(generator, critic)
					# load image data
					dataset = load_real_samples(base_dir, category, target_size=(IMG_WIDTH, IMG_HEIGHT))
					# train model
					print('\n\n\n\n\n\n\n\n\n\n\n\n')
					print('TRAINING: n_crtic = {} | latent_dim = {} | n_batch = {}'.format(n_critic, latent_dim, n_batch))
					train(
						generator,
						critic,
						gan_model,
						dataset,
						latent_dim=latent_dim,
						n_epochs=n_epochs,
						n_batch=n_batch,
						n_critic=n_critic
						)
				except:
					pass
