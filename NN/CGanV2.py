
import tensorflow as tf
tf.__version__
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from tensorflow.keras.layers import Input, Dense, Concatenate, Reshape, Flatten
from tqdm import tqdm
import time
from IPython import display

noise_dim = 100
input_size = 110
clases_totales = 10
# input del discrimindador (28,28,2)
i1 = 28
i2 = 28
i3 = 2

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  print("Generate and save")
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()


  train(train_dataset, EPOCHS)
  checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
  display_image(EPOCHS)


  # Display a single image using the epoch number
def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[i1, i2, i3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
# ------------------------------------------------
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(input_size,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model


def train(dataset, epochs):
  # i = 0
  for epoch in range(epochs):
  # for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
    # i = i + 1
    # print("Epoca")
    # print(i)
    start = time.time()

    for image_batch,label_batch in dataset:
      
      # print("Train ")
      # print(i)
      # print(label_batch.shape[0])
      train_step(image_batch,label_batch)

    # Produce images for the GIF as you go
    # display.clear_output(wait=True)
    # generate_and_save_images(generator,
    #                          epoch + 1,
    #                          seed)
    print("Train --> termino generate")
    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)

units = 784   # M*M 28*28
dense_layer = Dense(units, activation='relu')





# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images,labels):
# def train_step(images):

    ruido = tf.random.normal([BATCH_SIZE, noise_dim])
    # clase_seleccionada = tf.random.uniform([BATCH_SIZE], minval=0, maxval=clases_totales, dtype=tf.int32) # clase_seleccionada representa a los labels
    # ruido_one_hot = tf.one_hot(clase_seleccionada, depth=clases_totales)
    # print("Labels")
    # print(labels)
    # print(labels.shape)
    labels_one_hot = tf.one_hot(labels, depth=clases_totales)
    # labels_one_hot_resized = tf.image.resize(labels_one_hot, (256, clases_totales))
    noise = tf.concat([ruido, tf.cast(labels_one_hot, ruido.dtype)], axis=-1)

    # noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

#-------------------------------------------------------------------------------
      # M = generated_images.shape
      M = generated_images.shape[1]

      flat_labels = Flatten()(labels)
      flattened_images = Flatten()(generated_images)

      dense_output = dense_layer(flat_labels)

      reshaped_output = Reshape((M, M, 1))(dense_output)



# Concatenar la etiqueta remodelada a la imagen

      imagen_reshaped = Concatenate(axis=-1)([images, reshaped_output])
      generated_reshaped = Concatenate(axis=-1)([generated_images, reshaped_output])

      real_output = discriminator(imagen_reshaped, training=True)
      fake_output = discriminator(generated_reshaped, training=True)
#-------------------------------------------------------------------------------
      # real_output = discriminator(images, training=True)
      # fake_output = discriminator(generated_images, training=True)
#-------------------------------------------------------------------------------
      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    print("Fin step")
# ---------------------------------------------




(train_images,train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

print(train_images.shape)
BUFFER_SIZE = 59904   # cantidad de imagenes del dataset
BATCH_SIZE = 256
# dataset = [DatasetEntry(image, label) for image, label in zip(train_images, train_labels)]

train_images = train_images[:BUFFER_SIZE] #tomo los primeros 59904 datos solamente
train_labels = train_labels[:BUFFER_SIZE]

dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


generator = make_generator_model()

discriminator = make_discriminator_model()
# decision = discriminator(generated_image)
# print (decision)


 # This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# --------------------------------------------
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
# ---------------------------------------------
EPOCHS = 1

num_examples_to_generate = 16

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, input_size])

train(train_dataset, EPOCHS)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
display_image(EPOCHS)