import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential, Model
from keras.optimizers import Adam
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import os

data_dir = 'F:\MLPROJECTS\shoreline3\\' 

img_width, img_height = 565, 54

batch_size = 29

datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True,
    seed=42
)
print(train_generator)

X_train = np.zeros((train_generator.samples, img_width, img_height, 3), dtype=np.float32)
print(X_train.shape)
num_batches = len(train_generator)
batch_size = train_generator.batch_size

for i in range(num_batches):
    image_batch = train_generator.next()[0]
    for j in range(batch_size):
        image_i = image_batch[j]
        image_i = image_i.reshape(img_width, img_height, 3)
        X_train[i*batch_size + j] = image_i

print(X_train.shape)

X_train = (X_train - 0.5) * 2.0

def build_generator(latent_dim, img_shape):
    model = Sequential()
    
    
    model.add(layers.Dense(128, input_dim=latent_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    
    
    model.add(layers.Dense(np.prod(img_shape), activation="tanh"))
    model.add(layers.Reshape(img_shape))
    
   
    return model

def build_discriminator(img_shape):
    model = Sequential()
    
    model.add(layers.Flatten(input_shape=img_shape))
    
    
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(128))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    
    model.add(layers.Dense(1, activation="sigmoid"))
    
    
    return model

latent_dim = 100
img_shape = X_train.shape[1:]

generator = build_generator(latent_dim, img_shape)
discriminator = build_discriminator(img_shape)

# Combine the generator and discriminator into a GAN
discriminator.trainable = False
gan_input = keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = Model(gan_input, gan_output)
optimizer = Adam(lr=0.0002, beta_1=0.5)
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
def train_gan(generator, discriminator, gan, X_train, latent_dim, epochs=1000, batch_size=29):
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    gan.compile(optimizer='adam', loss='binary_crossentropy')
    for epoch in range(epochs):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_images = X_train[idx]
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_images = generator.predict(noise)
        d_loss_real = discriminator.train_on_batch(real_images, real)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, real)
        print(f"Epoch {epoch+1}/{epochs}: Discriminator loss: {d_loss}, Generator loss: {g_loss}")

train_gan(generator, discriminator, gan, X_train, latent_dim, epochs=1000, batch_size=29)

noise = np.random.normal(0, 1, (1000, latent_dim))
generated_images = generator.predict(noise)
generated_images = generated_images * 127.5 + 127.5

dir_path = 'F:\MLPROJECTS\\generatedless\\'

for i in range(min(1000, generated_images.shape[0])):
    try:
        img = generated_images[i, :, :, :]
        img = np.clip(img, 0, 255).astype("uint8")
         # swap dimensions
        img = np.transpose(img, (1, 0, 2))
        # resize image to desired dimensions
        img = keras.preprocessing.image.smart_resize(img, size=(54, 565))
        filename = "generated_image_{}.png".format(i)
        keras.preprocessing.image.save_img(os.path.join(dir_path, filename), img)
    except Exception as e:
        print(f"Error processing image {i}: {e}")


