#Project Name - GAN MODEL
#Author - Rhythm
#Course - COMP 3710
#TRUID - T00684614
#Thompson Rivers University

#Importing all vital librarues
import tensorflow as tf
import keras
import numpy as np
import cv2
import os
from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose, BatchNormalization, LeakyReLU

#Loading 7000 images from own directory
data_dir = r'F:\GAN_Project\data\resized_paintings'
imgs = []

i = 0
for img_file in os.listdir(data_dir):
    img = cv2.imread(os.path.join(data_dir, img_file))
    #Images have already been resized to 224 by 224 in this directory
    imgs.append(img)
    i = i+1
    if (i %1000 == 0):
        print("Done loading ", i, "images!")
    if (i==7000):
        break
        
# Define Generator and Discriminator
def make_generator_model():
    model = tf.keras.Sequential()
    # Dense layer with 7x7x256 output shape
    # This layer takes the 100-dimensional input vector and transforms it into a higher dimensional representation of size 7x7x256.
    model.add(Dense(7*7*256, use_bias=False, input_shape=(100,)))

    # This layer normalizes the activations of the previous layer, which helps to stabilize the learning process and speed up training.
    model.add(BatchNormalization())

    # This activation function introduces a small negative slope in the negative region of the activation, which helps to prevent the
    # "dying ReLU" problem and allows the generator to learn more complex features.
    model.add(LeakyReLU())
    # This layer reshapes the output of the previous layer into a 4D tensor of size (batch_size, 7, 7, 256).
    model.add(Reshape((7, 7, 256)))
    
    # Transpose convolutional layer with 14x14x128 output shape
    model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    
    # Transpose convolutional layer with 28x28x64 output shape
    model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    
    # Transpose convolutional layer with 56x56x32 output shape
    model.add(Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    print(model.output_shape)
    
    # Transpose convolutional layer with 112x112x16 output shape
    model.add(Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    
    # Transpose convolutional layer with 224x224x3 output shape
    model.add(Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    # Transpose convolutional layer with 224x224x3 output shape
    model.add(Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='sigmoid'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    print(model.output_shape)
    
    return model


 
def make_discriminator_model():
    
    model = tf.keras.Sequential()

    # Conv2D layer with 64 filters, kernel size of (5, 5), stride of (2, 2), and padding='same'. The input shape is [224, 224, 3],
    # which corresponds to an RGB image with height and width of 224 pixels. This layer applies 64 convolution filters to
    # the input image and outputs a feature map with the same height and width as the input but with 64 channels.
    model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[224, 224, 3]))
    
    # Allows small negative values to pass through the network, which can help prevent the "dying ReLU" problem.
    model.add(tf.keras.layers.LeakyReLU())
    
    # This layer randomly drops out 30% of the previous layer's outputs during training to prevent overfitting.
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    # This layer flattens the output from the previous layer into a 1D tensor.
    model.add(tf.keras.layers.Flatten())
    
    # This layer performs a dot product between the flattened tensor from the previous layer and a weight matrix
    # to produce a scalar output, which represents the probability of the input image being real or fake.
    model.add(tf.keras.layers.Dense(1))

    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator = make_generator_model();
# Compiling it with the Adam optimizer and binary cross-entropy loss. The optimizer is Adam with a learning rate of 0.002 and a beta1 value of 0.5.
generator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002, beta_1=0.5), loss='binary_crossentropy')

discriminator = make_discriminator_model();
# Compiling it with the Adam optimizer and binary cross-entropy loss. The optimizer is the same as for the generator.
discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002, beta_1=0.5), loss='binary_crossentropy')

# Define GAN
def gan_model(generator, discriminator):
    discriminator.trainable = False
    model = tf.keras.Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002, beta_1=0.5), loss='binary_crossentropy')
    return model

# Train GAN
def train_gan(generator, discriminator, gan, imgs, epochs=230, batch_size=256):
    X_train = np.array(imgs) / 127.5 - 1.
    y_train = np.ones((batch_size, 1))
    noise_dim = 100
    steps_per_epoch = X_train.shape[0] // batch_size
    
    for epoch in range(epochs):
        d_loss_total = 0
        g_loss_total = 0
        for step in range(steps_per_epoch):
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_imgs = X_train[idx]
            noise = np.random.normal(0, 1, (batch_size, noise_dim))
            fake_imgs = generator.predict(noise)

            # Train the discriminator
            d_loss_real = discriminator.train_on_batch(real_imgs, y_train)
            d_loss_fake = discriminator.train_on_batch(fake_imgs, np.zeros((batch_size, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            d_loss_total += d_loss

            # Train the generator
            noise = np.random.normal(0, 1, (batch_size, noise_dim))
            y_gan = np.ones((batch_size, 1))
            g_loss = gan.train_on_batch(noise, y_gan)
            g_loss_total += g_loss

        # Print the average loss for each epoch
        d_loss_avg = d_loss_total / steps_per_epoch
        g_loss_avg = g_loss_total / steps_per_epoch
        print("Epoch:", epoch, "D Loss:", d_loss_avg, "G Loss:", g_loss_avg)
        if (epoch % 5 == 0):
            output_dir = "F:\GAN_Project\generated_images\output_image" + str(epoch) + ".jpeg"
            # Using the generation program to save outputs every 5 epochs
            generate_renaissance_portrait(generator, "input_image.jpeg", output_dir)
            

# Generate Renaissance Portrait
def generate_renaissance_portrait(generator, img_path, save_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.array(img) / 127.5 - 1.
    noise = np.random.normal(0, 1, (1, 100))
    gen_img = generator.predict(noise)[0]
    gen_img = (gen_img + 1) * 127.5
    gen_img = cv2.cvtColor(gen_img, cv2.COLOR_RGB2BGR)
    h, w, _ = img.shape
    gen_img = cv2.resize(gen_img, (w, h))
    img = img.astype(gen_img.dtype)

    # Using addWeighted exclusively to get more definition
    out = cv2.addWeighted(img, 0.5, gen_img, 0.5, 0)
    cv2.imwrite(save_path, out)


#Create the GAN
gan = gan_model(generator, discriminator)

#Train the GAN
train_gan(generator, discriminator, gan, imgs)

# Using the generation program to convert normal photos to Renaissance paintings
# Portraits
generate_renaissance_portrait(generator, "input_image1.jpeg", "output_image1.jpg")
generate_renaissance_portrait(generator, "input_image2.jpeg", "output_image2.jpg")
generate_renaissance_portrait(generator, "input_image3.jpeg", "output_image3.jpg")
generate_renaissance_portrait(generator, "input_image4.jpg", "output_image4.jpg")

# Landscapes
generate_renaissance_portrait(generator, "input_image5.jpeg", "output_image5.jpg")
generate_renaissance_portrait(generator, "input_image6.jpeg", "output_image6.jpg")
generate_renaissance_portrait(generator, "input_image7.jpeg", "output_image7.jpg")


