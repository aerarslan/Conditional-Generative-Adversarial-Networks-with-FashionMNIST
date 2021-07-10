#%% KERAS IMPORTS
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Embedding, multiply
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
#%% BASIC IMPORTS
import numpy as np
import gzip
import matplotlib.pyplot as plt

#%% IMPORT DATASET
with gzip.open("dataset\\fashion_mnist_labels.gz", 'rb') as label_path:
    labels = np.frombuffer(label_path.read(), dtype=np.uint8,offset=8)
with gzip.open("dataset\\fashion_mnist_images.gz", 'rb') as image_path:
    images = np.frombuffer(image_path.read(), dtype=np.uint8,offset=16).reshape(len(labels), 28,28)

#%% PREPARE CLASS MAP BASED ON https://github.com/zalandoresearch/fashion-mnist
classes = {0:"T-shirt/top", 1:"Trouser", 2:"Pullover", 3:"Dress", 4:"Coat", 5:"Sandal", 6:"Shirt", 7:"Sneaker",8:"Bag",9:"Ankle boot"}

#%% BASIC VARIABLES
image_rows = 28
image_colomns = 28
channel = 1
image_shape = (image_rows, image_colomns, channel)
num_features = 100
num_classes = 10
optimizer = Adam(0.0002, 0.5)

#%% GENERATOR Conditional GAN with Dense Layers

generator_model = Sequential()

#cnn
generator_model.add(Dense(7 * 7 * 256, activation='relu', input_dim=num_features))
generator_model.add(Reshape((7, 7, 256)))
generator_model.add(BatchNormalization(momentum=0.8))
generator_model.add(UpSampling2D())
generator_model.add(Conv2D(256, kernel_size=3, padding='same'))
generator_model.add(Activation('relu'))
generator_model.add(BatchNormalization(momentum=0.8))
generator_model.add(UpSampling2D())
generator_model.add(Conv2D(128, kernel_size=3, padding='same'))
generator_model.add(Activation('relu'))
generator_model.add(BatchNormalization(momentum=0.8))
generator_model.add(Conv2D(64, kernel_size=3, padding='same'))
generator_model.add(Activation('relu'))
generator_model.add(BatchNormalization(momentum=0.8))
generator_model.add(Conv2D(1, kernel_size=3, padding='same'))
generator_model.add(Activation("tanh"))

# #dense
# generator_model.add(Dense(256, input_dim=num_features))
# generator_model.add(LeakyReLU(alpha=0.2))
# generator_model.add(BatchNormalization(momentum=0.8))
# generator_model.add(Dense(512))
# generator_model.add(LeakyReLU(alpha=0.2))
# generator_model.add(BatchNormalization(momentum=0.8))
# generator_model.add(Dense(1024))
# generator_model.add(LeakyReLU(alpha=0.2))
# generator_model.add(BatchNormalization(momentum=0.8))
# generator_model.add(Dense(np.prod(image_shape), activation='tanh'))

generator_model.add(Reshape(image_shape))
random_noise = Input(shape=(num_features,))
label_g = Input(shape=(1,), dtype='int32')
embedding_label = Flatten()(Embedding(num_classes, num_features)(label_g))

generator_model_input = multiply([random_noise, embedding_label])
image_g = generator_model(generator_model_input)

generator_CGAN = Model([random_noise,label_g], image_g)

#%% DISCRIMINATOR Conditional GAN

discriminator_model = Sequential()

discriminator_model.add(Dense(1024, input_dim=np.prod(image_shape)))
discriminator_model.add(LeakyReLU(alpha=0.2))
discriminator_model.add(Dense(512))
discriminator_model.add(LeakyReLU(alpha=0.2))
discriminator_model.add(Dense(256))
discriminator_model.add(LeakyReLU(alpha=0.2))
discriminator_model.add(Dense(128))
discriminator_model.add(LeakyReLU(alpha=0.2))
discriminator_model.add(Dropout(0.5))
discriminator_model.add(Dense(1, activation='sigmoid'))

image_d = Input(shape=image_shape)
label_d = Input(shape=(1,), dtype='int32')
embedding_label = Flatten()(Embedding(num_classes, np.prod(image_shape))(label_d))
flat_image = Flatten()(image_d)
discriminator_model_input = multiply([flat_image, embedding_label])
validity = discriminator_model(discriminator_model_input)

discriminator_CGAN = Model([image_d,label_d], validity)

discriminator_CGAN.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            
#%% TRAIN

input_z = Input(shape=(num_features,))
label = Input(shape=(1,))

image = generator_CGAN([input_z,label])
discriminator_CGAN.trainable = False
validity = discriminator_CGAN([image,label])

merged_CGAN = Model([input_z,label], validity)
merged_CGAN.compile(loss='binary_crossentropy', optimizer=optimizer)

batch_size = 32
epochs= 100000
save_interval = 1000

#arrays for the plot
d_real_l = []
d_fake_l = []
gen_l = []

# Rescale
images = images / 127.5 - 1.
images = np.expand_dims(images, axis=3)
labels = labels.reshape(-1, 1)

discriminator_real = np.ones((batch_size, 1))
discriminator_fake = np.zeros((batch_size, 1))

for epoch in range(epochs):

    # * TRAIN DISCRIMINATOR * 

    # A RANDOM BATCH OF IMAGES ARE SELECTED
    random_id = np.random.randint(0, images.shape[0], batch_size)
    image, label = images[random_id], labels[random_id]

    random_noise = np.random.normal(0, 1, (batch_size, num_features))

    # Generate a batch of new images
    generator_images = generator_CGAN.predict([random_noise, label])
    discriminator_loss_real = discriminator_CGAN.train_on_batch([image, label], discriminator_real)
    discriminator_loss_fake = discriminator_CGAN.train_on_batch([generator_images, label], discriminator_fake)
    discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)

    # * TRAIN GENERATOR *

    sampled_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)

    generator_loss = merged_CGAN.train_on_batch([random_noise, sampled_labels], discriminator_real)
    
    # assign losses to the arrays
    d_real_l.append(discriminator_loss[0])
    d_fake_l.append(discriminator_loss[1])
    gen_l.append(generator_loss)
    
    # PLOT THE PROGRESS
    if epoch % 100 == 99:
        print ("Epoch: %d Discriminator loss: %f, accuracy: %.2f%% Generator loss: %f" % ((epoch+1), discriminator_loss[0], discriminator_loss[1]*100, generator_loss))

    row = 2
    column = 5
    
    # in every save_interval, save newly generated objects and the generator-discriminator models
    if epoch % save_interval == (save_interval-1):
        
        # SAVE GENERATED IMAGES
        noise = np.random.normal(0, 1, (row*column, num_features))
        sampled_labels = np.arange(0, 10).reshape(-1, 1)
        generator_images = generator_CGAN.predict([noise, sampled_labels])
        generator_images = 0.5 * generator_images + 0.5
        fig, axs = plt.subplots(row, column)
        cnt = 0
        for i in range(row):
            for j in range(column):
                axs[i,j].imshow(generator_images[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
                
        # starting epoch to save models
        startSaving = 3000
        
        # SAVE MODELS
        if(epoch>startSaving):
            discriminator_CGAN_json = discriminator_CGAN.to_json()
            with open("models/discriminator" + str(epoch+1) + "_CGAN.json", "w") as json_file:
                json_file.write(discriminator_CGAN_json)
            discriminator_CGAN.save_weights("models/discriminator" + str(epoch+1) + "_CGAN_model.h5")
            generator_CGAN_json = generator_CGAN.to_json()
            with open("models/generator" + str(epoch+1) + "_CGAN.json", "w") as json_file:
                json_file.write(generator_CGAN_json)
            # serialize weights to HDF5
            generator_CGAN.save_weights("models/generator" + str(epoch+1) + "_CGAN_model.h5")
            
        # DRAW GENERATED IMAGES
        plt.savefig("epoch-figure\\" + "epoch" + str(epoch+1) +".png")
        plt.show()
        plt.close()
        
#%%

# DATA PLOTTING

d_fake_l = d_fake_l[0::100]
d_real_l = d_real_l[0::100]
gen_l = gen_l[0::100]
epochs = np.arange(1, len(gen_l)+1)

# plotting Loss
plt.title("Loss Plot")
plt.xlabel("Epochs")
plt.ylabel("")
plt.plot(epochs, d_fake_l, color ="#FD9A02", label='Discriminator Fake Loss')
plt.plot(epochs, d_real_l, color ="blue", label='Discriminator Real Loss')
plt.plot(epochs, gen_l, color ="green", label='Generator Loss')
plt.legend()
plt.show()

# # plotting Accuracy
# plt.title("Accuracy Plot")
# plt.xlabel("Epochs")
# plt.ylabel("")
# plt.plot(epochs, d_fake_l, color ="#FD9A02", label='Fake Accuracy')
# plt.plot(epochs, d_real_l, color ="blue", label='Real Accuracy')
# plt.legend()
# plt.show()

