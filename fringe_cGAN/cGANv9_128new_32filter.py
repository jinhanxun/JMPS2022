import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D,Conv2DTranspose,UpSampling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
from numpy import asarray
import glob

from random import randint
import random
import datetime


# In[122]:


def plot_results(images, epoch, label, n_cols=None):
    '''visualizes fake images'''
    
    n_cols = n_cols or len(images)
    n_rows = (len(images) - 1) // n_cols + 1
    
    if images.shape[-1] == 1:
        images = np.squeeze(images, axis=-1)
    
    plt.figure(figsize=(n_cols, n_rows))
    
    
    
    filename = 'Epoch_'+str(epoch)+ 'label_'+str(label)
    
    for index, image in enumerate(images):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(image, cmap='gray')
        plt.axis("off")

    plt.suptitle('Epoch: '+str(epoch))    
    plt.savefig('./cGAN_output/' + filename +'.png', bbox_inches='tight',dpi=100)
    plt.close()


# In[123]:


# load full fringe images
img_size =128
newsize=(img_size,img_size)
images =  [asarray(Image.open(file).resize(newsize)) for file in sorted(glob.glob("./FEM961/fringes_961_PU_Nocor/*.png"))]
images_array_label = asarray(images)

images_array_label= (images_array_label.reshape(-1, img_size, img_size, 1))/255.0

#check one img
#plt.imshow(images_array_label[500],cmap='gray')

# load missed fringe images
newsize=(img_size,img_size)
images =  [asarray(Image.open(file).resize(newsize)) for file in sorted(glob.glob("./FEM961/fringes_961_PU_cor_half_new/*.png"))]
images_array_input = asarray(images)


images_array_input= (images_array_input.reshape(-1, img_size, img_size, 1))/255.0
#check one img
#plt.imshow(images_array_input[500],cmap='gray')

#prepare the dataset
#dataset_norm = [images_array_input, images_array_label]


# In[124]:


X_train, X_test, y_train, y_test = train_test_split(images_array_input, images_array_label, test_size=0.1)


# In[125]:


#train dataset
dataset_a=tf.data.Dataset.from_tensor_slices(X_train) # miss fringe image
dataset_b=tf.data.Dataset.from_tensor_slices(y_train) # full fringe image

zip_dataset=tf.data.Dataset.zip((dataset_b,dataset_a))

BATCH_SIZE = 32
train_dataset = zip_dataset.shuffle(961).batch(BATCH_SIZE, drop_remainder=True).prefetch(1)



#test dataset
dataset_a=tf.data.Dataset.from_tensor_slices(X_test) # miss fringe image
dataset_b=tf.data.Dataset.from_tensor_slices(y_test) # full fringe image

zip_dataset=tf.data.Dataset.zip((dataset_b,dataset_a))

BATCH_SIZE = 32
test_dataset = zip_dataset.shuffle(961).batch(BATCH_SIZE, drop_remainder=True).prefetch(1)


# In[111]:


def build_generator():
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=3, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=3, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input]) #skip connection
            return u

        
        d0 = Input(shape=img_shape)

        # Downsampling
        d1 = conv2d(d0, gf, bn=False)
        d2 = conv2d(d1, gf*2)
        d3 = conv2d(d2, gf*4)
        d4 = conv2d(d3, gf*8)
        d5 = conv2d(d4, gf*8)
        d6 = conv2d(d5, gf*8)
        d7 = conv2d(d6, gf*8)

        # Upsampling
        u1 = deconv2d(d7, d6, gf*8)
        u2 = deconv2d(u1, d5, gf*8)
        u3 = deconv2d(u2, d4, gf*8)
        u4 = deconv2d(u3, d3, gf*4)
        u5 = deconv2d(u4, d2, gf*2)
        u6 = deconv2d(u5, d1, gf)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(channels, kernel_size=3, strides=1, padding='same', activation='tanh')(u7)

        return Model(d0, output_img)


# In[126]:


def build_discriminator():
        # a small function to make one layer of the discriminator
        def d_layer(layer_input, filters, f_size=3, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=img_shape)
        img_B = Input(shape=img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, df, bn=False)
        d2 = d_layer(d1, df*2)
        d3 = d_layer(d2, df*4)
        d4 = d_layer(d3, df*8)

        validity = Conv2D(1, kernel_size=3, strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity)


# In[132]:


# Input shape
img_rows = 128
img_cols = 128
channels = 1
img_shape = (img_rows, img_cols, channels)


# Calculate output shape of D (PatchGAN)
patch = int(img_rows / 2**4)
disc_patch = (patch, patch, 1)

# Number of filters in the first layer of G and D
gf = 32
df = 32

optimizer = Adam(0.0002, 0.5)

# Build and compile the discriminator
discriminator = build_discriminator()
discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

# Build the generator
generator = build_generator()

# Input images and their conditioning images
img_A = Input(shape=img_shape)
img_B = Input(shape=img_shape)

# By conditioning on B generate a fake version of A
fake_A = generator(img_B)

# For the combined model we will only train the generator
discriminator.trainable = False

# Discriminators determines validity of translated images / condition pairs
valid = discriminator([fake_A, img_B])

combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)


# In[141]:


def train(train_dataset,test_dataset, epochs, batch_size=1, show_interval=1):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + disc_patch)
        fake = np.zeros((batch_size,) + disc_patch)
        
        norm_train_epoch =[]
        norm_test_epoch =[]

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(train_dataset):

                #  Train Discriminator
                

                # Condition on B and generate a translated version
                fake_A = generator.predict(imgs_B)

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = discriminator.train_on_batch([imgs_A, imgs_B], valid)
                d_loss_fake = discriminator.train_on_batch([fake_A, imgs_B], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

               
                #  Train Generator
                g_loss = combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

                elapsed_time = datetime.datetime.now() - start_time
                
                # Plot the progress
                
                # If at show interval => show generated image samples
            
            
            #train performance
            norm_train_list = []
            for batch_i, (imgs_A_tr, imgs_B_tr) in enumerate(train_dataset):
                
                fake_train = generator.predict(imgs_B_tr)
                norm_train_list.append(np.square(tf.keras.losses.mean_squared_error(imgs_A_tr, fake_train)/np.mean(np.square(imgs_A_tr))))
                #norm_train_list.append (tf.norm(fake_train-imgs_A_tr, ord='euclidean')) #L2 norm
            
            #test performance
            norm_test_list = []
            for batch_i, (imgs_A_te, imgs_B_te) in enumerate(test_dataset):
                
                fake_test = generator.predict(imgs_B_te)
                norm_test_list.append(np.square(tf.keras.losses.mean_squared_error(imgs_A_te, fake_test)/np.mean(np.square(imgs_A_te))))
                #norm_test_list.append (tf.norm(fake_test-imgs_A_te, ord='euclidean')) #L2 norm
                           
            cur_norm_train = np.mean(np.array(norm_train_list))
            cur_norm_test = np.mean(np.array(norm_test_list))
            
            

            norm_train_epoch.append(cur_norm_train)
            norm_test_epoch.append(cur_norm_test)
            
            if epoch % show_interval == 0:
                plot_results(fake_A, epoch+1, 'Gen',8)
                plot_results(imgs_A, epoch+1, 'True', 8)
                plot_results(imgs_B, epoch+1, 'Miss', 8)
                    
            #epoch, Dloss, Gloss, train L2 norm, test L2 norm
            with open('epoch_L2norm.txt', 'a+') as f:
                f.write('%d, %f, %3d%%, %f, %f, %f\n' % (epoch+1,d_loss[0], 100*d_loss[1], g_loss[0], cur_norm_train,cur_norm_test ))                                                
            
            print ("[Epoch %d/%d]  [D loss: %f, acc: %3d%%] [G loss: %f] [train L2 error: %f] [test L2 error: %f] time: %s" % (epoch+1, epochs,
                                                                       
                                                                         d_loss[0], 100*d_loss[1],
                                                                         g_loss[0], cur_norm_train,cur_norm_test,
                                                                         elapsed_time))
        return norm_train_epoch, norm_test_epoch


# In[142]:


norm_train_epoch, norm_test_epoch = train(train_dataset,test_dataset, epochs=500, batch_size=32, show_interval=1)


generator.save('saved_model/my_cGANmodel_half')
