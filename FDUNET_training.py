#import library
import tensorflow as tf
import numpy as np
from skimage.io import imread, imshow
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os
from os import makedirs
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.mixed_precision import experimental as mixed_precision

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

tf.config.optimizer.set_jit(True)

########################################################################################################################
'''initialize constants'''
########################################################################################################################
seed = 7
np.random.seed = seed
tf.random.set_seed(seed)


IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 1
########################################################################################################################
'''Load dataset'''
########################################################################################################################
TRAIN_PATH = 'Simulated_dataset/'
data_ids = [filename for filename in os.listdir(TRAIN_PATH) if filename.startswith("x_")]

NUMBER_OF_SAMPLES = int(len(data_ids))
print(NUMBER_OF_SAMPLES)


########################################################################################################################
'''Folder for saving the model'''
########################################################################################################################
MODEL_NAME = 'modelFDUNET.h5'

X_total = np.zeros((NUMBER_OF_SAMPLES, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_total = np.zeros((NUMBER_OF_SAMPLES, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
NUMBER_EPOCHS = 250
PATIENCE = 10
MONITOR = 'val_loss'

im = TRAIN_PATH.split("d", 1)[1]
FOLDER_NAME = "modelARPAM"
makedirs(FOLDER_NAME)
MODEL_NAME = FOLDER_NAME + MODEL_NAME
LOG_NAME = FOLDER_NAME + "logs"

########################################################################################################################
'''Image augmentation'''
########################################################################################################################
print('Resizing training images and masks')
for data, val in enumerate(data_ids):
    ext = val.split("_", 1)[1] #To get the number after x_
    xpath = TRAIN_PATH + val
    ypath = TRAIN_PATH + 'y_' + ext

    img = imread(xpath)#[:, :, :IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    img = np.expand_dims(img, axis=2)
    X_total[data] = img  # Fill empty X_train with values from img

    true_img = imread(ypath)#[:, :, :IMG_CHANNELS]
    true_img = resize(true_img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    true_img = np.expand_dims(true_img, axis=2)
    Y_total[data] = true_img


########################################################################################################################
'''Divide in training and test data'''
########################################################################################################################
test_split = 0.1
X_train, X_test, Y_train, Y_test = train_test_split(X_total, Y_total, test_size=test_split, random_state=seed)

Y_pred = np.zeros((len(X_test), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)

X_train, Y_train = shuffle(X_train, Y_train, random_state=seed)

print('Done splitting and shuffling')

########################################################################################################################
'''Network functions'''
########################################################################################################################
def Conv2D_BatchNorm(input, filters, kernel_size, strides, activation, kernel_initializer, padding):
    out = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides= strides, activation=activation, kernel_initializer=kernel_initializer, padding=padding)(input)
    out = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                            beta_initializer='zeros', gamma_initializer='ones',
                                            moving_mean_initializer='zeros',
                                            moving_variance_initializer='ones', beta_regularizer=None,
                                            gamma_regularizer=None,
                                            beta_constraint=None, gamma_constraint=None)(out)
    return out


def Conv2D_Transpose_BatchNorm(input, filters, kernel_size, strides, activation, kernel_initializer, padding):
    out = tf.keras.layers.Conv2DTranspose(filters, kernel_size=kernel_size, strides= strides, activation=activation, kernel_initializer=kernel_initializer, padding=padding)(input)
    out = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                            beta_initializer='zeros', gamma_initializer='ones',
                                            moving_mean_initializer='zeros',
                                            moving_variance_initializer='ones', beta_regularizer=None,
                                            gamma_regularizer=None,
                                            beta_constraint=None, gamma_constraint=None)(out)
    return out

def DownBlock(input, filters, kernel_size, padding, activation, kernel_initializer):
    out = FD_Block(input, f_in=filters // 2, f_out=filters, k=filters // 8, kernel_size=3, padding='same',
                   activation=activation, kernel_initializer='glorot_normal')
    shortcut = out
    out = DownSample(out, filters, kernel_size, strides=2, padding=padding,
                     activation=activation, kernel_initializer=kernel_initializer)
    return [out, shortcut]


def BrigdeBlock(input, filters, kernel_size, padding, activation, kernel_initializer):
    out = FD_Block(input, f_in=filters // 2, f_out=filters, k=filters // 8, kernel_size=3, padding='same',
                   activation=activation, kernel_initializer='glorot_normal')
    out = UpSample(out, filters, kernel_size, strides=2, padding=padding,
                   activation=activation, kernel_initializer=kernel_initializer)
    return out


def UpBlock(input, filters, kernel_size, padding, activation, kernel_initializer):
    out = Conv2D_BatchNorm(input, filters= filters//2, kernel_size=1, strides=1, activation=activation,
                           kernel_initializer=kernel_initializer, padding=padding)
    out = FD_Block(out, f_in=filters // 2, f_out=filters, k=filters // 8, kernel_size=3, padding='same',
                   activation=activation, kernel_initializer='glorot_normal')
    out = UpSample(out, filters, kernel_size, strides=2, padding=padding,
                     activation=activation, kernel_initializer=kernel_initializer)
    return out


def FD_Block(input, f_in, f_out, k, kernel_size, padding, activation, kernel_initializer):
    out = input
    for i in range(f_in, f_out, k):
        shortcut = out
        out = Conv2D_BatchNorm(out, filters=f_in, kernel_size=1, strides=1, padding=padding,
                               activation=activation, kernel_initializer=kernel_initializer)
        out = Conv2D_BatchNorm(out, filters=k, kernel_size=kernel_size, strides=1, padding=padding,
                               activation=activation, kernel_initializer=kernel_initializer)
        out = tf.keras.layers.Dropout(0.7, seed=seed)(out)
        out = tf.keras.layers.concatenate([out, shortcut])
    return out


def DownSample(input, filters, kernel_size, strides, padding, activation, kernel_initializer):
    out = Conv2D_BatchNorm(input, filters, kernel_size=1, strides=1, activation= activation, kernel_initializer= kernel_initializer, padding=padding)
    out = Conv2D_BatchNorm(out, filters, kernel_size=kernel_size, strides=strides, activation=activation,
                           kernel_initializer=kernel_initializer, padding=padding)
    return out

def UpSample(input, filters, kernel_size, strides, padding, activation, kernel_initializer):
    out = Conv2D_BatchNorm(input, filters, kernel_size=1, strides=1, padding=padding,
                           activation=activation, kernel_initializer=kernel_initializer)
    out = Conv2D_Transpose_BatchNorm(out, filters//2, kernel_size=kernel_size, strides=strides, activation=activation,
                           kernel_initializer=kernel_initializer, padding=padding)
    return out


########################################################################################################################
'''Define parameters'''
########################################################################################################################
kernel_initializer = tf.keras.initializers.glorot_normal(seed=seed)
activation = 'relu'
filters = 16
padding = 'same'
kernel_size = 3
strides = 1

inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = inputs

out = Conv2D_BatchNorm(s, filters, kernel_size=kernel_size, strides= strides, activation=activation, kernel_initializer=kernel_initializer, padding=padding)

[out, c1] = DownBlock(out, filters*2**1, kernel_size, padding, activation, kernel_initializer)
[out, c2] = DownBlock(out, filters*2**2, kernel_size, padding, activation, kernel_initializer)
[out, c3] = DownBlock(out, filters*2**3, kernel_size, padding, activation, kernel_initializer)
[out, c4] = DownBlock(out, filters*2**4, kernel_size, padding, activation, kernel_initializer)
[out, c5] = DownBlock(out, filters*2**5, kernel_size, padding, activation, kernel_initializer)

out = BrigdeBlock(out, filters*2**6, kernel_size, padding, activation, kernel_initializer)

out = tf.keras.layers.concatenate([out, c5])
out = UpBlock(out, filters*2**5, kernel_size, padding, activation, kernel_initializer)


out = tf.keras.layers.concatenate([out, c4])
out = UpBlock(out, filters*2**4, kernel_size, padding, activation, kernel_initializer)
out = tf.keras.layers.concatenate([out, c3])
out = UpBlock(out, filters*2**3, kernel_size, padding, activation, kernel_initializer)
out = tf.keras.layers.concatenate([out, c2])
out = UpBlock(out, filters*2**2, kernel_size, padding, activation, kernel_initializer)
out = tf.keras.layers.concatenate([out, c1])

out = Conv2D_BatchNorm(out, filters, kernel_size=1, strides=1, activation=activation, kernel_initializer=kernel_initializer, padding=padding)
out = FD_Block(out, f_in=filters, f_out=filters*2, k=filters // 4, kernel_size=3, padding=padding,
                   activation=activation, kernel_initializer=kernel_initializer)

out = tf.keras.layers.Conv2D(filters=1, kernel_size=1, strides=1, padding=padding, activation='linear', kernel_initializer=kernel_initializer)(out)
out = tf.keras.layers.Add()([out, s])
out = tf.keras.layers.ReLU()(out)
outputs = out
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

########################################################################################################################
'''define adam'''
########################################################################################################################
opt = tf.keras.optimizers.Adam(learning_rate = 0.001)
model.compile(optimizer=opt, loss='mean_squared_error', metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()])

########################################################################################################################
'''Model checkpoints'''
########################################################################################################################
callbacks = [tf.keras.callbacks.ModelCheckpoint(MODEL_NAME, verbose=1, save_best_only=True),
             tf.keras.callbacks.TensorBoard(log_dir=LOG_NAME), 
		tf.keras.callbacks.EarlyStopping(patience=PATIENCE, monitor=MONITOR)]

########################################################################################################################
'''Compile model'''
########################################################################################################################
results = model.fit(X_train, Y_train, validation_split=0.2, batch_size=8, epochs=NUMBER_EPOCHS, callbacks=callbacks)
print('Model Trained')

########################################################################################################################
'''Model evaluvation'''
########################################################################################################################
model.evaluate(X_test, Y_test, verbose=1)
print('Done evaluation')
preds_test = np.zeros((142, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
preds_train = model.predict(X_train[:int(X_train.shape[0] * 0.8)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0] * 0.8):], verbose=1)
preds_test = model.predict(X_test, verbose=1)
print('Done prediction on simulation')
preds_test_int = preds_test.astype(np.uint8)
preds_test_t = tf.convert_to_tensor(preds_test_int)
Y_test_t = tf.convert_to_tensor(Y_test)
X_test_t = tf.convert_to_tensor(X_test)
ssim_test = tf.image.ssim(Y_test_t, preds_test_t, max_val=255)
ssim_test_orig = tf.image.ssim(Y_test_t, X_test_t, max_val=255)
psnr_test = tf.image.psnr(Y_test_t, preds_test_t, max_val=255)
psnr_test_orig = tf.image.psnr(Y_test_t, X_test_t, max_val=255)

print('SSIM Test')
print(np.mean(ssim_test.numpy()))

print('PSNR Test')
print(np.mean(psnr_test.numpy()))

print('SSIM original')
print(np.mean(ssim_test_orig.numpy()))

print('PSNR original')
print(np.mean(psnr_test_orig.numpy()))

