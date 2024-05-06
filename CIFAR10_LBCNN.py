import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.stats import bernoulli
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

def new_weights_non_trainable(h, w, num_input, num_output, sparsity=0.5):
    # Number of elements
    num_elements = h * w * num_input * num_output
    
    # Create an array with n number of elements
    array = np.arange(num_elements)
    
    # Random shuffle it
    np.random.shuffle(array)
    
    # Fill with 0
    weight = np.zeros([num_elements])
    
    # Get number of elements in array that need be non-zero
    ind = int(sparsity * num_elements + 0.5)
    
    # Get it piece as indexes for weight matrix
    index = array[:ind]
  
    for i in index:
        # Fill those indexes with bernoulli distribution
        # Method rvs = random variates
        weight[i] = bernoulli.rvs(0.5)*2-1

    # Reshape weights array for matrix that we need
    weights = weight.reshape(h, w, num_input, num_output)
    
    return weights

# Define SL1 sublayer
class SL1Layer(layers.Layer):
    def __init__(self, filters, kernel_size, input_shape, sparsity=0.5):
        super(SL1Layer, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.input_shape_ = input_shape
        self.sparsity = sparsity
        self.kernel = self.add_weight(shape=(kernel_size, kernel_size, input_shape[-1], filters),
                                      initializer=self.ternary_weight_init,
                                      trainable=False)

    def ternary_weight_init(self, shape, dtype=None):
        return tf.Variable(new_weights_non_trainable(h=shape[0],
                                                     w=shape[1],
                                                     num_input=shape[2],
                                                     num_output=shape[3],
                                                     sparsity=self.sparsity).astype(np.float32),
                                                     trainable=False)

    def call(self, inputs):
        output = tf.nn.conv2d(inputs, self.kernel, strides=[1, 1, 1, 1], padding='SAME')
        output = tf.nn.relu(output)
        return output

# Define SL2 sublayer
class SL2Layer(layers.Layer):
    def __init__(self, filters):
        super(SL2Layer, self).__init__()
        self.filters = filters

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(1, 1, input_shape[-1], self.filters),
                                      initializer='random_normal',
                                      trainable=True)

    def call(self, inputs):
        output = tf.nn.conv2d(inputs, self.kernel, strides=[1, 1, 1, 1], padding='SAME')
        return output

# Load CIFAR-10 dataset
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
#x_train, x_test = x_train / 255.0, x_test / 255.0

(X_train, y_train), (X_val, y_val) = cifar10.load_data()

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

# Build the LBCNN model
def LBCNN():
    model = models.Sequential()
    model.add(SL1Layer(filters=8, kernel_size=3, input_shape=(32, 32, 3)))
    model.add(SL2Layer(filters=8))

    model.add(SL1Layer(filters=32, kernel_size=3, input_shape=(32, 32, 8)))
    model.add(SL2Layer(filters=32))

    model.add(SL1Layer(filters=64, kernel_size=3, input_shape=(32, 32, 32)))
    model.add(SL2Layer(filters=64))

    model.add(SL1Layer(filters=128, kernel_size=3, input_shape=(32, 32, 64)))
    model.add(SL2Layer(filters=128))

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

# Compile and train the model
model = LBCNN()
model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
#model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    channel_shift_range=50,
    horizontal_flip=True)
validationgen = ImageDataGenerator(
    rescale=1./255)

# フィット
datagen.fit(X_train)
validationgen.fit(X_val)

history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=128),
                    steps_per_epoch=len(X_train) / 128, validation_data=validationgen.flow(X_val, y_val), epochs=100).history