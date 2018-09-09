from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import *
import matplotlib.pyplot as plt
from PIL import Image

#------------------Model Architecture-----------------#
model = Sequential()
model.add(Conv2D(12, (5, 5), strides=1, activation='relu', input_shape=(300, 100, 3), kernel_initializer='glorot_uniform', use_bias=True))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(8, (3, 3), strides=1, activation='relu', kernel_initializer='glorot_uniform', use_bias=True))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=100, activation="relu", use_bias=True))
model.add(Dense(units=50, activation="sigmoid", use_bias=True))
model.add(Dense(units=3, activation="softmax"))
model.compile(loss="mse", optimizer=SGD(lr=0.03), metrics=['accuracy'])
model.load_weights('rf.h5')

layer_index = 0

#-----------------Image Pre-processing----------------#
image1 = Image.open('shot1.png').convert('RGB')
image1 = image1.crop((203, 80, 591, 1280)).resize((100, 300))
R, G, B = image1.split()
r = R.load()
g = G.load()
b = B.load()
w, h = image1.size
for i in range(w):
    for j in range(h):
        if((r[i, j] > 175 and g[i, j] > 175 and b[i, j] > 175)):
            r[i, j] = 99
            g[i, j] = 101
            b[i, j] = 99
image1 = Image.merge('RGB', (R, G, B))
image1 = np.array(image1)
plt.imshow(image1)
plt.show()

newModel = Model(model.inputs, model.layers[layer_index].output)
output = newModel.predict(np.expand_dims(image1, axis=0))[0, :, :, :]

# shows convolution feature maps
for i in range(output.shape[2]):
    plt.imshow(output[:, :, i])
    plt.show()