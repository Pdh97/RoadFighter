from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from keras.models import Sequential
from keras.constraints import max_norm
from keras.optimizers import *
from keras.layers import *
import numpy as np

def getReward():
    ans = ""
    im = Image.open("shot.png")
    w, h = im.size
    im.crop((553, 0, w-35, h-1233)).save("score.png", "PNG")
    # left upper right lower
    im = Image.open("score.png")
    w, h = im.size
    # (im.crop((0, 0, w-100, h)).save("L1t0.png", "PNG"))
    # (im.crop((20, 0, w-80, h)).save("L2t0.png", "PNG"))
    # (im.crop((40, 0, w-60, h)).save("L3t1.png", "PNG"))
    # (im.crop((60, 0, w-40, h)).save("L4t9.png", "PNG"))
    # (im.crop((80, 0, w-20, h)).save("L5t0.png", "PNG"))
    # (im.crop((100, 0, w, h)).save("L6t0.png", "PNG"))
    ImageEnhance.Contrast(im.crop((0, 0, w-100, h)).filter(ImageFilter.MedianFilter())).enhance(2).save("1t.png", "PNG")
    ImageEnhance.Contrast(im.crop((20, 0, w-80, h)).filter(ImageFilter.MedianFilter())).enhance(2).save("2t.png", "PNG")
    ImageEnhance.Contrast(im.crop((40, 0, w-60, h)).filter(ImageFilter.MedianFilter())).enhance(2).save("3t.png", "PNG")
    ImageEnhance.Contrast(im.crop((60, 0, w-40, h)).filter(ImageFilter.MedianFilter())).enhance(2).save("4t.png", "PNG")
    ImageEnhance.Contrast(im.crop((80, 0, w-20, h)).filter(ImageFilter.MedianFilter())).enhance(2).save("5t.png", "PNG")
    ImageEnhance.Contrast(im.crop((100, 0, w, h)).filter(ImageFilter.MedianFilter())).enhance(2).save("6t.png", "PNG")
    model = Sequential()
    model.add(Conv2D(20, (3, 3), strides=1, activation='relu', use_bias=True, kernel_initializer=initializers.RandomNormal(mean = 0, stddev = 0.03, seed=None), bias_initializer='zeros', input_shape=(48,28,1)))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    model.add(Dense(units=350, activation='relu', use_bias=True))
    model.add(Dense(units=130, activation='sigmoid'))
    model.add(Dense(units=10, activation='sigmoid'))
    opt = SGD(lr=0.03)
    model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
    model.load_weights("model.h5")

    for i in range(1, 7):
        im = Image.open("%dt.png" %i)
        img = np.array(im)
        imo = ImageOps.fit(im, (28, 48), Image.NEAREST, 0, (0,0))
        imo = ImageEnhance.Contrast(imo.filter(ImageFilter.MedianFilter())).enhance(2).convert("1")
        imo.load()
        data = np.asarray(imo, dtype=np.uint8)
        data = data.reshape(1, 48, 28, 1)
        #print(model.predict(data, batch_size=1, verbose=1))
        ans += str(np.argmax(model.predict(data, batch_size=1, verbose=1)))

    print(ans)
    return int(ans)

def isOver():
    im = Image.open("shot.png")
    im = im.resize((1080, 1920), Image.ANTIALIAS)
    w, h = im.size
    print w, h
    im.crop((195, 420, w-180, h-1350)).save("go.png", "PNG")
    im = Image.open("go.png")
    img = np.array(im)
    n = np.sum(img == 255)
    # text = pytesseract.image_to_string(im)
    # print(text)
    if n == 167748:
        return True
    else:
        return False
