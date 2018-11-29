import time
from PIL import Image
import numpy as np
import random
from ocrCall import *
from com.dtmilano.android.viewclient import ViewClient
from keras.models import Sequential
from keras.layers import MaxPooling2D, Conv2D, Dense, Flatten
from keras.optimizers import SGD
import math

#---------------------connecting the phone---------------------#
device, serialno = ViewClient.connectToDeviceOrExit(verbose=1)
cmdl = 'input touchscreen swipe 260 700 260 700 400;'
cmdr = 'input touchscreen swipe 500 700 500 700 400;'
back = 'input keyevent KEYCODE_BACK'
package = 'com.minigame.carracing'
activity = 'com.unity3d.player.UnityPlayerNativeActivity'
runComponent = package + '/' + activity
device.startActivity(component=runComponent)
ViewClient.sleep(3.0)
device.touch(365, 780, 'DOWN_AND_UP');
ViewClient.sleep(10.0)
LOAD = False

#-----------------------Environment------------------------#
class Environment:
    max = 0
    iter = 0
    count = 0

    def __init__(self):
        self.image2 = device.takeSnapshot(reconnect=True)
        device.shell(back)
        self.image2 = self.image2.convert('RGB').crop((203, 80, 591, 1280)).resize((100, 300))
        R, G, B = self.image2.split()
        r = R.load()
        g = G.load()
        b = B.load()
        w, h = self.image2.size
        for i in range(w):
            for j in range(h):
                if((r[i, j] > 175 and g[i, j] > 175 and b[i, j] > 175)):
                    r[i, j] = 99
                    g[i, j] = 101
                    b[i, j] = 99
        self.image2 = Image.merge('RGB', (R, G, B))
        self.image2 = np.array(self.image2).astype('float32')

    def touch(self, action):
        device.touch(350, 650, 'DOWN_AND_UP');
        if action == 0:
            device.shell(cmdl)
        elif action == 1:
            device.shell(cmdr)
        image = device.takeSnapshot(reconnect = True)
        device.shell(back)
        # image2 = Image.open('shot.png').convert('RGB')
        self.image2 = image.crop((203, 80, 591, 1280)).resize((100, 300)).convert('RGB')
        R, G, B = self.image2.split()
        r = R.load()
        g = G.load()
        b = B.load()
        w, h = self.image2.size
        for i in range(w):
            for j in range(h):
                if((r[i, j] > 175 and g[i, j] > 175 and b[i, j] > 175)):
                    r[i, j] = 99
                    g[i, j] = 101
                    b[i, j] = 99
        self.image2 = Image.merge('RGB', (R, G, B))
        self.image2 = np.array(self.image2).astype('float32')
        return self.image2, getReward(image), isOver(image)

    def run(self, agent):
        totalReward = 0
        while True:
            self.count+=1
            image1 = self.image2
            action = agent.extrapolate(image1)
            print('epsilon' + str(agent.epsilon))
            state = image1
            device.touch(350, 650, 'DOWN_AND_UP');
            nextState, reward, done = self.touch(action)
            if done:
                nextState = None
            agent.observe((state, action, reward, nextState))
            agent.replay()
            state = nextState
            totalReward = reward
            if done:
                break
        self.iter+=1
        if self.max < totalReward:
            self.max = totalReward
        print("Total reward: " + str(totalReward) + " | Max till now: " + str(self.max) + " | I: " + str(self.iter))

#---------------------------Brain------------------------------#
class Brain:

    def __init__(self, actionCount):
        self.model = Sequential()
        self.model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(300, 100, 3), output_shape=(300, 100, 3)))

        self.model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(2), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))

        self.model.add(Conv2D(64, (3, 3),strides=(1), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))

        self.model.add(Conv2D(64, (3, 3), strides=(1), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))

        self.model.add(Flatten())
        self.model.add(Dense(92, activation='relu'))
        self.model.add(Dense(actionCount, activation='relu'))
        self.model.compile(loss="mse", optimizer=SGD(lr=0.05), metrics=['accuracy'])
        if LOAD:
            self.model.load_weights('rf_new_1537535039.01.h5')

    def train(self, x, y):
        self.model.fit(x, y, batch_size = 1, epochs = 1, verbose = 1)

    def predict(self, data):
        return self.model.predict(data)

    def predictOne(self, image):
        return self.model.predict(image.reshape(1, 300, 100, 3)).flatten()

#--------------------------Memory-------------------------------#
class Memory:   # stored as ( s, a, r, s_ )
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        # print("add...")
        self.samples.append(sample)
        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        # print("sample...")
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

#-------------------------------Agent-------------------------------#
memoryCapacity = 1000
batchSize = 4
gamma = 0.99
maxEpsilon = 0.9
minEpsilon = 0.01
lambda_ = 0.01

class Agent:
    steps = 0
    epsilon = maxEpsilon

    def __init__(self, actionCount):
        self.actionCount = actionCount
        self.brain = Brain(actionCount)
        self.memory = Memory(memoryCapacity)

    def extrapolate(self, state):
        self.steps += 1
        self.epsilon = minEpsilon + (maxEpsilon - minEpsilon) * math.exp(-lambda_ * self.steps)
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCount-1)
        else:
            return np.argmax(self.brain.predictOne(state))

    def observe(self, sample):
        self.memory.add(sample)

    def replay(self):
        batch = self.memory.sample(batchSize)
        batchLen = len(batch)
        no_state = np.zeros((300, 100, 3))
        states = np.array([ o[0] for o in batch ])
        states_ = np.array([ (no_state if o[3] is None else o[3]) for o in batch ])
        p = self.brain.predict(states)
        p_ = self.brain.predict(states_)
        x = np.zeros((batchLen, 300, 100, 3)) #image dimensions
        y = np.zeros((batchLen, self.actionCount))

        for i in range(batchLen):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + gamma * np.amax(p_[i])

            x[i] = s
            y[i] = t

        self.brain.train(x, y)

#------------------------------Main--------------------------------#
if __name__ == '__main__':
    roadFighter = Environment()
    actionCount = 2
    agent = Agent(actionCount)
    MAX_EP = 128
    try:
        t = 0
        while t < MAX_EP:
            t += 1
            roadFighter.run(agent)
            device.touch(365, 780, 'DOWN_AND_UP');
            ViewClient.sleep(10.0)
    finally:
        agent.brain.model.save('rf_new_' + str(time.time()) + '.h5')
        print("GAME OVER! after ", roadFighter.count, "touches!")