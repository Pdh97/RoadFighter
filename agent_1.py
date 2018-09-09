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
package = 'com.minigame.carracing'
activity = 'com.unity3d.player.UnityPlayerNativeActivity'
runComponent = package + '/' + activity
device.startActivity(component=runComponent)
ViewClient.sleep(3.0)
device.touch(365, 780, 'DOWN_AND_UP');
ViewClient.sleep(10.0)

#-----------------------Environment------------------------#
class Environment:
    max = 0
    iter = 0
    count = 0
    def touch(self, action):
        if action == 1:
            device.shell(cmdl)
        elif action == 2:
            device.shell(cmdr)
        else:
            x = random.randint(0, 1)
            if x == 1:
                device.shell(cmdl)
            else:
                device.shell(cmdr)
        device.takeSnapshot(reconnect = True).save('shot.png', 'PNG')
        image2 = Image.open('shot.png').convert('RGB')
        image2 = image2.crop((203, 80, 591, 1280)).resize((100, 300))
        R, G, B = image2.split()
        r = R.load()
        g = G.load()
        b = B.load()
        w, h = image2.size
        for i in range(w):
            for j in range(h):
                if((r[i, j] > 175 and g[i, j] > 175 and b[i, j] > 175)):
                    r[i, j] = 99
                    g[i, j] = 101
                    b[i, j] = 99
        image2 = Image.merge('RGB', (R, G, B))
        image2 = np.array(image2)
        return image2, getReward(), isOver()

    def run(self, agent):
        count = self.count
        totalReward = 0
        while True:
            count+=1
            device.takeSnapshot(reconnect=True).save('shot1.png', 'PNG')
            image1 = Image.open('shot1.png').convert('RGB')
            image1 = image1.crop((203, 80, 591, 1280)).resize((100, 300))
            R, G, B = image1.convert('RGB').split()
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
            action = agent.extrapolate(image1)
            state = image1
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
        print("Total reward:" + str(totalReward) + "Max till now: " + str(self.max) + "I: " + str(self.iter))
        self.count += count

#---------------------------Brain------------------------------#
class Brain:

    def __init__(self, actionCount):
        self.model = Sequential()
		#image dimensions can be changed from here
        self.model.add(Conv2D(12, (5, 5), strides=1, activation='relu', input_shape=(300, 100, 3), kernel_initializer='glorot_uniform', use_bias=True))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(8, (3, 3), strides=1, activation='relu', kernel_initializer='glorot_uniform', use_bias=True))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(units=100, activation="tanh", use_bias=True))
        self.model.add(Dense(units=50, activation="tanh", use_bias=True))
        self.model.add(Dense(units=actionCount, activation="relu"))
        self.model.compile(loss="mse", optimizer=SGD(lr=0.05), metrics=['accuracy'])
        if LOAD:
            self.model.load_weights('rf.h5')

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
maxEpsilon = 0.3
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
roadFighter = Environment()
LOAD = False
actionCount = 3
agent = Agent(actionCount)
try:
    while True:
        roadFighter.run(agent)
        device.touch(365, 780, 'DOWN_AND_UP');
        ViewClient.sleep(10.0)
finally:
    agent.brain.model.save("rf.h5")
    print("GAME OVER! after ", roadFighter.count, "touches!")
