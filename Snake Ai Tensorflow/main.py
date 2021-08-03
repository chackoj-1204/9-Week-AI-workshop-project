import pygame
import random
import numpy as np

Pygame_mul = 30  # Pygame Multiplier (used to draw np.array as blocks in pygame )
WIN_WIDTH = 900
WIN_HEIGHT = 900
pygame.init()
STAT_FONT = pygame.font.SysFont("comicsans",50)
import matplotlib.pyplot as plt
import time
import os
from collections import deque
import matplotlib.pyplot as plt
from IPython import display
import keras
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)

def mse(pred,true):
    return (true - pred)**2

class Field:
    def __init__(self, height=30, width=30):
        self.reset()

    class Snek:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.vel = 1
            self.direction = 0
            self.actions = [[1, 0], [0, -1], [-1, 0], [0, 1]]
            self.fullSnek = [[self.x, self.y], [self.x - 1, self.y - 1], [self.x - 2, self.y - 2]]
            # 0 is no action 1 is up -2 is left -1 is down and 2 is right

        def move(self, action=0):
            if action[0] == 1:
                x_trans, y_trans = self.actions[self.direction]
                self.x += x_trans
                self.y += y_trans
            elif action[1] == 1:
                self.direction = (self.direction + 1) % 4
                x_trans, y_trans = self.actions[self.direction]
                self.x += x_trans
                self.y += y_trans
            else:
                self.direction = (self.direction - 1) % 4
                x_trans, y_trans = self.actions[self.direction]
                self.x += x_trans
                self.y += y_trans
            self.updateSnek()

        def updateSnek(self):
            temp = [self.x, self.y]
            for x, part in enumerate(self.fullSnek):
                self.fullSnek[x] = temp
                temp = part

    class Fruit:
        def __init__(self):
            self.x = random.randint(1, 28)
            self.y = random.randint(1, 28)

        def new(self, fullsnek):
            self.x = random.randint(1, 28)
            self.y = random.randint(1, 28)
            while [self.x, self.y] in fullsnek:
                self.x = random.randint(1, 28)
                self.y = random.randint(1, 28)
    def update_field(self):
        try:
            self.body = np.zeros(shape=(self.height, self.width))
            self.body[self.Snake.x][self.Snake.y]+=2
            for part in self.Snake.fullSnek[1:]:
                self.body[int(part[0])][int(part[1])] += 1
            self.body[self.Dot.x][self.Dot.y] = 4
        except:
            pass


    def run(self, action):
        # print(action)
        self.Snake.move(action)
        self.num_moves += 1
        reward = 0#1800**.5 - ((self.Snake.x - self.Dot.x) ** 2 + (self.Snake.y - self.Dot.y) ** 2) ** .5
        if self.Snake.x == self.Dot.x and self.Snake.y == self.Dot.y:
            reward += 10
            self.Dot.new(self.Snake.fullSnek)
            checkX = self.Snake.fullSnek[-1][0] - self.Snake.fullSnek[-2][0]
            checkY = self.Snake.fullSnek[-1][1] - self.Snake.fullSnek[-2][1]
            if checkX != 0:
                self.Snake.fullSnek.append([self.Snake.fullSnek[-1][0] + 1 * checkX, self.Snake.fullSnek[-1][1]])
            else:
                self.Snake.fullSnek.append([self.Snake.fullSnek[-1][0], self.Snake.fullSnek[-1][1] + 1 * checkY])
        self.update_field()
        if ((30 <= self.Snake.x or 0 > self.Snake.x) or (30 <= self.Snake.y or 0 > self.Snake.y)) or 3 in self.body or self.num_moves > 100 * len(self.Snake.fullSnek):
            return -10, True, len(self.Snake.fullSnek) - 3
        else:
            return reward, False, len(self.Snake.fullSnek) - 3

    def reset(self,height=30, width=30):
        self.width = width
        self.height = height
        self.body = np.zeros(shape=(self.height, self.width))
        self.Dot = self.Fruit()
        self.Snake = self.Snek(15, 15)
        self.update_field()
        self.num_moves = 0

    def iscollision(self, x, y):
        if ((30 <= x or 0 > x) or (30 <= y or 0 > y)) or [x, y] in self.Snake.fullSnek[1:]:
            return True
        else:
            return False





class Agent():
    def __init__(self):
        self.maxmem = 100000
        self.batch_size = 1000
        self.lr = 0.001
        self.num_games = 0
        self.epsilsion = 0
        self.gamma = .95
        self.mem = deque(maxlen=self.maxmem)
        self.model = self.build_model()
        self.opt = Adam(lr=self.lr)

    def get_state(self, game):
        left = game.Snake.actions[game.Snake.direction] == [-1, 0]
        right = game.Snake.actions[game.Snake.direction] == [1, 0]
        up = game.Snake.actions[game.Snake.direction] == [0, 1]
        down = game.Snake.actions[game.Snake.direction] == [-1, 0]

        return np.array([right and game.iscollision(game.Snake.x + 1, game.Snake.y) or
                         left and game.iscollision(game.Snake.x - 1, game.Snake.y) or
                         up and game.iscollision(game.Snake.x, game.Snake.y + 1) or
                         down and game.iscollision(game.Snake.x, game.Snake.y - 1),

                         right and game.iscollision(game.Snake.x, game.Snake.y - 1) or
                         left and game.iscollision(game.Snake.x, game.Snake.y + 1) or
                         up and game.iscollision(game.Snake.x + 1, game.Snake.y) or
                         down and game.iscollision(game.Snake.x - 1, game.Snake.y),

                         right and game.iscollision(game.Snake.x, game.Snake.y + 1) or
                         left and game.iscollision(game.Snake.x, game.Snake.y - 1) or
                         up and game.iscollision(game.Snake.x - 1, game.Snake.y) or
                         down and game.iscollision(game.Snake.x + 1, game.Snake.y),
                         left,
                         right,
                         up,
                         down,
                         game.Dot.x <= game.Snake.x,
                         game.Dot.x >= game.Snake.x,
                         game.Dot.y <= game.Snake.y,
                         game.Dot.y >= game.Snake.y
                         ], dtype=int).reshape(11,1)

    def remember(self, state, action, reward, next_state, done):
        self.mem.append([state, action, reward, next_state, done])

    def train_long_mem(self):
        if len(self.mem) > self.batch_size:
            mini_sample = random.sample(self.mem, self.batch_size)
        else:
            mini_sample = self.mem
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.train_step(states, actions, rewards, next_states, dones)

    def train_short_mem(self, state, action, reward, next_state, done):
        self.train_step(state, action, reward, next_state, done)

    def train_step(self, state, action, reward, next_state, done):
        with tf.GradientTape(persistent=True) as tape:
            state = np.array(state, dtype="float64")
            action = np.array(action, dtype=np.long)
            reward = np.array(reward, dtype="float64")
            next_state = np.array(next_state)

            state = state.reshape((-1, 11))
            next_state = next_state.reshape((-1, 11))

            if state.shape[0] == 1:
                action = action.reshape((-1,3))
                reward = reward.reshape((-1,1))
                done = (done,)
            pred = self.model(state)
            targets = np.copy(pred)
            for ind in range(len(action)):
                Q_new = reward[ind]
                if not done[ind]:
                    Q_new = Q_new + self.gamma * np.max(self.model(next_state[ind].reshape((1,11)))) #* (1 - done[ind])
                targets[ind][np.argmax(action[ind],axis=-1)] = Q_new

            loss = mse(pred,targets)
            grad = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(grad, self.model.trainable_variables))

    def build_model(self):
        model = Sequential([Dense(256, activation="relu",input_shape=(11,)),Dense(3, activation="softmax")])
        return model

    def get_action(self, past):
        self.epsilsion = 80 - self.num_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilsion:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            pred = self.model(past.reshape(1,11))[0]
            final_move[np.argmax(pred)] = 1
        return final_move


def main():
    scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = Field()
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()
    while True:
        pygame.event.get()
        BLACK = (25, 25, 25)
        WHITE = (255, 255, 255)
        RED = (255, 80, 80)
        BLUE = (80, 80, 255)
        val2color = {0: BLACK, 1: WHITE,
                     2: BLUE, 4: RED}
        for r in range(game.body.shape[0]):
            for c in range(game.body.shape[1]):
                pygame.draw.rect(win,
                                 val2color[game.body[r][c]],
                                 (c * Pygame_mul, r * Pygame_mul, 30, 30))
        text = STAT_FONT.render("Snake Len: " + str(len(game.Snake.fullSnek) - 3), 1, (255, 255, 255))
        win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))
        pygame.display.update()
        clock.tick(60)
        state_old = agent.get_state(game)

        action = agent.get_action(state_old)
        reward, done, score = game.run(action)

        state_new = agent.get_state(game)
        agent.remember(state_old, action, reward, state_new, done)
        agent.train_short_mem(state_old,action,reward,state_new,done)
        if done:
            game.reset()
            agent.num_games += 1
            agent.train_long_mem()
            if score > record:
                record = score
                agent.model.save("model.h5")
            print("Game ", agent.num_games, " Over:::Score:", score)

            scores.append(score)
            total_score += score
            mean_score = total_score / agent.num_games
            plot_mean_scores.append(mean_score)
            plot(scores, plot_mean_scores)


if __name__ == '__main__':
    main()