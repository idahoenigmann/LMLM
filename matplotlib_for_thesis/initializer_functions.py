import tensorflow as tf
import random
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
from collections import defaultdict

if __name__ == '__main__':
    x_axis = []
    zeros_l = []
    ones_l = []
    constant_l = []
    randomnormal_l = []
    randomuniform_l = []
    truncatednormal_l = []

    for i in range(-220, 220):
        x_axis.append(i / 200)
        if i == 0:
            zeros_l.append(400)
        else:
            zeros_l.append(0)

        if i == 1 * 200:
            ones_l.append(400)
        else:
            ones_l.append(0)

        if i == -0.5 * 200:
            constant_l.append(400)
        else:
            constant_l.append(0)

        if -0.75 * 200 < i < 0.75 * 200:
            randomuniform_l.append(400 / 200)
        else:
            randomuniform_l.append(0)

        mu, sigma = 0, 0.05
        tmp = 1 / math.sqrt(2 * math.pi * 0.05 ** 2) * math.e ** (- (i ** 2) / 2 * 0.05 ** 2)
        randomnormal_l.append(tmp)

        if -2 * sigma * 200 < i < 2 * sigma * 200:
            truncatednormal_l.append(tmp)
        else:
            truncatednormal_l.append(0)

    plt.title("Zeros")
    plt.plot(x_axis, zeros_l, label='zeros')
    plt.savefig('methodology_neuralNetwork_initializerFunctions_zeros.png')
    plt.show()

    plt.title("Ones")
    plt.plot(x_axis, ones_l, label='ones')
    plt.savefig('methodology_neuralNetwork_initializerFunctions_ones.png')
    plt.show()

    plt.title("Constant")
    plt.plot(x_axis, constant_l, label='constant 0.5')
    plt.savefig('methodology_neuralNetwork_initializerFunctions_constant.png')
    plt.show()

    plt.title("Random Normal")
    plt.plot(x_axis, randomnormal_l, label='random normal')
    plt.savefig('methodology_neuralNetwork_initializerFunctions_randomNormal.png')
    plt.show()

    plt.title("Random Uniform")
    plt.plot(x_axis, randomuniform_l, label='random uniform -1, 1')
    plt.savefig('methodology_neuralNetwork_initializerFunctions_randomUniform.png')
    plt.show()

    plt.title("Truncated Normal")
    plt.plot(x_axis, truncatednormal_l, label='truncated normal')
    plt.savefig('methodology_neuralNetwork_initializerFunctions_truncatedNormal.png')
    plt.show()