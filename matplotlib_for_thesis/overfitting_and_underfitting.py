import tensorflow as tf
import random
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
from collections import defaultdict
import scipy


def f(x):
    return -0.0279 * x ** 5 + 0.517 * x ** 4 - 3.4812 * x ** 3 + 10.147 * x ** 2 - 9.7439 * x + 6.8562


def l(x):
    return 1.90409 * x + 3.70918


if __name__ == '__main__':
    x_axis = []
    p_y_axis = []
    l_y_axis = []

    x_coord_l = [1.680854, 2.289053, 3.381869, 4.007963, 5.719058, 6.355028]
    y_coord_l = [6.368434, 8.41116, 10.61292, 11.27132, 14.46627, 15.74507]

    for i in range(-10, 750):
        x_axis.append(i / 100)
        p_y_axis.append(f(i / 100))
        l_y_axis.append(l(i / 100))


    plt.title("Overfitting")
    # plt.plot(x_coord_l, y_coord_l, label='data points', marker='x', linestyle='None', s=15)
    plt.scatter(x_coord_l, y_coord_l, s=25, label='data points', marker='x', c='red')
    plt.plot(x_axis, p_y_axis, label='polynomal function')
    plt.plot(x_axis, l_y_axis, label='linear function', linestyle='--', color='lightblue')
    plt.savefig('methodology_neuralNetwork_overfitting.png')
    plt.show()

    for i in range(0, len(x_coord_l)):
        print("{}{}, {}{}, ".format('{', x_coord_l[i], y_coord_l[i], '}'))
