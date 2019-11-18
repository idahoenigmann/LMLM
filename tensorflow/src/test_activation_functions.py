import tensorflow as tf
import random
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math

if __name__ == '__main__':
    elu_l = []
    selu_l = []
    softplus_l = []
    softsign_l = []
    relu_l = []
    tanh_l = []
    sigmoid_l = []
    exponential_l = []
    linear_l = []

    with tf.Session() as sess:
        for i in range(-25, 25):
            elu_l.append(tf.keras.activations.elu(float(i)).eval())
            selu_l.append(tf.keras.activations.selu(float(i)).eval())
            softplus_l.append(tf.keras.activations.softplus(float(i)).eval())
            softsign_l.append(tf.keras.activations.softsign(float(i)).eval())
            relu_l.append(tf.keras.activations.relu(float(i)).eval())
            tanh_l.append(tf.keras.activations.tanh(float(i)).eval())
            sigmoid_l.append(tf.keras.activations.sigmoid(float(i)).eval())
            exponential_l.append(tf.keras.activations.exponential(float(i)).eval())
            linear_l.append(tf.keras.activations.linear(float(i)))

    plt.title("Exponential Linear Unit (ELU)")
    plt.plot(range(-25, 25), elu_l, label='elu')
    plt.savefig('methodology_neuralNetwork_activationFunction_elu.png')
    plt.show()

    plt.title("Scaled Exponential Linear Unit (SELU)")
    plt.plot(range(-25, 25), selu_l, label='selu')
    plt.savefig('methodology_neuralNetwork_activationFunction_selu.png')
    plt.show()

    plt.title("SoftPlus")
    plt.plot(range(-25, 25), softplus_l, label='softplus')
    plt.savefig('methodology_neuralNetwork_activationFunction_softplus.png')
    plt.show()

    plt.title("Softsign")
    plt.plot(range(-25, 25), softsign_l, label='softsign')
    plt.savefig('methodology_neuralNetwork_activationFunction_softsign.png')
    plt.show()

    plt.title("Rectified Linear Unit (ReLU)")
    plt.plot(range(-25, 25), relu_l, label='relu')
    plt.savefig('methodology_neuralNetwork_activationFunction_relu.png')
    plt.show()

    plt.title("Tanh")
    plt.plot(range(-25, 25), tanh_l, label='tanh')
    plt.savefig('methodology_neuralNetwork_activationFunction_tanh.png')
    plt.show()

    plt.title("Sigmoid")
    plt.plot(range(-25, 25), sigmoid_l, label='sigmoid')
    plt.savefig('methodology_neuralNetwork_activationFunction_sigmoid.png')
    plt.show()

    plt.title("Exponential")
    plt.plot(range(-25, 25), exponential_l, label='exponential')
    plt.savefig('methodology_neuralNetwork_activationFunction_exponential.png')
    plt.show()

    plt.title("Linear")
    plt.plot(range(-25, 25), linear_l, label='linear')
    plt.savefig('methodology_neuralNetwork_activationFunction_linear.png')
    plt.show()
