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

    x_axis = []

    with tf.Session() as sess:
        for i in range(-100, 100):
            elu_l.append(tf.keras.activations.elu(float(i) / 20).eval())
            selu_l.append(tf.keras.activations.selu(float(i) / 20).eval())
            softplus_l.append(tf.keras.activations.softplus(float(i) / 20).eval())
            softsign_l.append(tf.keras.activations.softsign(float(i) / 20).eval())
            relu_l.append(tf.keras.activations.relu(float(i) / 20).eval())
            tanh_l.append(tf.keras.activations.tanh(float(i) / 20).eval())
            sigmoid_l.append(tf.keras.activations.sigmoid(float(i) / 20).eval())
            exponential_l.append(tf.keras.activations.exponential(float(i) / 20).eval())
            linear_l.append(tf.keras.activations.linear(float(i) / 20))
            x_axis.append(float(i) / 20)

    plt.title("Exponential Linear Unit (ELU)")
    plt.plot(x_axis, elu_l, label='elu')
    plt.savefig('methodology_neuralNetwork_activationFunction_elu.png')
    plt.show()
    #
    plt.title("Scaled Exponential Linear Unit (SELU)")
    plt.plot(x_axis, selu_l, label='selu')
    plt.savefig('methodology_neuralNetwork_activationFunction_selu.png')
    plt.show()
    #
    plt.title("SoftPlus")
    plt.plot(x_axis, softplus_l, label='softplus')
    plt.savefig('methodology_neuralNetwork_activationFunction_softplus.png')
    plt.show()
    #
    plt.title("Softsign")
    plt.plot(x_axis, softsign_l, label='softsign')
    plt.savefig('methodology_neuralNetwork_activationFunction_softsign.png')
    plt.show()
    #
    plt.title("Rectified Linear Unit (ReLU)")
    plt.plot(x_axis, relu_l, label='relu')
    plt.savefig('methodology_neuralNetwork_activationFunction_relu.png')
    plt.show()
    #
    plt.title("Tanh")
    plt.plot(x_axis, tanh_l, label='tanh')
    plt.savefig('methodology_neuralNetwork_activationFunction_tanh.png')
    plt.show()
    #
    plt.title("Sigmoid")
    plt.plot(x_axis, sigmoid_l, label='sigmoid')
    plt.savefig('methodology_neuralNetwork_activationFunction_sigmoid.png')
    plt.show()

    plt.title("Exponential")
    plt.plot(x_axis, exponential_l, label='exponential')
    plt.savefig('methodology_neuralNetwork_activationFunction_exponential.png')
    plt.show()

    plt.title("Linear")
    plt.plot(x_axis, linear_l, label='linear')
    plt.savefig('methodology_neuralNetwork_activationFunction_linear.png')
    plt.show()
