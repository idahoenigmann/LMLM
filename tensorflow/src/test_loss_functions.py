import tensorflow as tf
import random
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math


def f(x):
    # return 0
    return x ** 3 - 6 * x ** 2 + 4 * x + 12


def g(x):
    return f(x) + ((math.sin(x / 2 + math.pi / 2) + 1) * random.uniform(-5, 5))


if __name__ == '__main__':
    actual = []
    estimated = []

    generate_data = True
    test_data = True

    if generate_data:
        for i in range(0, 600):
            j = i / 100
            actual.append(f(j))
            estimated.append(g(j))

        print("actual: {}".format(actual))
        print("estimated: {}".format(estimated))

        f = open("test_loss_functions_input_data.txt", "w+")
        f.write("{}\n{}".format(actual, estimated))
        f.close()

    if test_data:
        actual = []
        estimated = []
        f = open("test_loss_functions_input_data.txt", "r")
        actual_s = f.readline()
        estimated_s = f.readline()
        f.close()

        for ae in actual_s.split():
            actual.append(float(ae.strip(',').strip('[').strip(']')))

        for ee in estimated_s.split():
            estimated.append(float(ee.strip(',').strip('[').strip(']')))


        plt.title("Input Data")

        plt.scatter(range(0, 600), estimated, label='estimated', marker="+", c="orange")
        plt.plot(range(0, 600), actual, label='actual')

        plt.legend()

        plt.show()
        #plt.savefig('input_data.png')

        mse_l = []
        mae_l = []
        mape_l = []
        msle_l = []
        lc_l = []
        kld_l = []

        for i in range(0, len(actual), 10):
            # print("\ndata from 0 to {}".format(i))
            # perform test
            mse = tf.keras.losses.mean_squared_error(estimated[:i], actual[:i])
            mae = tf.keras.losses.mean_absolute_error(estimated[:i], actual[:i])
            mape = tf.keras.losses.mean_absolute_percentage_error(estimated[:i], actual[:i])
            msle = tf.keras.losses.mean_squared_logarithmic_error(estimated[:i], actual[:i])
            lc = tf.keras.losses.logcosh(estimated[:i], actual[:i])
            kld = tf.keras.losses.kullback_leibler_divergence(estimated[:i], actual[:i])

            with tf.Session() as sess:
                # print('mean_squared_error Loss: {}'.format(mse.eval()))
                mse_l.append(mse.eval())
                # print('mean_absolute_error Loss: {}'.format(mae.eval()))
                mae_l.append((mae.eval()))
                # print('mean_absolute_percentage_error Loss: {}'.format(mape.eval()))
                mape_l.append((mape.eval()))
                # print('mean_squared_logarithmic_error Loss: {}'.format(msle.eval()))
                msle_l.append((msle.eval()))
                # print('logcosh Loss: {}'.format(lc.eval()))
                lc_l.append((lc.eval()))
                # print('kullback_leibler_divergence Loss: {}'.format(kld.eval()))
                kld_l.append((kld.eval()))

        plt.title("Mean Squared Error")

        # plt.plot(range(0, 600), estimated, label='estimated')
        # plt.plot(range(0, 600), actual, label='actual')
        plt.subplot(3, 2, 1)
        plt.plot(range(0, 600, 10), mse_l, label='mean squared error')

        plt.subplot(3, 2, 2)
        plt.plot(range(0, 600, 10), mae_l, label='mean absolute error')

        plt.subplot(3, 2, 3)
        plt.plot(range(0, 600, 10), mape_l, label='mean absolute percentage error')

        plt.subplot(3, 2, 4)
        plt.plot(range(0, 600, 10), msle_l, label='mean squared logarithmic error')

        plt.subplot(3, 2, 5)
        plt.plot(range(0, 600, 10), lc_l, label='logcosh')

        plt.subplot(3, 2, 6)
        plt.plot(range(0, 600, 10), kld_l, label='kullback leibler divergence')

        plt.legend()

        plt.show()
        #plt.savefig('input_data.png')