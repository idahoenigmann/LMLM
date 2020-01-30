import tensorflow as tf

input_data = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
output_data = [[0.0], [1.0], [1.0], [0.0]]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, name="dense_1", input_shape=(2,), activation=tf.keras.activations.sigmoid),
    tf.keras.layers.Dense(2, name="dense_2", activation=tf.keras.activations.sigmoid),
    tf.keras.layers.Dense(1, name="dense_3", activation=tf.keras.activations.sigmoid)
])

model.summary()

model.compile(loss=tf.keras.losses.mean_absolute_error, learning_rate=0.15)

for i in range(1000):
    model.fit(input_data, output_data, epochs=100, verbose=0)

    idx = 0
    for actual in model.predict(input_data):
        print("expected: {}; actual: {}".format(output_data[idx], actual))
        idx += 1
    print()
