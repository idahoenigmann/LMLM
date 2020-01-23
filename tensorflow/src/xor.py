import tensorflow as tf

input_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
output_data = [[0], [1], [1], [0]]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, name="dense_1", input_shape=(2,), activation=tf.keras.activations.sigmoid),
    tf.keras.layers.Dense(2, name="dense_2", activation=tf.keras.activations.sigmoid),
    tf.keras.layers.Dense(1, name="dense_3", activation=tf.keras.activations.sigmoid)
])

model.summary()

model.compile(loss=tf.keras.losses.mean_absolute_error, learning_rate=0.15)
model.fit(input_data, output_data, epochs=10000)
