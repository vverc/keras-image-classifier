import tensorflow as tf


def create_model(num_width=64, num_layers=1, activation="sigmoid", learning_rate=0.001):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

    for i in range(num_layers):
        model.add(tf.keras.layers.Dense(num_width, activation=activation))

    model.add(tf.keras.layers.Dense(10, activation="sigmoid"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return model
