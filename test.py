import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(y_train)
print(y_test)

x_train, x_test = x_train / 255.0, x_test / 255.0


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
predictions2 = model(x_train[:3]).numpy()
predictions3 = model(x_train[:10]).numpy()
print(predictions)
print(predictions2)
print(predictions3)

tf.nn.softmax(predictions3).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:10], predictions3).numpy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)

probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

result = probability_model.predict(x_test)

print(result[0])
# The predicted result
print(np.argmax(result[0]))
# The wanted result
print(y_test[0])
