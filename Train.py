import tensorflow as tf
from tensorflow.keras import layers, models

# 1️⃣ Cargar dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2️⃣ Normalizar
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3️⃣ Ajustar forma 
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# 4️⃣ Construir CNN
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 5️⃣ Compilar
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 6️⃣ Entrenar
model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_split=0.1
)

# 7️⃣ Guardar modelo 
model.save("models/mnist_cnn.keras")

print("✅ Modelo CNN guardado correctamente")

