import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# ===============================
# 1️⃣ Cargar modelo CNN
# ===============================
model = load_model("models/mnist_cnn.keras")

# ===============================
# 2️⃣ Cargar MNIST (test)
# ===============================
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizar
x_test = x_test / 255.0

# Ajustar shape para CNN
x_test = x_test.reshape(-1, 28, 28, 1)

# ===============================
# 3️⃣ Evaluar modelo
# ===============================
loss, accuracy = model.evaluate(x_test, y_test, verbose=2)

print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")
print(f"❌ Test Loss: {loss:.4f}")

# ===============================
# 4️⃣ Predicción sobre MNIST
# ===============================
prediction = model.predict(x_test[:1])
predicted_label = prediction.argmax()

print("\n🧠 Predicción del modelo:")
print("Etiqueta real:", y_test[0])
print("Etiqueta predicha:", predicted_label)

# ===============================
# 5️⃣ Predicción imagen externa
# ===============================
img = Image.open("data/my_digit.png").convert("L")
img = img.resize((28, 28))
img = ImageOps.invert(img)

img_array = np.asarray(img, dtype=np.float32)
img_array /= 255.0
img_array = img_array.reshape(1, 28, 28, 1)

prediction_ext = model.predict(img_array)
predicted_digit = prediction_ext.argmax()

print("\n🖼️ Predicción de imagen externa:")
print("Número predicho:", predicted_digit)
