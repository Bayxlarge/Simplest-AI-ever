import tensorflow as tf
from tensorflow import keras
import random 
import numpy as np

X_train = []
Y_train = []
for i in range (5000):
  rnd = random.randint(1,76)
  X_train.append(rnd)
  Y_train.append(rnd*7)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1), metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=15, batch_size=10)

X_test = [30, 75] 
X_test = np.array(X_test)

AI_output = model.predict(X_test)
print (AI_output)

while True:
  num = int(input("> "))
  AA = []
  AA.append(num)
  AA = np.array(AA)
  a = model.predict(AA)
  print(a)
