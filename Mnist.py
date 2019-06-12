import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten,Dense
from tensorflow.keras.utils import normalize


df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

train = df_train.iloc[:,1:].values.astype("float32")
labels = df_train.iloc[:,0].values.astype("int32")
test = df_test.iloc[:,:].values.astype("float32")
train = train.reshape(42000,28,28)
test = test.reshape(test.shape[0],28,28)

x_train = tf.keras.utils.normalize(train, axis=1)
x_test = tf.keras.utils.normalize(test, axis=1)


model = Sequential()
model.add(Flatten())
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dense(10, activation=tf.nn.softmax))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,labels,epochs=5)

model.save("Mnist.model")
# val_loss,val_acc = model.evaluate(train,labels)