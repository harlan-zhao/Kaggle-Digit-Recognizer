import tensorflow as tf
import pandas as pd
import numpy as np

df_test = pd.read_csv("test.csv")
test = df_test.iloc[:,:].values.astype("float32")
test = test.reshape(test.shape[0],28,28)

model = tf.keras.models.load_model("Mnist.model")
predictions = model.predict(test)

n=1
ID = []
Label = []
for line in predictions:
    ID.append(n)
    Label.append(np.argmax(line))
    n+=1
res = pd.DataFrame()
res["ImageId"] = ID
res["Label"] = Label

res.to_csv("Mnist.csv")
