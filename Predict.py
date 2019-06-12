import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.utils import normalize

df_test = pd.read_csv("test.csv")
test = df_test.iloc[:,:].values.astype("float32")
test = test.reshape(test.shape[0],28,28)
x_test = normalize(test, axis=1)
x_test = np.array(x_test).reshape(-1,28,28,1)
model = tf.keras.models.load_model("CNN_1")
predictions = model.predict(x_test)

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

res.to_csv("CNN.csv")
