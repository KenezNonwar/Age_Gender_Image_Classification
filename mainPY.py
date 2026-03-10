import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import Sequential
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split


path="dataset/crop_part1"
IMG_SIZE=200

def preprocessing():
    age, gender, image = [], [], []
    for file in os.listdir(path):
        age.append(int(file.split("_")[0]))
        gender.append(int(file.split("_")[1]))
        image.append(path + "/" + file)
    df = pd.DataFrame({"age": age, "gender": gender, "image_path": image})
    df = df.loc[(df['age'] >= 5) & (df['age'] <= 80), ['age', 'gender', 'image_path']]
    return df

def data_visual(df):
    plt.hist(df["age"], bins=20)
    plt.xticks(np.arange(0, 100, 5))
    plt.show()

def load_image(path):
  img=cv2.imread(path)
  img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  img=cv2.resize(img,(200,200))
  img=img/255.0
  img = np.expand_dims(img, axis=0)
  return img

def create_data(df,batch_size=32):
  paths=df['image_path'].values
  ages=df['age'].values
  genders=df['gender'].values

  def load(path,age,gender):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return img, {"age": age, "gender": gender}

  dataset = tf.data.Dataset.from_tensor_slices((paths, ages, genders))
  dataset = dataset.map(lambda p, a, g: load(p, a, g))
  dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
  return dataset

def train_test(df):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_ds = create_data(train_df)
    test_ds = create_data(test_df)
    return train_ds, test_ds

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.models import Model

def modeling():
    vggnet = VGG16(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    vggnet.trainable = False
    for Layers in vggnet.layers[-4:]:
        Layers.trainable = True

    out = vggnet.layers[-1].output
    flat = Flatten()(out)

    dense0A = Dense(128, activation="relu")(flat)
    dense0B = Dense(128, activation="relu")(flat)

    dense1A = Dense(128, activation="relu")(dense0A)
    dense1B = Dense(128, activation="relu")(dense0B)

    dense2A = Dense(64, activation="relu")(dense1A)
    dense2B = Dense(64, activation="relu")(dense1B)

    dropoutA = Dropout(0.5)(dense2A)
    dropoutB = Dropout(0.5)(dense2B)

    outputA = Dense(1, activation='linear', name="age")(dropoutA)
    outputB = Dense(1, activation="sigmoid", name="gender")(dropoutB)

    model = Model(inputs=vggnet.input, outputs=[outputA, outputB])
    return model
def model_plotting(model):
    from tensorflow.keras.utils import plot_model
    plot_model(model)

def comp(model):
    model.compile(
        optimizer="adam",
        loss={"age": 'mae', "gender": "binary_crossentropy"},
        metrics={"age": "mae", "gender": "accuracy"},
        loss_weights={"age": 0.5, "gender": 1})

def train(model, train_ds, test_ds):
    model.fit(train_ds, validation_data=test_ds, epochs=10,
              callbacks=[
                  tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                  tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)
              ])
    model.save('model_AGE_GENDER.keras')

import warnings
warnings.filterwarnings('ignore')
from IPython.display import Image, display
from tensorflow.keras.models import load_model

def predict(df):
    # Download the trained model and paste the Link here
    model = load_model('model/model_AGE_GENDER.keras')

    n = np.random.randint(0, len(df))
    T = df['image_path'][n]
    print(T)
    display(Image(filename=df['image_path'][n]))
    print("Age", df['age'][n])
    sex = "Female" if df['gender'][n] >= 0.5 else "Male"
    print("Gender", sex)
    im = load_image(T)
    age_pred, gender_pred = model.predict(im)
    print("Age Predicted:", np.round(age_pred[0]))
    sex_pred = "Female" if gender_pred >= 0.5 else "Male"
    print("Gender Predicted: ", sex_pred)

def main():
    df=preprocessing()
    train_test(df)
    modeling()
    predict(df)

main()