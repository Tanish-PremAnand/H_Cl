
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.callbacks import *
from keras.layers.normalization import BatchNormalization


# convert grayscale vector data into image
def vector2Img28x28(imgLoc, saveName, im_loc):
    train_x = pd.read_csv(imgLoc)
    train_x.drop('label', 1, inplace=True)
    #     print(train_x)
    a = train_x.iloc[im_loc]
    #     print(a)
    a = a.values.reshape(28, 28)
    print(a.shape)
    array = a.astype(np.uint8)
    # #     print(array)
    img = Image.fromarray(array)
    img.save(saveName + '.png')
    plt.imshow(img, interpolation='Nearest', cmap='binary_r')
    plt.show()
    return img


# a_img = vector2Img28x28(r"C:\Users\tanis\Documents\Projects\H_Cl\experiment\sign-language-mnist\sign_mnist_train.csv",
#                         'b', 13987)

train_data = pd.read_csv("sign_mnist_train.csv", header=0)
test_data = pd.read_csv("sign_mnist_test.csv", header=0)


# Split train and test data into label and feature datasets
def label_feature_split(data):
    X = []
    y = []
    for i in range(len(data)):
        row = data.iloc[i, 1:]
        row = row.values.reshape(28, 28)
        X.append(row)
        y.append(data.iloc[i, 0])
    print("trainData loaded", len(X))
    print("labels loaded", len(y))
    X = np.array(X, dtype="uint8")
    X = X.reshape(len(data), 28, 28, 1)  # Needed to reshape so CNN knows it's different images
    y = np.array(y)
    return X, y


X, y = label_feature_split(train_data)
X_test, y_test = label_feature_split(test_data)

ts = 0.25
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=ts, random_state=353)

# Model Callbacks   
es = EarlyStopping(monitor="val_accuracy", mode='max', patience=3, verbose=1)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(25, activation='softmax'))


model.compile(optimizer='Adadelta', loss='sparse_categorical_crossentropy', metrics=['accuracy'])



print(model.summary())

model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1, validation_data=(X_val, y_val), callbacks=[es])

test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy: {:2.2f}%'.format(test_acc * 100))

predictions = model.predict_classes(X_test)

y_pred = predictions

# cnfsnmtrx = pd.DataFrame(metrics.confusion_matrix(y_test, y_pred))

# Save model and weights to disk
model_json = model.to_json()
with open("H_Cl_model_1.json", 'w') as json_file:
    json_file.write(model_json)
model.save_weights("H_Cl_model_weights_1.h5")
print('model saved to local storage')

