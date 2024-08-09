import tensorflow as tf
import numpy as np
from keras import Sequential
import keras
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras

(X_train,y_train),(X_test,y_test) = keras.datasets.mnist.load_data()
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}")
print(type(X_train),type(y_train), type(X_test), type(y_test))

num_train_images = X_train.shape[0]
num_test_images = X_test.shape[0]
image_width = X_train.shape[1]
image_height = X_train.shape[2]
num_channels = 1
epochs = 10


print(image_height)
print(image_width)
plt.imshow(X_train[0], cmap="gray")
plt.show()
print(f"class: {y_train[0]}")

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2],num_channels)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2],num_channels)

X_train = X_train/255
X_test = X_test/255

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

print(type(X_train),type(y_train), type(X_test), type(y_test))
mlflow.set_experiment("Keras_mlflow")

def model() -> keras.Model:
    model = Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu',input_shape=(image_height, image_width,num_channels)))
    model.add(keras.layers.MaxPooling2D(2,2))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Conv2D(64,(2,2),activation='relu'))
    model.add(keras.layers.MaxPooling2D(2,2))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))


    return model

def plot(h):
    Loss = 0; Accuracy=1
    training = np.zeros((2,epochs)); testing = np.zeros((2,epochs))
    training[Loss] = h.history["loss"]
    testing[Loss] = h.history["val_loss"]
    training[Accuracy] = h.history["accuracy"]
    testing[Accuracy] = h.history["val_accuracy"]
    epoch = range(1,epochs+1)
    fig,axs = plt.subplots(1,2, figsize=(17,5))
    for i, label in zip((Loss, Accuracy),("loss", "accuracy")):
        axs[i].plot(epoch, training[i], 'b-', label='Training' + label)
        axs[i].plot(epoch, testing[i], 'r-', label='testing' + label)
        axs[i].set_title("Training and test" + label)
        axs[i].set_xlabel("epochs")
        axs[i].set_ylabel(label)
        axs[i].legend()
    plt.show()
    loss, accuraccy = model.evaluate(X_test,y_test)
    print(f"evaluate loss: {loss}, evaluate accuracy: {accuraccy}")
    return loss, accuraccy




def train(model: keras.Model):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    with mlflow.start_run():
        mlflow.log_param("epochs", 10)
        mlflow.log_param("batch_size", 64)
        mlflow.log_param("optimizer", "adam")
    
        mlflow.keras.autolog()
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=64, validation_data=(X_test,y_test))
        loss, accuracy = plot(history)
        mlflow.log_metric("evaluation loss", loss)
        mlflow.log_metric("evaluation accuracy", accuracy)
    mlflow.end_run()



model = model()
train(model)