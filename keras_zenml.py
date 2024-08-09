import tensorflow as tf
import numpy as np
import numpy
from keras import Sequential
import keras
import matplotlib.pyplot as plt
from zenml import step, pipeline
from typing import Annotated, Tuple

@step
def load_dataset() -> Tuple[
    Annotated[numpy.ndarray, "X_train"],
    Annotated[numpy.ndarray, "X_test"],
    Annotated[numpy.ndarray, "y_train"],
    Annotated[numpy.ndarray, "y_test"],
]:
    """Load the MNIST dataset."""
    (X_train,y_train),(X_test,y_test) = keras.datasets.mnist.load_data()
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")

    return X_train, X_test, y_train, y_test

@step
def preporcessing(
    X_train: numpy.ndarray, 
    X_test: numpy.ndarray, 
    y_train: numpy.ndarray, 
    y_test: numpy.ndarray
    ) -> Tuple[
        Annotated[numpy.ndarray, "X_train"],
        Annotated[numpy.ndarray, "X_test"],
        Annotated[numpy.ndarray, "y_train"],
        Annotated[numpy.ndarray, "y_test"],
    ]:
    """Preprocess the MNIST dataset."""
    num_channels = 1

    plt.imshow(X_train[0], cmap="gray")
    plt.show()
    print(f"class: {y_train[0]}")

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2],num_channels)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2],num_channels)

    X_train = X_train/255.0
    X_test = X_test/255.0

    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)


    return X_train, X_test, y_train, y_test



@step 
def build_model() -> keras.Model:
    """Build a CNN model."""
    input_shape = (28,28,1)
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.3),
        keras.layers.Conv2D(64,(2,2),activation='relu'),
        keras.layers.MaxPooling2D(2,2),
        keras.layers.Dropout(0.3),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax'),

    ])
    return model

@step
def train(X_train: numpy.ndarray, 
    y_train: numpy.ndarray, 
    X_test: numpy.ndarray, 
    y_test: numpy.ndarray,
    model: keras.Model) -> float:
    """Train the model and return the accuracy."""
    epochs = 10
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=64, validation_data=(X_test,y_test))
    loss, accuraccy = model.evaluate(X_test,y_test)
    print(f"evaluate loss: {loss}, evaluate accuracy: {accuraccy}")
    return accuraccy





@pipeline
def main():
    """Run the full pipeline."""
    X_train, X_test, y_train, y_test = load_dataset()
    X_train, X_test, y_train, y_test = preporcessing(X_train, X_test, y_train, y_test)
    model = build_model()

    train(X_train, y_train,X_test, y_test,model)

if __name__ == "__main__":
    main()