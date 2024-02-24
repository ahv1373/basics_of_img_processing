from typing import Optional, Tuple, List
import cv2
import numpy as np
import random
import imutils
import tensorflow as tf


class CustomCNNClassifier:
    def __init__(self, number_of_classes: int = 10, input_shape: Tuple[int, int, int] = (28, 28, 1),
                 optimizer: str = 'adam', loss: str = 'categorical_crossentropy', learning_rate: float = 0.001,
                 metrics: str = 'accuracy', epochs: int = 5, batch_size: int = 32):
        self.history: Optional[tf.keras.callbacks.History] = None
        self.epochs: int = epochs
        self.batch_size: int = batch_size
        self.input_shape: Tuple[int, int, int] = input_shape  # Tuple of 3 integers (height, width, channels)
        self.num_classes: int = number_of_classes
        self.optimizer: str = optimizer  # 'adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adamax', 'nadam'
        self.loss: str = loss  # 'categorical_crossentropy', 'sparse_categorical_crossentropy', 'binary_crossentropy'
        self.metrics: List[str] = [metrics]
        self.learning_rate = learning_rate
        self.model: tf.keras.models.Model = self.build_model()

    def build_model(self) -> tf.keras.models.Model:
        model_ = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape,
                                   padding='same', strides=(1, 1)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', strides=(1, 1)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', strides=(1, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=self.num_classes, activation='softmax')  # softmax for multi-class
        ])
        model_.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        return model_

    def train(self, x_train_: tf.Tensor, y_train_: tf.Tensor, validation_data: Optional[Tuple[tf.Tensor, tf.Tensor]] = None) -> None:
        self.history = self.model.fit(x_train_, y_train_, epochs=self.epochs, batch_size=self.batch_size,
                                      validation_data=validation_data)

    def save_model(self, model_path: str = "cnn_model.h5") -> None:
        self.model.save(model_path)

    def load_model(self, model_path: str = "cnn_model.h5") -> None:
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, x: tf.Tensor) -> int:
        prediction_tensor = self.model.predict(x)
        return np.argmax(prediction_tensor)

    def evaluate(self, x_test_: tf.Tensor, y_test_: tf.Tensor) -> List[float]:
        return self.model.evaluate(x_test_, y_test_)

    def summary(self) -> None:
        self.model.summary()

    def plot_history(self):
        import matplotlib.pyplot as plt
        acc = self.history.history['accuracy']
        loss = self.history.history['loss']
        epochs = range(1, len(acc) + 1)
        # plot the accuracy and loss in one plot and two y-axis
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(epochs, acc, 'g', label='Training Accuracy')
        ax1.plot(epochs, self.history.history['val_accuracy'], 'r', label='Validation Accuracy')
        ax2.plot(epochs, loss, 'b', label='Training Loss')
        ax2.plot(epochs, self.history.history['val_loss'], 'y', label='Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy', color='g')
        ax2.set_ylabel('Loss', color='b')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        # grid
        ax1.grid()
        ax2.grid()
        # show and save the plot
        plt.savefig('training_plot.png', dpi=300)
        plt.show()


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape((60000, 28, 28, 1))
    x_train = x_train.astype('float32') / 255
    x_test = x_test.reshape((10000, 28, 28, 1))
    x_test = x_test.astype('float32') / 255
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    # split the training data into a training and validation set
    x_train, x_val = x_train[10000:], x_train[:10000]
    y_train, y_val = y_train[10000:], y_train[:10000]

    model = CustomCNNClassifier()
    # Train the model
    model.train(x_train, y_train, validation_data=(x_val, y_val))
    model.evaluate(x_test, y_test)
    model.plot_history()
    model.summary()
    model.save_model()

    # Load the model for some predictions
    model.load_model()
    while True:
        index = random.randint(0, len(x_test))
        img = x_test[index]
        img = np.expand_dims(img, axis=0)  # Add the batch dimension
        # convert to tensorflow tensor
        img = tf.convert_to_tensor(img, dtype=tf.float32)
        prediction = model.predict(img)
        opencv_img = imutils.resize(x_test[index], width=300)
        opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_GRAY2BGR)
        cv2.putText(opencv_img, f"Predicted: {prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("Image", opencv_img)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
