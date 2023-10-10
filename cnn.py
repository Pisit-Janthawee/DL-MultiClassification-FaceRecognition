import time
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from IPython.display import display, HTML


class CNN:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.optimizer = None
        # Build the CNN model
        self.model = self.build_model()

    def build_model(self):
        model = keras.Sequential()

        # Convolutional layers
        model.add(layers.Conv2D(32, (3, 3), activation='relu',
                  input_shape=self.input_shape))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        # Flatten the output of the convolutional layers
        model.add(layers.Flatten())

        # Fully connected layers
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        self.optimizer = tf.keras.optimizers.Adam()
        # Compile the model
        model.compile(optimizer=self.optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(
                          from_logits=True),
                      metrics=['accuracy'])

        return model

    def get_config(self):
        # Configuration dictionary
        config_param = {
            'Model Configuration': self.model.get_config(),
            'Optimizer parameters Configuration': self.optimizer.get_config(),
            'Model Architecture Configuration': {
                'input_shape': self.input_shape,
            },
            'Model Training Configuration': {
                'batch_size': self.batch_size,
                'epochs': self.epochs
            },
        }
        config_df = pd.DataFrame([self.get_config()])
        display(HTML(config_df.to_html()))
        return config_param

    def summary(self):
        self.model.summary()
    
    def save_model(self, name):
        os.makedirs('artifacts', exist_ok=True)
        save_path = os.path.abspath(f'artifacts/{name}.h5')
        self.model.save(save_path)

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size):
        self.batch_size = batch_size
        self.epochs = epochs
        start_time = time.time()
        history = self.model.fit(X_train, y_train,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_data=(X_val, y_val))
        end_time = time.time()
        self.training_time = end_time - start_time
        return history

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def predict(self, X):
        return self.model.predict(X)
