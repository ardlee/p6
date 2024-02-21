# from models.model import Model
# from tensorflow.keras import Sequential, layers
# from tensorflow.keras.layers.experimental.preprocessing import Rescaling
# from tensorflow.keras.optimizers import RMSprop, Adam

# class BasicModel(Model):
#     def _define_model(self, input_shape, categories_count):
#         # Your code goes here
#         # you have to initialize self.model to a keras model
#         self.model = models.Sequential()
        
#         self.model.add(layers.Conv2D(8, (3, 3), activation='relu', input_shape=input_shape))
#         self.model.add(layers.MaxPooling2D((2, 2)))
#         self.model.add(layers.Conv2D(16, (3, 3), activation='relu'))
#         self.model.add(layers.MaxPooling2D((2, 2)))
#         self.model.add(layers.Conv2D(32, (3, 3), activation='relu'))

#         self.model.add(layers.Flatten())

#         self.model.add(layers.Dense(64, activation='relu'))

#         self.model.add(layers.Dense(categories_count, activation='softmax'))
    
#     def _compile_model(self):
#         # Your code goes here
#         # you have to compile the keras model, similar to the example in the writeup
#         self.model.compile(
#             optimizer= tf.keras.optimizers.RMSprop(learning_rate=0.001),
#             # optimizer='rmsprop',
#             loss='categorical_crossentropy',
#             metrics=['accuracy'],
#         )

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import RMSprop
from models.model import Model

class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Initialize self.model to a Sequential model
        self.model = models.Sequential()
        
        # Add layers to the model
        self.model.add(layers.Conv2D(8, (3, 3), activation='relu', input_shape=input_shape))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(16, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(categories_count, activation='softmax'))
    
    def _compile_model(self):
        # Compile the Keras model
        self.model.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),  # Using RMSprop optimizer
            loss='categorical_crossentropy',  # Categorical cross-entropy loss function
            metrics=['accuracy'],  # Monitoring accuracy during training
        )
