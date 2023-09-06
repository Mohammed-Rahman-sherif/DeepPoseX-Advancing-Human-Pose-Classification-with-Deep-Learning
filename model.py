# Import the required libraries.
import os
import cv2
import pafy
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers

from moviepy.editor import *
#%matplotlib inline

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional

# Import the required libraries.
import os
import cv2
import pafy
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt

from moviepy.editor import *
#%matplotlib inline

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers




# DATA

BATCH_SIZE = 8
AUTO = tf.data.AUTOTUNE
INPUT_SHAPE = (20, 64, 64, 3)
NUM_CLASSES = 4

# OPTIMIZER
LEARNING_RATE = 1e-3   #0.0001.
WEIGHT_DECAY = 1e-5

# TRAINING
EPOCHS = 65

# TUBELET EMBEDDING
PATCH_SIZE = (8, 8, 8)
NUM_PATCHES = (INPUT_SHAPE[0] // PATCH_SIZE[0]) ** 2

# ViViT ARCHITECTURE
LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 8

class TubeletEmbedding(layers.Layer):
    def __init__(self, embed_dim, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.projection = layers.Conv3D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="VALID",
        )
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))

    def call(self, videos):
        projected_patches = self.projection(videos)
        flattened_patches = self.flatten(projected_patches)
        return flattened_patches
    

class PositionalEncoder(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        _, num_tokens, _ = input_shape
        self.position_embedding = layers.Embedding(
            input_dim=num_tokens, output_dim=self.embed_dim
        )
        self.positions = tf.range(start=0, limit=num_tokens, delta=1)

    def call(self, encoded_tokens):
        # Encode the positions and add it to the encoded tokens
        encoded_positions = self.position_embedding(self.positions)
        encoded_tokens = encoded_tokens + encoded_positions
        return encoded_tokens
    
class ConvViVIT:
  
    
  def create_vivit_classifier(self,
    tubelet_embedder,
    positional_encoder,
    input_shape=INPUT_SHAPE,
    transformer_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    embed_dim=PROJECTION_DIM,
    layer_norm_eps=LAYER_NORM_EPS,
    num_classes=NUM_CLASSES):
    # Get the input layer
    inputs = layers.Input(shape=input_shape)
    # Create patches.
    patches = tubelet_embedder(inputs)
    # Encode patches.
    encoded_patches = positional_encoder(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization and MHSA
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.2
        )(x1, x1)


        # Skip connection
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer Normalization and MLP
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        #Try covolution instead MLP

        x3 = layers.Conv1D(filters=128, kernel_size=1, activation="relu")(x3)
        x3 = layers.Dropout(0.1)(x3)
        x3 = layers.Conv1D(filters=128, kernel_size=1, activation="relu")(x3)
        x3 = layers.LayerNormalization(epsilon=1e-6)(x3)
        """x3 = keras.Sequential(
            [
                layers.Dense(units=embed_dim * 4, activation=tf.nn.gelu),
                layers.Dense(units=embed_dim, activation=tf.nn.gelu),
            ]
        )(x3)"""

        # Skip connection
        encoded_patches = layers.Add()([x3, x2])
        encoded_patches = layers.Dropout(0.2)(encoded_patches)

    # Layer normalization and Global average pooling.
    #representation = layers.LayerNormalization(epsilon=layer_norm_eps)(encoded_patches)
    #representation = layers.GlobalAvgPool1D()(representation)

    # Classify outputs.
    #outputs = layers.Dense(units=num_classes, activation="softmax")(representation)

    # Create the Keras model.

    model = keras.Model(inputs=inputs, outputs=encoded_patches)
    #model.summary()
    return model
  
  def build_model(self):

    inputs = layers.Input(shape=(20,64, 64, 3))
    """for i in range(SEQUENCE_LENGTH):
        frame_features = vit_model(inputs[:,i,:,:,:])
        features.append(frame_features)
    vit_features = tf.stack(features, axis=1)
    #set the shape of LSTM
    lstm_shape = vit_features.shape"""

    vit_features = self.create_vivit_classifier(
        tubelet_embedder=TubeletEmbedding(
            embed_dim=PROJECTION_DIM, patch_size=PATCH_SIZE
        ),
        positional_encoder=PositionalEncoder(embed_dim=PROJECTION_DIM),
    )(inputs)
    #print("shape:",vit_features.shape)
    #(shape: (None, 20, 144, 64))
    #data reshape
    #vit_features = layers.Reshape((20,144*64))(vit_features)
    #print("after reshape:",vit_features.shape)
    x1 = Bidirectional(LSTM(512, return_sequences=True, activation='tanh'), input_shape=(128, 128))(vit_features)
    x2= Bidirectional(LSTM(512, return_sequences=True,activation='tanh'))(x1)
    x3 =Bidirectional(LSTM(512, return_sequences=True,activation='tanh',kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)))(x2)
    x4 =Bidirectional(LSTM(512, return_sequences=True,activation='tanh'))(x3)
    x4 = layers.LayerNormalization(epsilon=1e-6)(x4)
    x5 =Bidirectional(LSTM(512, return_sequences=True,activation='tanh',dropout=0.2))(x4)
    x6 =Bidirectional(LSTM(512, return_sequences=True,activation='tanh'))(x5)
    x7 =Bidirectional(LSTM(512, return_sequences=True,activation='tanh',kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)))(x6)  
    x8 =Flatten()(x7)
    outputs = Dense(4, activation = "softmax")(x8)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # display the model summary
    model.summary()
    return model
  