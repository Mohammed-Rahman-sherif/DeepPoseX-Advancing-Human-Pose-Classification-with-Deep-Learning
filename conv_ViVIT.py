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

"""physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
"""

seed_constant = 27
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)

img_height,img_width = 64,64
#specifiying the number of frames fed into the model
SEQUENCE_LENGTH = 20
# Specify the directory containing the UCF50 dataset.
DATASET_DIR = "UCF50"
# Specify the list containing the names of the classes used for training. Feel free to choose any set of classes.
CLASSES_LIST = ["WalkingWithDog", "TaiChi", "Swing", "HorseRace"]

def frames_extraction(video_path):
    '''
    This function will extract the required frames from a video after resizing and normalizing them.
    Args:
        video_path: The path of the video in the disk, whose frames are to be extracted.
    Returns:
        frames_list: A list containing the resized and normalized frames of the video.
    '''

    # Declare a list to store video frames.
    frames_list = []

    # Read the Video File using the VideoCapture object.
    video_reader = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)

    # Iterate through the Video Frames.
    for frame_counter in range(SEQUENCE_LENGTH):

        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

        # Reading the frame from the video.
        success, frame = video_reader.read()

        # Check if Video frame is not successfully read then break the loop
        if not success:
            break

        # Resize the Frame to fixed height and width.
        resized_frame = cv2.resize(frame, (img_height,img_width))

        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = resized_frame / 255

        # Append the normalized frame into the frames list
        frames_list.append(normalized_frame)

    # Release the VideoCapture object.
    video_reader.release()

    # Return the frames list.
    return frames_list

def create_dataset():
    '''
    This function will extract the data of the selected classes and create the required dataset.
    Returns:
        features:          A list containing the extracted frames of the videos.
        labels:            A list containing the indexes of the classes associated with the videos.
        video_files_paths: A list containing the paths of the videos in the disk.
    '''

    # Declared Empty Lists to store the features, labels and video file path values.
    features = []
    labels = []
    video_files_paths = []

    # Iterating through all the classes mentioned in the classes list
    for class_index, class_name in enumerate(CLASSES_LIST):

        # Display the name of the class whose data is being extracted.
        print(f'Extracting Data of Class: {class_name}')

        # Get the list of video files present in the specific class name directory.
        files_list = os.listdir(os.path.join(DATASET_DIR, class_name))

        # Iterate through all the files present in the files list.
        for file_name in files_list:

            # Get the complete video path.
            video_file_path = os.path.join(DATASET_DIR, class_name, file_name)

            # Extract the frames of the video file.
            frames = frames_extraction(video_file_path)

            # Check if the extracted frames are equal to the SEQUENCE_LENGTH specified above.
            # So ignore the vides having frames less than the SEQUENCE_LENGTH.
            if len(frames) == SEQUENCE_LENGTH:

                # Append the data to their repective lists.
                features.append(frames)
                labels.append(class_index)
                video_files_paths.append(video_file_path)

    # Converting the list to numpy arrays
    features = np.asarray(features)
    labels = np.array(labels)

    # Return the frames, class index, and video file path.
    return features, labels, video_files_paths

# Create the dataset.
features, labels, video_files_paths = create_dataset()

one_hot_encoded_labels = to_categorical(labels)

# Split the Data into Train ( 75% ) and Test Set ( 25% ).
features_train, features_test, labels_train, labels_test = train_test_split(features, one_hot_encoded_labels, test_size = 0.25, shuffle = True, random_state = seed_constant)

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
    
def create_vivit_classifier(
    tubelet_embedder,
    positional_encoder,
    input_shape=INPUT_SHAPE,
    transformer_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    embed_dim=PROJECTION_DIM,
    layer_norm_eps=LAYER_NORM_EPS,
    num_classes=NUM_CLASSES,
):
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
    model.summary()
    return model


trans_model = create_vivit_classifier(
        tubelet_embedder=TubeletEmbedding(
            embed_dim=PROJECTION_DIM, patch_size=PATCH_SIZE
        ),
        positional_encoder=PositionalEncoder(embed_dim=PROJECTION_DIM),
    )

from tensorflow.keras import regularizers
def build_model():

    inputs = layers.Input(shape=(20,64, 64, 3))
    """for i in range(SEQUENCE_LENGTH):
        frame_features = vit_model(inputs[:,i,:,:,:])
        features.append(frame_features)
    vit_features = tf.stack(features, axis=1)
    #set the shape of LSTM
    lstm_shape = vit_features.shape"""

    vit_features = create_vivit_classifier(
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

trans_model = build_model()

def run_experiment(model):
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )
    """
    However, you mentioned earlier that you were using the SparseCategoricalCrossentropy loss function, which expects integer labels rather than one-hot-encoded vectors. If you're using one-hot-encoded labels, you should switch to the CategoricalCrossentropy loss function instead.


    """
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[
            keras.metrics.MeanAbsoluteError(name='mean_absolute_error'),
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )
    #define where the model wants to save
    checkpoint_filepath = 'Models/vitbidirection_model_weights.h5'
    csv_logger_path = 'Logger/LogAdam5.csv'
    checkpoint_callback = [keras.callbacks.CSVLogger(csv_logger_path, separator=',', append=True),
    keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )]

    history = model.fit(
        x=features_train,
        y=labels_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.1,
        callbacks=[checkpoint_callback]
    )

    # Create a single graph for both Loss and Mean Absolute Error with different legends
    plt.figure(figsize=(8, 6))

    # Plot Loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')

    # Plot Mean Absolute Error
    plt.plot(history.history['mean_absolute_error'], label='Training MAE')
    plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')

    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.grid()
    plt.title('Metrics Visualization')
    plt.legend()

    plt.show()
    #model.load_weights(checkpoint_filepath)
    #_, accuracy, top_5_accuracy = model.evaluate(features_test,labels_test)
    #print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    #print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return history


vit_classifier = build_model()
history = run_experiment(vit_classifier)