import create_data as cd
from model import ConvViVIT
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import random


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

dataset = cd.UCF50dataset(dataset_dir ='E:/Pose_estimization/UCF50/',sequence_len=20,classes_list=["WalkingWithDog", "TaiChi", "Swing", "HorseRace"])
aug_data = cd.UCF50dataset(dataset_dir ='E:/Deepox/aug_data/',sequence_len=20,classes_list=[ '0','1','2','3'])
# Split the Data into Train ( 75% ) and Test Set ( 25% ).
features_train, features_test, labels_train, labels_test = dataset.create_dataset(normalize=False)
labels_train = dataset.augmentation(features_train,labels_train,dir='aug_data/')
#accesing the Augmented data and normalize it. To use for normalize the Train data
feature_train = aug_data.create_dataset(normalize=True)
features_test = dataset.normalize(features_test)
"""data =data = tf.data.Dataset.from_generator(
    generator=dataset.my_data_generator_function,  # Define this function to yield batches
    args=(feature_train),
    output_signature=(
        tf.TensorSpec(shape=(None,20,64, 64, 3), dtype=tf.float32),  # Input frames 
    )
)
data = data.batch(BATCH_SIZE).shuffle(buffer_size=1000).repeat()"""
def run_experiment(model,plots_path,method):
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
    checkpoint_filepath = 'E:/Pose_estimization/Models/vitbidirection_model_weights.h5'
    csv_logger_path = f'Logger/{method}.csv'
    checkpoint_callback = [keras.callbacks.CSVLogger(csv_logger_path, separator=',', append=True),
    
    keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    ),TensorBoard(log_dir='Logger/',
                         histogram_freq=1,
                         write_graph=True,
                         write_images=True,
                         update_freq='epoch',
                         profile_batch=2,
                         embeddings_freq=1)]

    history = model.fit(
        x=feature_train,
        y=labels_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.1,
        callbacks=[checkpoint_callback]
    )
    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(features_test,labels_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

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
    plt.savefig(f'{plots_path}/{method}.pdf')
    #model.load_weights(checkpoint_filepath)
    #_, accuracy, top_5_accuracy = model.evaluate(features_test,labels_test)
    #print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    #print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return history


if __name__ == '__main__':    
        con_vivit = ConvViVIT()
        vit_classifier = con_vivit.build_model()
        history = run_experiment(vit_classifier,plots_path='plots/',method='Test1')
