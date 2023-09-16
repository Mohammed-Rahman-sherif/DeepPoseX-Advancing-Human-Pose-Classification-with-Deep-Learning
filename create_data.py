import os
import numpy as np
import cv2
from Augment import augmentors as va
from PIL import Image, ImageSequence
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import random
import tensorflow as tf

seed_constant = 27
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)


class UCF50dataset:
  def __init__(self,dataset_dir,sequence_len,classes_list,image_height=64,image_width=64):
     self.dataset_dir = dataset_dir
     self.sequence_len = sequence_len
     self.classes_list = classes_list
     self.image_height = image_height
     self.image_width = image_width


  def frames_extraction(self,video_path,normalize):
    '''
    This function will extract the required frames from a video after resizing and normalizing them.
    Args:
        video_path: The path of the video in the disk, whose frames are to be extracted.
    Returns:
        frames_list: A list containing the resized and normalized frames of the video.
    '''

    # Declare a list to store video frames.
    frames_list = []
    nm_frames_list = []
    
    # Read the Video File using the VideoCapture object.
    video_reader = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count/self.sequence_len), 1)

    # Iterate through the Video Frames.
    for frame_counter in range(self.sequence_len):

        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

        # Reading the frame from the video.
        success, frame = video_reader.read()

        # Check if Video frame is not successfully read then break the loop
        if not success:
            break

        # Resize the Frame to fixed height and width.
        resized_frame = cv2.resize(frame, (self.image_height,self.image_width))

        if normalize==True:
               normalize_frame = resized_frame/255
               nm_frames_list.append(normalize_frame)      
        else:
              frames_list.append(resized_frame)
    # Release the VideoCapture object.
    video_reader.release()
    
    # Return the frames list.
    if normalize==True:
        return nm_frames_list
    else:
        return frames_list
    
  
  def augmentation(self,data,labels,dir):

         augmented_frame = []
         augmented_labels= []
         function_list = []
         index =0
         
         sometimes = lambda aug: va.Sometimes(1,aug)
         seq1 = va.Sequential([ # randomly rotates the video with a degree randomly choosen from [-10, 10]  
         sometimes(va.HorizontalFlip()),
                      va.RandomRotate(degrees=10)])
         seq2 = va.Sequential([ # randomly rotates the video with a degree randomly choosen from [-10, 10]  
            sometimes(va.VerticalFlip()),
                      va.RandomRotate(degrees=45),])
            
                       # horizontally flip the video with 100% probability
         seq3 = va.Sequential([ # randomly rotates the video with a degree randomly choosen from [-10, 10]  
            sometimes(va.ElasticTransformation()),
                      va.RandomRotate(degrees=90),
            
                       # horizontally flip the video with 100% probability
])       
         

            
         for frames,label in zip(data,labels):
            
            #convert the label into category
            cat_label = np.argmax(label,axis=0)
            #save the first frame in the directory
          
            frame = np.array(frames)
            #augmented_frame.append(frame)
            self.save_augmented_frames(frame=frame,label=cat_label,save_dir=dir,aug_name=index)
            augmented_labels.append(label)
            index+=1
           
           
            augmented_video_frames = []
            #augment each image
            for frame in frames:  
              aug1 = seq1(frame)  
              #aug1 = np.array(aug1)
              augmented_video_frames.append(aug1)
              
            augmented_video_frames= np.array(augmented_video_frames)
            augmented_labels.append(label)
            index+=1                  
            self.save_augmented_frames(frame=augmented_video_frames,label=cat_label,save_dir=dir,aug_name=index)
            augmented_frame.append(augmented_video_frames)

            augmented_video_frames = []
            #augment each image
            for frame in frames:  
              aug1 = seq2(frame)  
              #aug1 = np.array(aug1)
              augmented_video_frames.append(aug1)
              
            augmented_video_frames= np.array(augmented_video_frames)
            augmented_labels.append(label)
            index+=1                  
            self.save_augmented_frames(frame=augmented_video_frames,label=cat_label,save_dir=dir,aug_name=index)
            augmented_frame.append(augmented_video_frames) 

            '''augmented_video_frames = []
            #augment each image
            for frame in frames:  
              aug1 = seq3(frame)  
              #aug1 = np.array(aug1)
              augmented_video_frames.append(aug1)
              
            augmented_video_frames= np.array(augmented_video_frames)
            augmented_labels.append(label)
            index+=1                  
            self.save_augmented_frames(frame=augmented_video_frames,label=label,save_dir=dir,aug_name=index)
            augmented_frame.append(augmented_video_frames)'''   
              
            #save new frames to new folder
            
            #save to a location
         augmented_labels = np.array(augmented_labels)
         #create fuction to load image from directory
         #augmented_frame = tf.convert_to_tensor(augmented_frame)
         return augmented_labels                     
  def normalize(self,data):
        #print("shape of data:",data.shape)
        frames_li = []
        for x in data:
              sample_frame=[]
              for frame in x:
                  frame=(frame/255.0).astype(np.uint8)
                  sample_frame.append(frame)
              frames_li.append(sample_frame)  
        return np.array(frames_li)          

  def my_data_generator_function(self,data):
      for features in data:
            yield (features)

  def save_augmented_frames(self,frame,label,save_dir,aug_name):
      label_dir = os.path.join(save_dir,str(label))
      os.makedirs(label_dir,exist_ok=True)
      filename =   f"{label}_{aug_name}.avi"
      filepath = os.path.join(label_dir,filename)
      print(frame.shape)
      frame_width, frame_height = frame.shape[2],frame.shape[1]
      # Define the codec and create a VideoWriter object for each frame
      fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
      out= cv2.VideoWriter(filepath,fourcc,30,(frame_width, frame_height),isColor=True)
      for img in frame:
        out.write(img)  # Write each frame to the video

      out.release()       
      
            
  
  def create_dataset(self,normalize):
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
    for class_index, class_name in enumerate(self.classes_list):

        # Display the name of the class whose data is being extracted.
        print(f'Extracting Data of Class: {class_name}')

        # Get the list of video files present in the specific class name directory.
        files_list = os.listdir(os.path.join(self.dataset_dir, class_name))

        # Iterate through all the files present in the files list.
        for file_name in files_list:

            # Get the complete video path.
            video_file_path = os.path.join(self.dataset_dir, class_name, file_name)

            #augment the data and give different path for train and test

            # Extract the frames of the video file.
            frames = self.frames_extraction(video_file_path,normalize)

            # Check if the extracted frames are equal to the SEQUENCE_LENGTH specified above.
            # So ignore the vides having frames less than the SEQUENCE_LENGTH.
            if len(frames) == self.sequence_len:

                # Append the data to their repective lists.
                features.append(frames)
                labels.append(class_index)
                video_files_paths.append(video_file_path)

    
    labels = np.array(labels)
    one_hot_encoded_labels = to_categorical(labels)
    features_train, features_test, labels_train, labels_test = train_test_split(features, one_hot_encoded_labels, test_size = 0.25, shuffle = True, random_state = seed_constant)
    # Return the frames, class index, and video file path.
    #accessing the normalize data for training data
    if normalize==True:
          features=np.array(features)
          reshaped_features = features.reshape(-1,20,64,64,3)
          return reshaped_features
    else:    
        return features_train, features_test, labels_train, labels_test