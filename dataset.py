import os
import cv2
import numpy as np
import keras
from sklearn.model_selection import train_test_split

def get_best_size(data_dir):
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]

    print("Folders: ",directories)
    images_sizes = []


    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir)
                      if f.endswith(".jpg") or f.endswith(".jpeg")]

        for f in file_names:
            img = cv2.imread(f)
            images_sizes.append([img.shape[0], img.shape[1]])

    print("Total Images: ",len(images_sizes))
    print("Mean dimension:", np.mean(images_sizes,axis=0).round())


# Load the data
def load_data(data_dir, h=80, w=48):
    
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []

    category = 0
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir)
                      if f.endswith(".jpg") or f.endswith(".jpeg")]
        
        for f in file_names:
            img = cv2.imread(f)
            imresize = cv2.resize(img, (h, w))
            images.append(imresize)
            labels.append(category)
            
        category += 1

    # 70% train
    X_train, X_tmp, y_train, y_tmp = train_test_split( images, labels, test_size=0.3, random_state=100) 
    # 15% valid and 15% test
    X_valid, X_test, y_valid, y_test = train_test_split( X_tmp, y_tmp, test_size=0.5, random_state=100) 

    # normalize inputs from 0-255 and 0.0-1.0
    X_train = np.array(X_train).astype('float32')
    X_valid = np.array(X_valid).astype('float32')
    X_test = np.array(X_test).astype('float32')
    X_train = X_train / 255.0
    X_valid = X_valid / 255.0
    X_test = X_test / 255.0

    # one hot encode outputs
    y_train = np.array(y_train)
    y_valid = np.array(y_valid)
    y_test = np.array(y_test)
    y_train = keras.utils.to_categorical(y_train)
    y_valid = keras.utils.to_categorical(y_valid)
    y_test = keras.utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    return X_train, X_valid, X_test, y_train, y_valid, y_test, num_classes




if __name__ == "__main__":
    # get_best_size("C:/Users/Cabral/Downloads/pedestrian_signal_classification/classification")
    X_train, X_valid, X_test, y_train, y_valid, y_test, num_classes = load_data("C:/Users/marcelo/Downloads/pedestrian_signal_classification/classification")

    print(X_train.shape)
    print(X_valid.shape)
    print(X_test.shape)
    print(len(y_train))
    print(len(y_valid))
    print(len(y_test))
    print(num_classes)
