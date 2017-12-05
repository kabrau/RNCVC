'''Train a simple deep CNN on the CIFAR10 small images dataset.

It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import dataset as ds

def run(netName, data_augmentation, activation, optimizer):

    model_name = netName+"_"+activation+"_"+optimizer
    if data_augmentation:
        model_name = model_name+"_aug"
    else:
        model_name = model_name+"_notAug"

    print('NET=',model_name)

    save_dir = os.path.join(os.getcwd(), 'saved_models')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    x_train, x_valid, x_test, y_train, y_valid, y_test, num_classes = ds.load_data("E:/Datasets/pedestrian_signal/classification")

    batch_size = 32
    epochs = 50

    if netName=="rede1":
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same',
                        input_shape=x_train.shape[1:]))
        model.add(Activation(activation))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation(activation))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation(activation))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        # initiate RMSprop optimizer
        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    elif netName=="rede2":
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same',
                        input_shape=x_train.shape[1:]))
        model.add(Activation(activation))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation(activation))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

    if optimizer=='rms':
        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    elif optimizer=='adagrad':
        opt = keras.optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)

    elif optimizer=='adadelta':
        opt = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)

    elif optimizer=='adamax':
        opt = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)


    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])

    print(model.summary()) #Marcelo
 
    with open(os.path.join(save_dir,model_name+'-summary.txt'),'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    keras.utils.plot_model(model, to_file=os.path.join(save_dir,model_name+'.png'))

    # Marcelo
    callbacks = []
    callbacks.append(
    keras.callbacks.TensorBoard(log_dir='./logs/'+model_name, 
                                histogram_freq=0, #1
                                batch_size=batch_size, 
                                write_graph=True, 
                                write_grads=False, 
                                write_images=False, 
                                embeddings_freq=0, 
                                embeddings_layer_names=None, 
                                embeddings_metadata=None)
    )

    fileBestWeights = filepath=os.path.join(save_dir,'%s-best.h5' % model_name)
    callbacks.append(
        keras.callbacks.ModelCheckpoint(fileBestWeights,
                                        monitor='val_loss', 
                                        verbose=0, 
                                        save_best_only=True, 
                                        save_weights_only=True, 
                                        mode='auto', 
                                        period=1)
    )

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_valid, y_valid),
                shuffle=True,
                callbacks = callbacks)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train,
                                        batch_size=batch_size),
                            epochs=epochs,
                            steps_per_epoch=len(x_train)/batch_size,
                            validation_data=(x_valid, y_valid),
                            workers=1,
                            callbacks = callbacks)

    # Save model and weights
    model_path = os.path.join(save_dir, model_name+"-last.h5")
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    # Score trained model.
    model.load_weights(fileBestWeights)
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    with open(os.path.join(save_dir,model_name+'-evaluate.txt'),'w') as fh:
        fh.write(model_name+';'+str(scores[0]) +';'+str(scores[1])+ '\n')

if __name__ == "__main__":

    for netName in ['rede1','rede2']:
        for data_augmentation in [True, False]:
            for activation in ['relu', 'tanh']:
                for optimizer in ['rms','adagrad','adadelta','adamax']:
                    run(netName, data_augmentation, activation, optimizer)
                    #print(netName, data_augmentation, activation, optimizer)




