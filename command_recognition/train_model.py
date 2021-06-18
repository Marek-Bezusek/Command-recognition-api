import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout,  Dense, Flatten, Reshape
import tensorflow as tf
import matplotlib.pyplot as plt
from time import time
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from command_recognition.build_features import DATASET_PATH, THIS_DIR

MODEL_PATH = os.path.join(THIS_DIR, "model.h5")


def load_dataset(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])
    return X, y


def prepare_dataset(dataset_path, test_size=0.2, validation_size=0.2):
    """Creates train, validation and test sets.
    :param dataset_path (str): Path to json file containing data
    :param test_size (flaot): Percentage of dataset used for testing
    :param validation_size (float): Percentage of train set used for cross-validation
    """
    # Load dataset
    X, y = load_dataset(dataset_path)

    # Split dataset: train, validation, test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    return X_train, y_train, X_validation, y_validation, X_test, y_test


def create_model(input_shape=(44, 13), learning_rate=0.0001, dropout_rate=0.3):
    # Create model
    model = Sequential()

    # Reshape input data for CNN: add one more dimension for convolution
    model.add(Reshape(input_shape + (1,), input_shape=input_shape))

    # 1st conv layer
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape,
              kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

    # 2nd conv layer
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

    # 3rd conv layer
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

    # Dense layer; classifier
    model.add(Flatten())  # flatten output; to vector
    model.add(Dense(64, activation='relu'))
    Dropout(dropout_rate)

    # Softmax output layer; get probability distribution
    model.add(Dense(10, activation='softmax'))

    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    # Compile model
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs
        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # accuracy subplot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="validation accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # loss subplot
    axs[1].plot(history.history["loss"], label="train loss")
    axs[1].plot(history.history["val_loss"], label="validation loss")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss eval")

    plt.show()


def train(dataset_path, batch_size=8, epochs=40):
    """Train model"""
    # load dataset
    X_train, y_train, X_validation, y_validation, X_test, y_test = prepare_dataset(dataset_path, test_size=0.2, validation_size=0.2)
    input_shape = (X_train.shape[1], X_train.shape[2])

    start = time()  # time training
    # create network
    model = create_model(input_shape)

    # train network
    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor="accuracy", min_delta=0.001, patience=5)
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation),
                        batch_size=batch_size, epochs=epochs, callbacks=[earlystop_callback])

    # plot accuracy/loss for training/validation set as a function of the epochs
    plot_history(history)

    # evaluate network on test set
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("\nTest loss: {}, test accuracy: {}".format(test_loss, 100 * test_acc))
    print('Time = {:.0f} sec'.format(time()-start))

    # save model
    model.save(MODEL_PATH)
    return model

def display_gs_results(search_results):
    """Display Grid Search result"""

    print('Best score = {:.4f} using {}'.format(search_results.best_score_, search_results.best_params_))
    means = search_results.cv_results_['mean_test_score']
    stds = search_results.cv_results_['std_test_score']
    params = search_results.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print('mean test accuracy +/- std = {:.4f} +/- {:.4f} with: {}'.format(mean, stdev, param))


def grid_search(dataset_path):
    """Search grid of hyperparameters. Return set of hyperparameters with highest accuracy"""

    # Load dataset
    X_train, y_train, X_validation, y_validation, X_test, y_test = prepare_dataset(dataset_path, test_size=0.2, validation_size=0.2)

    #%% Define hyperparameters for grid search
    batch_size = [4, 8, 16, 32, 64, 128]
    epochs = [10, 50, 100]

    learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
    # optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    # momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]

    # Tune Weight Initialization
    init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']

    # Regularization parameters
    dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    weight_constraint = [1, 2, 3, 4, 5]
    regularization = [0.001, 0.01]

    # Select hyperparameters for search
    param_grid = dict(batch_size=batch_size, epochs=epochs)

    #%% Time GridSearchCV
    start = time()

    #%% Perform Grid Search crossvalidation
    model = KerasClassifier(build_fn=create_model, verbose=1)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(X_train, y_train, validation_data=(X_validation, y_validation))

    # print and plot results
    print('Time for grid search = {:.0f} sec'.format(time()-start))
    display_gs_results(grid_result)

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def model_test(model_path, dataset_path):
    # Load model
    model = tf.keras.models.load_model(model_path)
    # Load dataset
    X_test, y_test = prepare_dataset(dataset_path, test_size=0.2, validation_size=0.2)[4:6]

    # Predict
    Y_pred = model.predict(X_test)
    y_pred = np.argmax(Y_pred, axis=1)
    labels = ["down", "off", "on", "no", "yes", "stop", "up", "right", "left", "go"]
    # Print/plot
    plot_confusion_matrix(cm=confusion_matrix(y_test, y_pred), classes=labels)
    print('Confusion Matrix')
    print(confusion_matrix(y_test, y_pred))
    print('Classification Report')
    print(classification_report(y_test, y_pred, target_names=labels))


if __name__ == "__main__":
    #%% Train model and plot validation results
    train(DATASET_PATH)
    #%% Test model and plot Confusion Matrix and Classification Report
    model_test(MODEL_PATH, DATASET_PATH)
