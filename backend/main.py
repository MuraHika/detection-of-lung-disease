import matplotlib

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
# from keras.constraints import maxnorm
# from keras.layers.convolutional import Conv2D, MaxPooling2D
# from keras.utils import np_utils
# from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os
# подключаем необходимые пакеты
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.layers.core import Dense
# from keras.optimizers import SGD
from imutils import paths

matplotlib.use("Agg")
lb = LabelBinarizer()


def main():
    # fix random seed for reproducibility
    seed = 21
    np.random.seed(seed)

    ############################

    # создаём парсер аргументов и передаём их
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
                    help="path to input dataset of images")
    ap.add_argument("-m", "--model", required=True,
                    help="path to output trained model")
    ap.add_argument("-l", "--label-bin", required=True,
                    help="path to output label binarizer")
    ap.add_argument("-p", "--plot", required=True,
                    help="path to output accuracy/loss plot")
    args = vars(ap.parse_args())
    # инициализируем данные и метки
    print("[INFO] loading images...")
    normal_dataset_path = list(paths.list_images('archive_images/NORMAL'))
    
    disease_dataset_path = list(paths.list_images(args["dataset"]))

    # берём пути к изображениям и рандомно перемешиваем
    imagePaths = sorted(normal_dataset_path + disease_dataset_path)
    random.seed(42)
    random.shuffle(imagePaths)
    data, labels = image_processing(imagePaths)
    print(labels)
    X_train, X_test, y_train, y_test = split_data(data, labels)
    print(len(X_train))
    print(len(y_train))
    model = create_architecture_neural_network()
    train_and_save_neural_network(model, X_train, X_test, y_train, y_test, args)

def image_processing(imagePaths):
    data = []
    labels = []
    # цикл по изображениям
    for imagePath in imagePaths:
        # загружаем изображение, меняем размер на 32x32 пикселей (без учета
        # соотношения сторон), сглаживаем его в 32x32x3=3072 пикселей и
        # добавляем в список
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (32, 32)).flatten()
        data.append(image)

        # извлекаем метку класса из пути к изображению и обновляем
        # список меток
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)

    # масштабируем интенсивности пикселей в диапазон [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    return data, labels

def split_data(data, labels):
    # разбиваем данные на обучающую и тестовую выборки, используя 75%
    # данных для обучения и оставшиеся 25% для тестирования
    (X_train, X_test, y_train, y_test) = train_test_split(data,
                                                    labels, test_size=0.25, random_state=42)

    # конвертируем метки из целых чисел в векторы (для 2х классов при
    # бинарной классификации вам следует использовать функцию Keras
    # “to_categorical” вместо “LabelBinarizer” из scikit-learn, которая
    # не возвращает вектор)
    y_train = lb.fit_transform(y_train)
    y_test = lb.transform(y_test)
   
    return X_train, X_test, y_train, y_test

def create_architecture_neural_network():
    print("[INFO] creating model...")


    # определим архитектуру 3072-1024-512-4 с помощью Keras
    model = Sequential()
    model.add(Dense(1024, input_shape=(3072,), activation="sigmoid"))
    model.add(Dense(512, activation="sigmoid"))
    model.add(Dense(len(lb.classes_), activation="softmax"))

    optimizer = 'Adam'

    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    print(model.summary())
    return model

def train_and_save_neural_network(model, X_train, X_test, y_train, y_test, args):
    print("[INFO] training network...")
    epochs = 40
    H = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64)

    # Final evaluation of the model

    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))


    # строим графики потерь и точности
    N = np.arange(0, epochs)
    plt.style.use("ggplot")
    plt.figure()
    print(H.history)
    plt.plot(N, H.history["loss"], label="train_loss")
    plt.plot(N, H.history["val_loss"], label="val_loss")
    plt.plot(N, H.history["accuracy"], label="train_acc")
    plt.plot(N, H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy (Simple NN)")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(args["plot"])

    # сохраняем модель и бинаризатор меток на диск
    print("[INFO] serializing network and label binarizer...")
    model.save(args["model"])
    f = open(args["label_bin"], "wb")
    f.write(pickle.dumps(lb))
    f.close()


if __name__ == '__main__':
    main()
# Create the model
# model = Sequential()
#
# model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:], padding='same'))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())
#
# model.add(Conv2D(64, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())
#
# model.add(Conv2D(64, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())
#
# model.add(Conv2D(128, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())
#
# model.add(Flatten())
# model.add(Dropout(0.2))
#
# model.add(Dense(256, kernel_constraint=maxnorm(3)))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())
# model.add(Dense(128, kernel_constraint=maxnorm(3)))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())
# model.add(Dense(num_classes))
# model.add(Activation('softmax'))