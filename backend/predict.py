from keras.models import load_model
import pickle
import cv2

def load_predict_model(desiase):
    # загружаем модель и бинаризатор меток
    print("[INFO] loading network and label binarizer...")
    model = load_model('output/' + desiase + '/simple_nn.model')
    return model


def predict(image, model, desiase):
    lb = pickle.loads(open('output/' + desiase + '/simple_nn_lb.pickle', "rb").read())
    # делаем предсказание на изображении
    preds = model.predict(image)
    print(preds)

    # находим индекс метки класса с наибольшей вероятностью
    # соответствия
    i = preds.argmax(axis=1)[0]
    label = lb.classes_[i]
    text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
    return label, text



def main(image_path, desiase, width=32, height=32, flatten=1):
    # загружаем входное изображение и меняем его размер на необходимый
    image = cv2.imread(image_path)
    output = image.copy()
    image = cv2.resize(image, (width, height))

    # масштабируем значения пикселей к диапазону [0, 1]
    image = image.astype("float") / 255.0

    # проверяем, необходимо ли сгладить изображение и добавить размер
    # пакета
    if flatten > 0:
        image = image.flatten()
        image = image.reshape((1, image.shape[0]))

    # в противном случае мы работаем с CNN -- не сглаживаем изображение
    # и просто добавляем размер пакета
    else:
        image = image.reshape((1, image.shape[0], image.shape[1],
                            image.shape[2]))

    model = load_predict_model(desiase)
    label, text = predict(image, model, desiase)
    return text
