# detection-of-lung-disease
| Для обучения нейронных сетей |

Перейти в папку backend:
```
cd backend
```
Создать окружение:
```
python3 -m venv venv
```
Установить необходимые библиотеки:
```
python3 -m pip install -r requirements.txt
```
Запустить каждую команду по отдельности:
```
python3 main.py --dataset archive_images/COVID19 --model output/covid/simple_nn.model --label-bin output/covid/simple_nn_lb.pickle --plot output/covid/simple_nn_plot.png
```
```
python3 main.py --dataset archive_images/PNEUMONIA --model output/pneumonia/simple_nn.model --label-bin output/pneumonia/simple_nn_lb.pickle --plot output/pneumonia/simple_nn_plot.png
```
```
python3 main.py --dataset archive_images/TUBERCULOSIS --model output/tuberculosis/simple_nn.model --label-bin output/tuberculosis/simple_nn_lb.pickle --plot output/tuberculosis/simple_nn_plot.png
```

| Для запуска системы |

Перейти в папку backend:
```
cd backend
```
Создать окружение:
```
python3 -m venv venv
```
Установить необходимые библиотеки:
```
python3 -m pip install -r requirements.txt
```
Запустить сервер:
```
python3 index.py
```

Перейти в папку frontend:
```
cd frontend
```
Установить необходимые библиотеки:
```
npm i
```
Запустить клиент:
```
npm run serve
```