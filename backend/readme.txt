Перед запуском программы:
python -m pip install -r requirements.txt

Для обучения нейронной сети:
python main.py --dataset archive/test --model output/simple_nn.model --label-bin output/simple_nn_lb.pickle --plot output/simple_nn_plot.png

Для прогнозирования снимка:
python predict.py --image archive/IMAGES/P.jpg --model output/simple_nn.model --label-bin output/simple_nn_lb.pickle --width 32 --height 32 --flatten 1