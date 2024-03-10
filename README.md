## Цель проекта

Создание автоматическое системы оценки техники катания хоккеистов. Система включает в себя аппаратно-программный комплекс оценки и сервисную платформу для создания комьюнити пользователей.

## Проверка гипотезы (хоть как то работает)

На данной стадии необходимо убедиться в принципиальной возможности формализации требований к технике катания в терминах математических алгоритмов.

Для упрощения задачи на первом этапе будут анализироваться несколько базовых параметров техники катания, а для упрощения процедуры сбора данных будет использоваться беговая дорожка.

В качестве входных данных будет использоваться видеозапись катания на беговой дорожке с фронтального и бокового ракурсов.

Будет создана тестовая выборка записей на которых будет проверяться работоспособность алгоритма. На данный момент непонятно сколько записей удастся собрать поэтому пока количественных критериев нет.

Неформально критерий успеха первого этапа можно сформулировать так: нужно чтобы у человека смотрящего на работу системы было ощущение что она скорей работает, чем не работает.

На первом этапе нет требований к производительности алгоритма или качеству входного видео.

Другие упрощения:

* Можно использовать видео со спортсменами в обтягивающей одежде (не в хоккейной форме)
* Можно наклеивать метки на ключевые точки спортсмена.
* Максимальное использование уже существующих программных решений.



Примеры видео: <https://disk.yandex.ru/a/f2qYBEjeIvBkRw>


## ОЦенка моделей Pose_Estimation

* Move_Net - плохая модель. теряет человека и переключается на других людей

* 

===========================ilya==============================

## Установка (Windows)

- python -m venv venv

- venv\Scripts\python -m pip install -r requirements.txt

- venv\Scripts\python -m pip install libs\ultralytics-8.1.11-py3-none-any.whl

## Добавление фукнциональности
- детекция фазы движения ног (начало толчка, толчок, окончание толчка)

На данном этапе использована YOLOv8 pose detector, в оригинальную библиотеку внесены изменения для удобства работы,
она ставится из изменённого архива в папке libs

- preprocessing.py содержит методы предподготовки изображений и видео

- cluster_classific.py содержит методы кластеризации и классификации исходя из суммарной матрицы

- predictor.py предсказывает фазу движения(ещё не реализован)
 

