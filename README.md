Описание проекта
- Это программа для обработки данных, визуализации и анализа двумерных полей с использованием:
- Гауссовых распределений
- Алгоритмов кластеризации (k-means)
- Триангуляции Делоне
- Поиска путей с учетом препятствий
- Генерации BMP изображений
- Программа реализована на C++ и использует конфигурационные файлы для настройки параметров.

Основные возможности

Генерация полей:
- Наложение Гауссовых распределений
- Нормализация данных
- 
Анализ данных:
- Бинаризация полей (срезы по высоте)
- Выделение компонент связности (алгоритм "волна")
- Кластеризация методом k-means
- 
Визуализация:
- Экспорт данных в формате для GNUplot
- Генерация BMP изображений (черно-белых и цветных)
- Отрисовка триангуляции Делоне
- Визуализация путей между точками
- 
Логирование:
- Серверные логи (Control)
- Клиентские логи (Interface)

Требования:
- Компилятор C++ с поддержкой C++11 (g++ или clang++)
- Linux или Windows (требуется адаптация для Linux)
 -Для визуализации через GNUplot требуется установленный GNUplot

Сборка и запуск:
- Скомпилируйте программу
- Подготовьте конфигурационный файл
- Запустите программу

Пример конфигурационного файла(см. config.txt)
Пример командного файла(см. Batch_file.txt)

Структура.
Основные классы:
Field - представление двумерного поля данных
Gauss - Гауссово распределение
Srez - операции с бинаризованными срезами
Control - управляющий модуль (логика)
Interface - пользовательский интерфейс
Server/Client - система логирования

Выходные данные.
Программа генерирует:
- BMP файлы с визуализацией (field.bmp, clusters.bmp, road.bmp и др.)
- Файлы данных для GNUplot
- Лог-файлы (server.log, client.log)

Для более полного ознакомления с возможностью программы см. Методичка.pdf
