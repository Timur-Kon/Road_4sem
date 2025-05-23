//212-Konchugarov-Timur
#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES // Для Visual Studio
#include <cmath>
#include <iostream>
#include <tuple>
#include <queue>    // Для std::queue
#include <utility>  // Для std::pair
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <sstream>
#include <cstdint>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <cstdlib>
#include <limits>
#include <algorithm>
#include <random>
#include <set>
#define EPS 0.00001

struct Point { // структура для хранения координат точки
    int x, y;
};

struct Color { // структура для хранения rgb цвета
    uint8_t r, g, b;
};

// Структура для точки с вещественными координатами (для точности)
struct PointD {
    double x, y;
    PointD(double x_ = 0, double y_ = 0) : x(x_), y(y_) {}

    // Оператор сравнения для упорядочивания
    bool operator<(const PointD& other) const {
        if (x < other.x) return true;
        if (x > other.x) return false;
        return y < other.y;
    }

    // Оператор равенства
    bool operator==(const PointD& other) const {
        return std::abs(x - other.x) < 1e-6 &&
            std::abs(y - other.y) < 1e-6;
    }
};

struct Triangle {
    PointD a, b, c;
    Triangle(PointD a_, PointD b_, PointD c_) : a(a_), b(b_), c(c_) {}

    // Добавляем оператор сравнения
    bool operator==(const Triangle& other) const {
        return (a == other.a && b == other.b && c == other.c) ||
            (a == other.a && b == other.c && c == other.b) ||
            (a == other.b && b == other.a && c == other.c) ||
            (a == other.b && b == other.c && c == other.a) ||
            (a == other.c && b == other.a && c == other.b) ||
            (a == other.c && b == other.b && c == other.a);
    }
};

struct Edge {
    PointD a, b;
    Edge(PointD a_, PointD b_) : a(a_), b(b_) {}

    // Оператор сравнения на равенство
    bool operator==(const Edge& other) const {
        return (a == other.a && b == other.b) ||
            (a == other.b && b == other.a);
    }

    // Оператор сравнения для упорядочивания (если нужно для std::set)
    bool operator<(const Edge& other) const {
        return std::tie(a, b) < std::tie(other.a, other.b);
    }
};

//212-Konchugarov-Timur

class Server { //Класс, который отвечает за логирование сервера (Contrl)
    std::ofstream server_log;  // Серверный лог
    bool logging_enabled = false; // Нужно для конфига, чтобы включать или выключать логи
    std::string log_filename = "server_log.txt"; // Имя файла лога (по умолчанию)
public:
    Server() {
    }

    ~Server() {
        if (server_log.is_open()) {
            server_log.close();
        }
    }

    void enable_logging(const std::string& filename) { // Функция для включения логирования сервера
        logging_enabled = true;
        log_filename = filename;
        if (server_log.is_open()) {
            server_log.close();
        }
        server_log.open(log_filename, std::ios::app);
    }

    void disable_logging() { // Функция для отключения логирования сервера
        logging_enabled = false;
        if (server_log.is_open()) {
            server_log.close();
        }
    }

    void log(const std::string& message) { // Функция-шаблон лога
        if (logging_enabled && server_log.is_open()) {
            server_log << current_time() << " - " << message << std::endl;
        }
    }

    std::string current_time() { // Как будет отображаться время в логах
        std::time_t t = std::time(nullptr);
        std::tm* now = std::localtime(&t);
        std::ostringstream oss;
        oss << std::put_time(now, "%d.%m.%Y %H:%M:%S");
        return oss.str();
    }
};

//212-Konchugarov-Timur

class Client { // Класс, овтечающий за логирвоание клиента (Interface)
    std::ofstream client_log;  // Клиентский лог
    bool logging_enabled = false; // //Нужно для конфига, чтобы включать или выключать логи
    std::string log_filename = "client_log.txt"; // Имя файла лога (по умолчанию)

public:
    Client() {
    }

    ~Client() {
        if (client_log.is_open()) {
            client_log.close();
        }
    }

    void enable_logging(const std::string& filename) { // Функция для включения логирования клиента
        logging_enabled = true;
        log_filename = filename;
        if (client_log.is_open()) {
            client_log.close();
        }
        client_log.open(log_filename, std::ios::app);
    }

    void disable_logging() { // Функция для выключения логирования клиента
        logging_enabled = false;
        if (client_log.is_open()) {
            client_log.close();
        }
    }

    void log(const std::string& message) { // Функция-шаблон лога
        if (logging_enabled && client_log.is_open()) {
            client_log << current_time() << " - " << message << std::endl;
        }
    }

    std::string current_time() { // Как будет отоборажаться время в логах
        std::time_t t = std::time(nullptr);
        std::tm* now = std::localtime(&t);
        std::ostringstream oss;
        oss << std::put_time(now, "%d.%m.%Y %H:%M:%S");
        return oss.str();
    }
};

//212-Konchugarov-Timur

class Component { // Класс, хранящий все компоненты в нужном нам срезе
public:
    std::vector<std::vector<double>> comp;
    int min_x, min_y, max_x, max_y;
    double center_x, center_y;
    double eigenvec1_x, eigenvec1_y;
    double eigenvec2_x, eigenvec2_y;
    double eigenvalue1, eigenvalue2;

    Component(const std::vector<std::vector<double>>& inpcomp) : comp(inpcomp) {}

    Component(int A, int B) {
        comp.resize(A, std::vector<double>(B, 255));
    }

    void calculate_metadata() {
        // Находим границы и центр
        min_x = std::numeric_limits<int>::max();
        min_y = std::numeric_limits<int>::max();
        max_x = std::numeric_limits<int>::min();
        max_y = std::numeric_limits<int>::min();
        double sum_x = 0, sum_y = 0;
        int count = 0;

        for (int i = 0; i < int(comp.size()); ++i) {
            for (int j = 0; j < int(comp[0].size()); ++j) {
                if (abs(comp[i][j]) < EPS) { // Точка принадлежит компоненте
                    min_x = std::min(min_x, i);
                    min_y = std::min(min_y, j);
                    max_x = std::max(max_x, i);
                    max_y = std::max(max_y, j);
                    sum_x += i;
                    sum_y += j;
                    count++;
                }
            }
        }
        center_x = sum_x / count;
        center_y = sum_y / count;
        // Расчет ковариационной матрицы
        double cov_xx = 0, cov_xy = 0, cov_yy = 0;
        for (int i = min_x; i <= max_x; ++i) {
            for (int j = min_y; j <= max_y; ++j) {
                if (abs(comp[i][j]) < EPS) {
                    double dx = i - center_x;
                    double dy = j - center_y;
                    cov_xx += dx * dx;
                    cov_xy += dx * dy;
                    cov_yy += dy * dy;
                }
            }
        }
        cov_xx /= count;
        cov_xy /= count;
        cov_yy /= count;

        // Собственные значения и векторы
        double trace = cov_xx + cov_yy;
        double det = cov_xx * cov_yy - cov_xy * cov_xy;
        eigenvalue1 = (trace + sqrt(trace * trace - 4 * det)) / 2;
        eigenvalue2 = (trace - sqrt(trace * trace - 4 * det)) / 2;

        // Собственные векторы (упрощенный расчет)
        if (abs(cov_xy) > EPS) {
            eigenvec1_x = eigenvalue1 - cov_yy;
            eigenvec1_y = cov_xy;
            eigenvec2_x = eigenvalue2 - cov_yy;
            eigenvec2_y = cov_xy;
        }
        else {
            eigenvec1_x = 1; eigenvec1_y = 0;
            eigenvec2_x = 0; eigenvec2_y = 1;
        }

    }

};


class Gauss { // Класс, хранящий Гауссы
public:
    double h, x0, y0, sigma_x, sigma_y;
    Gauss(double h, double x0, double y0, double sigma_x, double sigma_y)
        : h(h), x0(x0), y0(y0), sigma_x(sigma_x), sigma_y(sigma_y) {}

    double calculate(int x, int y) const { //Функция вычисления Гаусса
        double dx = x - x0;
        double dy = y - y0;
        double denom_x = 2 * sigma_x * sigma_x;
        double denom_y = 2 * sigma_y * sigma_y;

        if ((int)denom_x == 0 || (int)denom_y == 0) {
            std::cerr << "Error: sigma_x or sigma_y iz zero!" << std::endl;
            return 0;
        }

        double exponent = -((dx * dx) / denom_x +
            (dy * dy) / denom_y);

        if (exponent < -700) {
            return 0;
        }

        else if (exponent > 700) {
            return std::numeric_limits<double>::infinity();
        }

        return h * exp(exponent);
    }
};

//212-Konchugarov-Timur

class Field { //Класс поле
public:
    int length, width;
    std::vector<std::vector<double>> matrix;
    std::vector<std::vector<double>> weight_matrix; // Матрица сумм весов
    std::vector<std::vector<Color>> colors; //Матрица цветов для bmp (если нужно)

    //Все, что нужно для фактор-векторов
    double center_x, center_y;
    double eigenvec1_x, eigenvec1_y;
    double eigenvec2_x, eigenvec2_y;
    double eigenvalue1, eigenvalue2;

    Field(int l, int w) : length(l), width(w) {
        matrix.resize(length, std::vector<double>(width, 0));
        weight_matrix.resize(length, std::vector<double>(width, 0));
        colors.resize(length, std::vector<Color>(width, { 0, 0, 0 }));
    }

    void apply_gauss(const Gauss& g) { //Добавляет гаусс на поле

        double value;

        for (int i = 0; i < length; ++i) {
            for (int j = 0; j < width; ++j) {
                value = g.calculate(i, j);
                matrix[i][j] += value * g.h; // Взвешенное добавление значения
                weight_matrix[i][j] += g.h; // Вес в точке (на основе амплитуд)
            }
        }
    }

    void normalize(void) {
        for (int i = 0; i < length; ++i) {
            for (int j = 0; j < width; ++j) {
                if (weight_matrix[i][j] > 0) {
                    matrix[i][j] /= weight_matrix[i][j];
                }
            }
        }
    }
    /*
    // В класс Field добавить:
    void drawTriangles(const std::vector<std::tuple<Point, Point, Point>>& triangles) {
        Color lineColor = { 0, 255, 0 }; // Зеленый цвет для линий
        for (const auto& triangle : triangles) {
            Point p1 = std::get<0>(triangle);
            Point p2 = std::get<1>(triangle);
            Point p3 = std::get<2>(triangle);
            draw_line(p1.x, p1.y, p2.x, p2.y, lineColor);
            draw_line(p2.x, p2.y, p3.x, p3.y, lineColor);
            draw_line(p3.x, p3.y, p1.x, p1.y, lineColor);
        }
    }*/


    void drawTriangles(const std::vector<Triangle>& triangles) {
        Color lineColor = { 0, 255, 0 };
        for (const auto& tri : triangles) {
            draw_line(tri.a.x, tri.a.y, tri.b.x, tri.b.y, lineColor);
            draw_line(tri.b.x, tri.b.y, tri.c.x, tri.c.y, lineColor);
            draw_line(tri.c.x, tri.c.y, tri.a.x, tri.a.y, lineColor);
        }
    }


    void save_to_gnuplot(const std::string& filename) { // Функция сохранения гнуплота
        std::ofstream bmp_file(filename);
        for (int i = 0; i < length; ++i) {
            for (int j = 0; j < width; ++j) {
                bmp_file << i << " " << j << " " << matrix[i][j] << std::endl;
            }
            bmp_file << std::endl;
        }
        bmp_file.close();
    }

    void bmp_write(const std::string& filename, int k) { // Запись BMP
        const int BMP_HEADER_SIZE = 54;
        const int PIXEL_SIZE = 3;
        int file_size = BMP_HEADER_SIZE + PIXEL_SIZE * length * width;

        unsigned char bmp_header[BMP_HEADER_SIZE] = { 0 };

        // Заголовок BMP файла
        bmp_header[0] = 'B';
        bmp_header[1] = 'M';
        bmp_header[2] = file_size & 0xFF;
        bmp_header[3] = (file_size >> 8) & 0xFF;
        bmp_header[4] = (file_size >> 16) & 0xFF;
        bmp_header[5] = (file_size >> 24) & 0xFF;
        bmp_header[10] = BMP_HEADER_SIZE;

        // Заголовок DIB
        bmp_header[14] = 40; // Размер заголовка DIB
        bmp_header[18] = width & 0xFF;
        bmp_header[19] = (width >> 8) & 0xFF;
        bmp_header[20] = (width >> 16) & 0xFF;
        bmp_header[21] = (width >> 24) & 0xFF;
        bmp_header[22] = length & 0xFF;
        bmp_header[23] = (length >> 8) & 0xFF;
        bmp_header[24] = (length >> 16) & 0xFF;
        bmp_header[25] = (length >> 24) & 0xFF;
        bmp_header[26] = 1; // Число цветовых плоскостей
        bmp_header[28] = 24; // Количество бит на пиксель

        std::ofstream bmp_file(filename, std::ios::binary);
        bmp_file.write(reinterpret_cast<char*>(bmp_header), BMP_HEADER_SIZE);

        // Записываем пиксели (матрицу)
        if (k == 1) { //Для ч\б картинки компоненты

            // Первый вектор (красный)
            draw_line(center_x, center_y,
                center_x + eigenvec1_x * 100,
                center_y + eigenvec1_y * 100,
                { 255, 0, 0 });

            // Второй вектор (синий)
            draw_line(center_x, center_y,
                center_x + eigenvec2_x * 100,
                center_y + eigenvec2_y * 100,
                { 0, 0, 255 });

            for (int i = length - 1; i >= 0; --i) {
                for (int j = 0; j < width; ++j) {

                    if (colors[i][j].r != 0 || colors[i][j].g != 0 || colors[i][j].b != 0) {
                        unsigned char pixel[3] = { static_cast<unsigned char>(colors[i][j].r),
                        static_cast<unsigned char>(colors[i][j].g),
                        static_cast<unsigned char>(colors[i][j].b) };
                        bmp_file.write(reinterpret_cast<char*>(pixel), PIXEL_SIZE);
                    }

                    else {

                        int value = static_cast<int>(matrix[i][j] * 100); // Умножаю на коэффициент 100, чтобы отображалось красиво и ярко
                        unsigned char pixel[3] = { static_cast<unsigned char>(std::min(std::max(value, 0), 255)),
                                                static_cast<unsigned char>(std::min(std::max(value, 0), 255)),
                                                static_cast<unsigned char>(std::min(std::max(value, 0), 255)) };
                        bmp_file.write(reinterpret_cast<char*>(pixel), PIXEL_SIZE);
                    }

                }
            }
        }

        if (k == 2) { //Для ч\б картинки
            for (int i = length - 1; i >= 0; --i) {
                for (int j = 0; j < width; ++j) {

                    int value = static_cast<int>(matrix[i][j] * 100); // Умножаю на коэффициент 100, чтобы отображалось красиво и ярко
                    unsigned char pixel[3] = { static_cast<unsigned char>(std::min(std::max(value, 0), 255)),
                                            static_cast<unsigned char>(std::min(std::max(value, 0), 255)),
                                            static_cast<unsigned char>(std::min(std::max(value, 0), 255)) };
                    bmp_file.write(reinterpret_cast<char*>(pixel), PIXEL_SIZE);

                }
            }


        }

        else { //для разукраски класстеров
            for (int i = length - 1; i >= 0; --i) {
                for (int j = 0; j < width; ++j) {
                    unsigned char pixel[3] = { static_cast<unsigned char>(colors[i][j].r),
                                               static_cast<unsigned char>(colors[i][j].g),
                                               static_cast<unsigned char>(colors[i][j].b) };
                    bmp_file.write(reinterpret_cast<char*>(pixel), PIXEL_SIZE);
                }
            }
        }

        bmp_file.close();
    }

    void draw_line(int x1, int y1, int x2, int y2, Color color) {
        int dx = abs(x2 - x1);
        int dy = abs(y2 - y1);
        int sx = (x1 < x2) ? 1 : -1;
        int sy = (y1 < y2) ? 1 : -1;
        int err = dx - dy;

        while (true) {
            if (x1 >= 0 && x1 < length && y1 >= 0 && y1 < width) {
                colors[x1][y1] = color;
            }

            if (x1 == x2 && y1 == y2) break;
            int e2 = 2 * err;
            if (e2 > -dy) {
                err -= dy;
                x1 += sx;
            }
            if (e2 < dx) {
                err += dx;
                y1 += sy;
            }
        }
    }

    std::pair<int, int> bmp_read(const std::string& filename) { // Чтение BMP. Узнаем размер поля и инициализируем его.

        std::ifstream bmp_file(filename, std::ios::binary);

        if (!bmp_file) {
            std::cerr << "Failed to open BMP file: " << filename << std::endl;
            return { 0,0 };
        }

        unsigned char header[54];
        bmp_file.read(reinterpret_cast<char*>(header), 54);

        if (header[0] != 'B' || header[1] != 'M') {
            std::cerr << "Invalid BMP file:" << filename << std::endl;
            return { 0,0 };
        }

        int width = header[18] | (header[19] << 8) | (header[20] << 16) | (header[21] << 24);
        int height = header[22] | (header[23] << 8) | (header[24] << 16) | (header[25] << 24);

        return { height, width };
    }

    void load_data(std::ifstream& bmp_file, int length, int width) { // Чтение BMP. Выгружаем информацию в матрицу.
        for (int i = length - 1; i >= 0; --i) {
            for (int j = 0; j < width; j++) {
                unsigned char color = bmp_file.get();
                bmp_file.get();
                bmp_file.get();
                matrix[i][j] = color;
            }
            bmp_file.ignore((4 - (width * 3) % 4) % 4);
        }
    }

    void copyFromBMP(const std::string& filename) {
        auto dimensions = bmp_read(filename);
        if (dimensions.first == 0 || dimensions.second == 0) return;

        std::ifstream bmp_file(filename, std::ios::binary);
        bmp_file.ignore(54);
        load_data(bmp_file, dimensions.first, dimensions.second);
        bmp_file.close();
    }
};

//212-Konchugarov-Timur

class Srez { // Класс со срезом (bin) и алгоритмом волна (wave)

private:
    int length, width;
    Field trifield{ 0,0 };
    std::vector<std::vector<double>> CopyField;
    std::vector<Component> components;
    int count = 0;




    std::vector<Triangle> bowyerWatson(const std::vector<PointD>& points) {
        std::vector<Triangle> triangles;
        if (points.empty()) return triangles;

        // Создаем супер-треугольник
        double minX = points[0].x, maxX = points[0].x;
        double minY = points[0].y, maxY = points[0].y;
        for (const auto& p : points) {
            minX = std::min(minX, p.x); maxX = std::max(maxX, p.x);
            minY = std::min(minY, p.y); maxY = std::max(maxY, p.y);
        }

        double dx = (maxX - minX) * 10, dy = (maxY - minY) * 10;
        PointD p1(minX - dx, minY - dy);
        PointD p2(maxX + dx, minY - dy);
        PointD p3((minX + maxX) / 2, maxY + dy);
        triangles.emplace_back(p1, p2, p3);

        // Добавляем точки
        for (const auto& p : points) {
            std::vector<Triangle> badTriangles;
            for (const auto& tri : triangles) {
                if (isPointInCircumcircle(p, tri)) {
                    badTriangles.push_back(tri);
                }
            }

            std::vector<Edge> polygon;
            for (const auto& tri : badTriangles) {
                std::vector<Edge> edges = { {tri.a, tri.b}, {tri.b, tri.c}, {tri.c, tri.a} };
                for (const auto& edge : edges) {
                    bool isShared = false;
                    for (const auto& other : badTriangles) {
                        if (&tri == &other) continue;
                        if (otherHasEdge(other, edge)) isShared = true;
                    }
                    if (!isShared) polygon.push_back(edge);
                }
            }

            // Удаляем плохие треугольники
            triangles.erase(std::remove_if(triangles.begin(), triangles.end(),
                [&](const Triangle& t) { return std::find(badTriangles.begin(), badTriangles.end(), t) != badTriangles.end(); }),
                triangles.end());

            // Добавляем новые
            for (const auto& edge : polygon) {
                triangles.emplace_back(edge.a, edge.b, p);
            }
        }

        triangles.erase(std::remove_if(triangles.begin(), triangles.end(),
            [&](const Triangle& t) {
                return contains(t, p1) || contains(t, p2) || contains(t, p3);
            }), triangles.end());

        return triangles;
    }

    bool isPointInCircumcircle(const PointD& p, const Triangle& tri) {
        double ax = tri.a.x - p.x, ay = tri.a.y - p.y;
        double bx = tri.b.x - p.x, by = tri.b.y - p.y;
        double cx = tri.c.x - p.x, cy = tri.c.y - p.y;

        double det = ax * (by * (cx * cx + cy * cy) - cy * (bx * bx + by * by))
            - ay * (bx * (cx * cx + cy * cy) - cx * (bx * bx + by * by))
            + (ax * ax + ay * ay) * (bx * cy - by * cx);
        return det > 0;
    }

    bool otherHasEdge(const Triangle& tri, const Edge& edge) {
        return Edge(tri.a, tri.b) == edge ||
            Edge(tri.b, tri.c) == edge ||
            Edge(tri.c, tri.a) == edge;
    }

    bool contains(const Triangle& tri, const PointD& p) {
        return tri.a == p || tri.b == p || tri.c == p;
    }

    // Проверка попадания точки в компоненту
    bool isPointInComponent(int x, int y) const {
        for (const auto& comp : components) {
            if (x >= comp.min_x && x <= comp.max_x &&
                y >= comp.min_y && y <= comp.max_y) {
                if (abs(comp.comp[x][y]) < EPS) return true;
            }
        }
        return false;
    }

    // Поиск кратчайшего пути (алгоритм A*)
    std::vector<Point> findPath(const Point& start, const Point& end) {
        std::vector<Point> path;
        if (isPointInComponent(start.x, start.y) || isPointInComponent(end.x, end.y)) {
            return path; // Путь невозможен
        }

        // Матрица посещенных точек
        std::vector<std::vector<bool>> visited(length, std::vector<bool>(width, false));

        // Очередь с приоритетом для A*
        using Node = std::pair<int, Point>;
        auto cmp = [](const Node& a, const Node& b) { return a.first > b.first; };
        std::priority_queue<Node, std::vector<Node>, decltype(cmp)> pq(cmp);

        // Матрица предыдущих точек
        std::vector<std::vector<Point>> came_from(length, std::vector<Point>(width));

        // Эвристика (манхэттенское расстояние)
        auto heuristic = [](Point a, Point b) {
            return abs(a.x - b.x) + abs(a.y - b.y);
            };

        pq.push({ heuristic(start, end), start });
        visited[start.x][start.y] = true;

        const int dx[8] = { 1, -1, 0, 0, -1, 1, 1, -1 };
        const int dy[8] = { 0, 0, 1, -1, -1, 1, -1, 1 };

        while (!pq.empty()) {
            auto current = pq.top().second;
            pq.pop();

            if (current.x == end.x && current.y == end.y) {
                // Восстановление пути
                while (!(current.x == start.x && current.y == start.y)) {
                    path.push_back(current);
                    current = came_from[current.x][current.y];
                }
                path.push_back(start);
                std::reverse(path.begin(), path.end());
                return path;
            }

            for (int i = 0; i < 8; ++i) {
                Point neighbor{ current.x + dx[i], current.y + dy[i] };

                if (neighbor.x >= 0 && neighbor.y >= 0 &&
                    neighbor.x < length && neighbor.y < width &&
                    !visited[neighbor.x][neighbor.y] &&
                    !isPointInComponent(neighbor.x, neighbor.y)) {

                    visited[neighbor.x][neighbor.y] = true;
                    came_from[neighbor.x][neighbor.y] = current;
                    int priority = heuristic(neighbor, end);
                    pq.push({ priority, neighbor });
                }
            }
        }

        return path; // Путь не найден
    }

    // Проверка пересечения двух отрезков
    bool doSegmentsIntersect(const PointD& p1, const PointD& p2, const PointD& p3, const PointD& p4) {
        auto ccw = [](const PointD& a, const PointD& b, const PointD& c) {
            return (c.y - a.y) * (b.x - a.x) > (b.y - a.y) * (c.x - a.x);
            };

        return ccw(p1, p3, p4) != ccw(p2, p3, p4) && ccw(p1, p2, p3) != ccw(p1, p2, p4);
    }

    // Вычисление угла между двумя отрезками в точке пересечения
    double calculateIntersectionAngle(const PointD& p1, const PointD& p2,
        const PointD& p3, const PointD& p4) {
        // Вектор первого отрезка
        double v1x = p2.x - p1.x;
        double v1y = p2.y - p1.y;

        // Вектор второго отрезка
        double v2x = p4.x - p3.x;
        double v2y = p4.y - p3.y;

        // Угол между векторами
        double dot = v1x * v2x + v1y * v2y;
        double det = v1x * v2y - v1y * v2x;
        return atan2(det, dot);
    }

public:

    int machineRadius = 0;
    void setMachineRadius(int r) {
        machineRadius = r;
    }

    void bin(int h, Field& f) { // Функция для получения разреза на высоте h
        CopyField.resize(f.matrix.size(), std::vector<double>(f.matrix[0].size(), 0));
        length = f.matrix.size();
        width = f.matrix[0].size();

        for (int i = length - 1; i >= 0; --i) {
            for (int j = 0; j < width; ++j) {
                if (f.matrix[i][j] < h) {
                    CopyField[i][j] = 255;
                }
                else {
                    CopyField[i][j] = 0;
                }
            }
        }
        Field pole(f.length, f.width);
        pole.matrix = CopyField;
        pole.bmp_write("bin.bmp", 2);
        trifield = pole;
    }

    const std::vector<Component>& getComponents() const {
        return components;
    }

    void wave(int n) { // Функция запуска алгоритма wave и записи соотв. компонент в вектор
        Component Componenta(length, width);
        int c = 0; // Тут будет записано значение count

        for (int i = length - 1; i >= 0; --i) {
            for (int j = 0; j < width; ++j) {
                c = inc(Componenta.comp, i, j);

                if (c > n) {
                    components.emplace_back(Componenta);
                    components.back().calculate_metadata();
                    Componenta = Component(length, width);
                }

                else if (c > 0 && c < n) {
                    Componenta = Component(length, width);
                }

                count = 0;

            }
        }

        for (int i = 0; i < (int)components.size(); i++) {
            Field compole(length, width);
            compole.matrix = components[i].comp;
            compole.colors.resize(length, std::vector<Color>(width, { 255, 255, 255 }));
            compole.center_x = components[i].center_x;
            compole.center_y = components[i].center_y;
            compole.eigenvec1_x = components[i].eigenvec1_x;
            compole.eigenvec1_y = components[i].eigenvec1_y;
            compole.eigenvec2_x = components[i].eigenvec2_x;
            compole.eigenvec2_y = components[i].eigenvec2_y;
            compole.eigenvalue1 = components[i].eigenvalue1;
            compole.eigenvalue2 = components[i].eigenvalue2;
            compole.bmp_write("Comp" + std::to_string(i + 1) + ".bmp", 1);
        }

        // Обновляем cleanedField, оставляя только значимые компоненты
        trifield.matrix.assign(trifield.matrix.size(),
            std::vector<double>(trifield.matrix[0].size(), 255.0));
        for (const auto& comp : components) {
            for (size_t i = 0; i < comp.comp.size(); ++i) {
                for (size_t j = 0; j < comp.comp[0].size(); ++j) {
                    if (abs(comp.comp[i][j]) < EPS) {
                        trifield.matrix[i][j] = 0.0;
                    }
                }
            }
        }
    }
    /*
    int inc(std::vector<std::vector<double>>& component, int x, int y, int k) { // Сама функция wave
        if (x < 1 || y < 1 || x >(int) component.size() - 1 || y >(int) component[0].size() - 1 || (int)CopyField[x][y] == 255) return -1;

        else {
            CopyField[x][y] = 255;
            //count = count < k + 1 ? k + 1 : count;
            count++;
            component[x][y] = 0;
            inc(component, x + 1, y, k + 1);
            inc(component, x - 1, y, k + 1);
            inc(component, x, y + 1, k + 1);
            inc(component, x, y - 1, k + 1);
            inc(component, x - 1, y - 1, k + 1);
            inc(component, x + 1, y + 1, k + 1);
            inc(component, x + 1, y - 1, k + 1);
            inc(component, x - 1, y + 1, k + 1);
        }
        return count;
    }*/


    int inc(std::vector<std::vector<double>>& component, int startX, int startY) {
        // Проверка валидности начальной точки
        if (startX < 0 || startY < 0 ||
            startX >= int(component.size()) ||
            startY >= int(component[0].size()) ||
            int(CopyField[startX][startY]) == 255) {
            return 0;
        }

        std::queue<std::pair<int, int>> q;
        q.push(std::make_pair(startX, startY));
        CopyField[startX][startY] = 255;
        component[startX][startY] = 0;
        int count = 1;

        // Смещения для 8 соседей
        const int dx[8] = { 1, -1, 0, 0, -1, 1, 1, -1 };
        const int dy[8] = { 0, 0, 1, -1, -1, 1, -1, 1 };

        while (!q.empty()) {
            int x = q.front().first;
            int y = q.front().second;
            q.pop();

            for (int i = 0; i < 8; ++i) {
                int nx = x + dx[i];
                int ny = y + dy[i];

                if (nx >= 0 && ny >= 0 &&
                    nx < int(component.size()) &&
                    ny < int(component[0].size()) &&
                    int(CopyField[nx][ny]) != 255) {
                    CopyField[nx][ny] = 255;
                    component[nx][ny] = 0;
                    count++;
                    q.push(std::make_pair(nx, ny));
                }
            }
        }

        return count;
    }


    std::vector<Color> generateColor(int k) { //Генерируем свой цвет для каждого получившегося класстера
        std::vector<Color> colors;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(0, 255);

        for (int i = 0; i < k; i++) {
            colors.push_back({ static_cast<uint8_t>(dis(gen)),
                              static_cast<uint8_t>(dis(gen)),
                              static_cast<uint8_t>(dis(gen)) });
        }

        return colors;
    }

    void kMeans(int k, int p) { // Алгоритм kMeands

        //Первый этап: стандартный kMeans для поиска кластеров
        std::vector<Point> points;

        for (const auto& comp : components) {
            for (int i = 0; i < length; i++) {
                for (int j = 0; j < width; j++) {
                    if (abs(comp.comp[i][j]) < EPS) {
                        points.push_back({ i, j }); // Вытаскиваем точки из разреза
                    }
                }
            }
        }

        if (points.empty()) { // Проверка, смогли ли что-то вытащить 
            std::cerr << "No points available for clustering!" << std::endl;
            return;
        }

        std::vector<Point> centroids;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis_x(0, length - 1);
        std::uniform_int_distribution<> dis_y(0, width - 1);

        for (int i = 0; i < k; i++) {
            centroids.push_back({ dis_x(gen), dis_y(gen) }); // Рандомно выбираем k центроид
        }

        bool changed = true;
        std::vector<int> labels(points.size(), -1);

        while (changed) {
            changed = false;

            //Шаг 1: назначаем каждую точку ближайшему центроиду
            for (size_t i = 0; i < points.size(); i++) {
                double minDist = std::numeric_limits<double>::max();
                int cluster = -1;

                for (int j = 0; j < k; j++) { //Считаем расстояние от точки до каждого класстера, выбираем среди них минимальное
                    double dist = std::pow(points[i].x - centroids[j].x, 2) + std::pow(points[i].y - centroids[j].y, 2);
                    if (dist < minDist) {
                        minDist = dist;
                        cluster = j;
                    }
                }

                if (labels[i] != cluster) { //Проверяем, поменяла хоть одна точка класстер
                    labels[i] = cluster; // Если поменяла, то меняем ее класстер
                    changed = true;
                }
            }

            //Шаг 2: пересчитываем центроиды
            std::vector<Point> newCentroids(k, { 0, 0 });
            std::vector<int> counts(k, 0);

            for (size_t i = 0; i < points.size(); i++) { // Для каждой центроиды складываем все координаты точек, вычисляем количество точек в них
                newCentroids[labels[i]].x += points[i].x;
                newCentroids[labels[i]].y += points[i].y;
                counts[labels[i]]++;
            }

            for (int j = 0; j < k; j++) {
                if (counts[j] > 0) {
                    newCentroids[j].x /= counts[j];
                    newCentroids[j].y /= counts[j];
                }
            }

            centroids = newCentroids;

        }

        std::vector<Color> clusterColors = generateColor(k); //Генерируем цвета для кластеров
        std::vector<std::vector<Color>> clusterImage(length, std::vector<Color>(width, { 255, 255, 255 })); //Заополняем матрицу цветов каждого пикселя
        std::vector<std::vector<Point>> clusteredPoints(k);

        for (size_t i = 0; i < points.size(); i++) { //Каждой точке -- свой цвет
            int cluster = labels[i];
            clusterImage[points[i].x][points[i].y] = clusterColors[cluster];
            clusteredPoints[cluster].push_back(points[i]); //Добвавляем каждую точку в свой кластер
        }

        // Второй этап: запуск kMeans для каждого из k кластеров с числом центроид p
        std::vector<std::vector<Point>> subCentroids(k);

        for (int clusterIdx = 0; clusterIdx < k; clusterIdx++) { //Для каждого кластера выполняем алгоритм kMeans
            if (clusteredPoints[clusterIdx].empty()) continue;

            std::vector<Point>& clusterPoints = clusteredPoints[clusterIdx];
            std::vector<Point> subCenters(p);

            //Инициализируем p случайных центроид 
            for (int i = 0; i < p; i++) {
                subCenters[i] = clusterPoints[std::uniform_int_distribution<>(0, clusterPoints.size() - 1)(gen)];
            }

            bool subChanged = true;
            std::vector<int> subLabels(clusterPoints.size(), -1);

            while (subChanged) {
                subChanged = false;

                //Назначение точек 
                for (size_t i = 0; i < clusterPoints.size(); i++) {
                    double minDist = std::numeric_limits<double>::max();
                    int subCluster = -1;

                    for (int j = 0; j < p; j++) {
                        double dist = std::pow(clusterPoints[i].x - subCenters[j].x, 2) +
                            std::pow(clusterPoints[i].y - subCenters[j].y, 2);
                        if (dist < minDist) {
                            minDist = dist;
                            subCluster = j;
                        }
                    }

                    if (subLabels[i] != subCluster) {
                        subLabels[i] = subCluster;
                        subChanged = true;
                    }
                }

                //Пересчет субцентроид 
                std::vector<Point> newSubCenters(p, { 0, 0 });
                std::vector<int> subCounts(p, 0);

                for (size_t i = 0; i < clusterPoints.size(); i++) {
                    newSubCenters[subLabels[i]].x += clusterPoints[i].x;
                    newSubCenters[subLabels[i]].y += clusterPoints[i].y;
                    subCounts[subLabels[i]]++;
                }

                for (int j = 0; j < p; j++) {
                    if (subCounts[j] != 0) {
                        newSubCenters[j].x /= subCounts[j];
                        newSubCenters[j].y /= subCounts[j];
                    }
                }

                subCenters = newSubCenters;
            }
            subCentroids[clusterIdx] = subCenters;

            //Визуализация субцентроид 
            for (const auto& subCenter : subCenters) {
                clusterImage[subCenter.x][subCenter.y] = { 0, 0, 0 }; //Цвет для центров тяжести
            }
        }

        Field pole(length, width);
        pole.colors = clusterImage;
        pole.bmp_write("clusters.bmp", 3);
    }


    void triangulateAndDraw(const std::string& filename) {
        std::vector<PointD> centers;
        for (const auto& comp : components) {
            centers.emplace_back(comp.center_x, comp.center_y);
        }

        auto triangles = bowyerWatson(centers);
        Color green = { 0, 255, 0 };

        for (const auto& tri : triangles) {
            trifield.draw_line(tri.a.x, tri.a.y, tri.b.x, tri.b.y, green);
            trifield.draw_line(tri.b.x, tri.b.y, tri.c.x, tri.c.y, green);
            trifield.draw_line(tri.c.x, tri.c.y, tri.a.x, tri.a.y, green);
        }

        trifield.bmp_write(filename, 1);
    }


    // void kMeans(int k, int p){
    //     if(p <= 0){
    //         std::cerr << "Parametr p must be greater than 0!" << std::endl;
    //         return;
    //     }

    //     std::vector<Point> points;

    //     for(const auto& comp : components){
    //         for(int i = 0; i < length; i++){
    //             for(int j = 0; j < width; j++){
    //                 if(comp.comp[i][j] == 0){
    //                     points.push_back({i, j}); //Извлекаем точки из разреза
    //                 }
    //             }
    //         }
    //     }

    //     if(points.empty()){
    //         std::cerr << "No points avalable for clustering!" << std:: endl;
    //         return;
    //     }

    //     //Инициализируем p центроид для каждого класса
    //     std::vector<std::vector<Point>> cores (k, std::vector<Point>(p));
    //     std::random_device rd; //Высокорандомное число
    //     std::mt19937 gen(rd()); //Псевдорандом на основе истинно рандомного первого числа
    //     std::uniform_int_distribution<> dis (0, points.size() - 1); 

    //     std::set<int> usedIndexes; //Множество для проверки, выбран ли уже данный индекс

    //     for(int i = 0; i < k; i++){
    //         for(int j = 0; j < p; j++){
    //             int idx;
    //             //Пытаемся выбрать уникальный индекс

    //             do{
    //                 idx = dis(gen);
    //             } while(usedIndexes.find(idx) != usedIndexes.end()); // Проверка на уникальность выбранного индекса

    //             usedIndexes.insert(idx); //Запоминаем индекс как использованый
    //             cores[i][j] = points[idx]; //Присваиваем центроид
    //         }
    //     }

    //     bool changed = true;
    //     std::vector<int> labels(points.size(), -1); //Метки для каждой точки (к каким кластерам они относятся)

    //     while(changed){
    //         changed = false; 

    //         //Шаг 1: Назначаем каждую точку одному из k классов, учитывая p центроид 
    //         for(size_t i = 0; i < points.size(); i++){
    //             double minDist = std::numeric_limits<double>::max();
    //             int cluster = -1;

    //             for(int j = 0; j < k; j++){
    //                 for(int l = 0; l < p; l++){
    //                     double dist = std::pow(points[i].x - cores[j][l].x, 2) + 
    //                                   std::pow(points[i].y - cores[j][l].y, 2);
    //                     if(dist < minDist){
    //                         minDist = dist;
    //                         cluster = j;
    //                     }
    //                 }
    //             }

    //             if(labels[i] != cluster){ //Проверка, изменила ли точка свой кластер 
    //                 labels[i] = cluster;
    //                 changed = true;
    //             }
    //         }

    //         //Шаг 2: Пересчет центрид
    //         for(int j = 0; j < k; j++){
    //             std::vector<Point> clusterPoints; // Точки, которые принадлежат данному класстеру

    //             for(size_t i = 0; i < points.size(); i++){ // Если точка принадлежит кластеру j, то добавляем ее в clusterPoints
    //                 if(labels[i] == j){
    //                     clusterPoints.push_back(points[i]);
    //                 }
    //             }

    //             if(clusterPoints.empty()){
    //                 std::cerr << "Cluster" << j << "has no points" << std::endl;
    //                 continue;
    //             }

    //             //Далее применяем kMeans внутри класса для p центроид ядра
    //             std::vector<Point> newCentroids(p, {0, 0}); //Центроиды класса (p штук)
    //             std::vector<int> counts(p, 0); //Количество точек в каждом из p кластеров 

    //             for(const auto& pt : clusterPoints){ //Для каждой точки из кластера...
    //                 double minDist = std::numeric_limits<double>::max();
    //                 int centroidIndex = -1;

    //                 //Находим ближайшую центроиду внутри класса
    //                 for(int l = 0; l < p; l++){
    //                     double dist = std::pow(pt.x - cores[j][l].x, 2) +
    //                                   std::pow(pt.y - cores[j][l].y, 2); // Считаем минимальное расстояние до каждой центроиды класстера
    //                     if(dist < minDist){
    //                         minDist = dist;
    //                         centroidIndex = l;
    //                     }
    //                 }

    //                 //Добавляем к центроиде координаты точки, которая ей принадлежит
    //                 newCentroids[centroidIndex].x += pt.x;
    //                 newCentroids[centroidIndex].y += pt.y;
    //                 counts[centroidIndex]++;
    //             }

    //             //Считаем центры тяжести кластеров
    //             for(int l = 0; l < p; l++){
    //                 if(counts[l] != 0){ // Если в данном кластере есть хотя бы одна точка
    //                     newCentroids[l].x /= counts[l];
    //                     newCentroids[l].y /= counts[l];
    //                 }
    //             }
    //             cores[j] = newCentroids; // Обновляем ядро класса
    //         }
    //     }

    //     //Генерация цветов для кластеров
    //     std::vector<Color> clusterColors = generateColor(k);
    //     std::vector<std::vector<Color>> clusterImage(length, std::vector<Color>(width, {255, 255, 255}));
    //     Color brightRed = {0, 0, 255}; //Центроиды будут подсвечены ярко-красным

    //     for(size_t i = 0; i < points.size(); i++){ //Каждой точке свой цвет в зависимости от кластера
    //         int cluster = labels[i];
    //         clusterImage[points[i].x][points[i].y] = clusterColors[cluster];
    //     }

    //     //Подсвечиваем центроиды
    //     for(int j = 0; j < k; j ++){
    //         for(int l = 0; l < p; l++){
    //             int x = cores[j][l].x;
    //             int y = cores[j][l].y;

    //             if(x >= 0 && x < length && y>= 0 && y < width){
    //                 clusterImage[x][y] = brightRed;
    //             }
    //         }
    //     }

    //     Field pole(length, width);
    //     pole.colors = clusterImage;
    //     pole.bmp_write("cluster2.bmp", 2); //Сохраняем результат
    // }

    // Метод для построения и отрисовки пути
    void buildRoad(Field& field, const Point& start, const Point& end, const std::string& filename) {
        try {
            if (components.empty()) {
                throw std::runtime_error("Нет компонент — триангуляция невозможна");
            }

            // 1. Проверка входных данных
            if (field.length <= 0 || field.width <= 0) {
                throw std::runtime_error("Invalid field dimensions");
            }

            // Проверка координат точек
            auto check_point = [&](const Point& p, const std::string& name) {
                if (p.x < 0 || p.x >= field.length || p.y < 0 || p.y >= field.width) {
                    throw std::runtime_error(name + " point out of field bounds");
                }
                };
            check_point(start, "Start");
            check_point(end, "End");

            // 2. Инициализация результата
            Field resultField(field.length, field.width);

            // Копируем исходное поле с компонентами
            for (int i = 0; i < field.length; ++i) {
                for (int j = 0; j < field.width; ++j) {
                    resultField.colors[i][j] = { 255, 255, 255 }; // Белый фон
                    // Копируем препятствия из исходного поля
                    for (const auto& comp : components) {
                        if (i < int(comp.comp.size()) && j < int(comp.comp[i].size()) && abs(comp.comp[i][j]) <EPS) {
                            resultField.colors[i][j] = { 0, 0, 0 }; // Черные препятствия
                        }
                    }
                }
            }

            // 3. Получение центров компонент
            std::vector<PointD> centers;
            for (const auto& comp : components) {
                centers.emplace_back(comp.center_x, comp.center_y);
            }

            // 4. Триангуляция Делоне
            auto triangles = bowyerWatson(centers);
            if (triangles.empty()) {
                throw std::runtime_error("Triangulation failed");
            }

            // 5. Отрисовка триангуляции (серым цветом)
            Color triangColor = { 200, 200, 200 }; // Серый цвет для триангуляции
            for (const auto& tri : triangles) {
                resultField.draw_line(tri.a.x, tri.a.y, tri.b.x, tri.b.y, triangColor);
                resultField.draw_line(tri.b.x, tri.b.y, tri.c.x, tri.c.y, triangColor);
                resultField.draw_line(tri.c.x, tri.c.y, tri.a.x, tri.a.y, triangColor);
            }

            // 6. Сбор уникальных рёбер и их середин
            std::set<Edge> edgeSet;
            std::vector<PointD> midPoints;
            std::vector<Edge> allEdges; // Для визуализации пересекаемых рёбер

            for (const auto& tri : triangles) {
                Edge edges[3] = { {tri.a, tri.b}, {tri.b, tri.c}, {tri.c, tri.a} };
                for (const auto& edge : edges) {
                    if (edgeSet.insert(edge).second) {
                        // Добавляем середину ребра
                        midPoints.emplace_back(
                            (edge.a.x + edge.b.x) / 2,
                            (edge.a.y + edge.b.y) / 2
                        );
                        allEdges.push_back(edge);
                    }
                }
            }

            // 7. Функция проверки пересечения с препятствиями (улучшенная)
            auto doesIntersectObstacle = [&](const PointD& p1, const PointD& p2) -> bool {
                const int r = machineRadius;

                for (int i = -r; i <= r; ++i) {
                    for (int j = -r; j <= r; ++j) {
                        if (i * i + j * j > r * r) continue;

                        PointD offset1{ p1.x + i, p1.y + j };
                        PointD offset2{ p2.x + i, p2.y + j };

                        int x1 = static_cast<int>(offset1.x);
                        int y1 = static_cast<int>(offset1.y);
                        int x2 = static_cast<int>(offset2.x);
                        int y2 = static_cast<int>(offset2.y);

                        int dx = abs(x2 - x1);
                        int dy = abs(y2 - y1);
                        int sx = (x1 < x2) ? 1 : -1;
                        int sy = (y1 < y2) ? 1 : -1;
                        int err = dx - dy;

                        while (true) {
                            if (x1 >= 0 && x1 < field.length && y1 >= 0 && y1 < field.width) {
                                for (const auto& comp : components) {
                                    // Не локальная проверка по comp.comp[x][y], а глобальная — по координатам на всем поле
                                    if (x1 >= comp.min_x && x1 <= comp.max_x &&
                                        y1 >= comp.min_y && y1 <= comp.max_y) {
                                        if (x1 < int(comp.comp.size()) && y1 < int(comp.comp[0].size()) &&
                                            abs(comp.comp[x1][y1])<EPS) {
                                            return true; // столкновение!
                                        }
                                    }
                                }
                            }

                            if (x1 == x2 && y1 == y2) break;

                            int e2 = 2 * err;
                            if (e2 > -dy) { err -= dy; x1 += sx; }
                            if (e2 < dx) { err += dx; y1 += sy; }
                        }
                    }
                }

                return false;
                };



            // 8. Построение графа (узлы - середины рёбер + start + end)
            PointD startD{ static_cast<double>(start.x), static_cast<double>(start.y) };
            PointD endD{ static_cast<double>(end.x), static_cast<double>(end.y) };
            midPoints.push_back(startD);
            midPoints.push_back(endD);
            const size_t startIdx = midPoints.size() - 2;
            const size_t endIdx = midPoints.size() - 1;

            // Матрица смежности
            const size_t n = midPoints.size();
            std::vector<std::vector<double>> adjMatrix(n, std::vector<double>(n, std::numeric_limits<double>::infinity()));

            // Заполняем матрицу смежности
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = i + 1; j < n; ++j) {
                    if (!doesIntersectObstacle(midPoints[i], midPoints[j])) {
                        // Вычисляем расстояние с учетом углов пересечения
                        double dx = midPoints[j].x - midPoints[i].x;
                        double dy = midPoints[j].y - midPoints[i].y;
                        double dist = sqrt(dx * dx + dy * dy);

                        // Учитываем угол пересечения с ребрами триангуляции
                        for (const auto& edge : allEdges) {
                            if (doSegmentsIntersect(midPoints[i], midPoints[j], edge.a, edge.b)) {
                                // Находим угол между текущим отрезком и ребром триангуляции
                                double angle = calculateIntersectionAngle(
                                    midPoints[i], midPoints[j], edge.a, edge.b);
                                // Штрафуем за отклонение от 90 градусов
                                dist += abs(angle - M_PI / 2) * 0.1;
                            }
                        }

                        adjMatrix[i][j] = adjMatrix[j][i] = dist;
                    }
                }
            }

            // 9. Алгоритм Дейкстры
            std::vector<double> dist(n, std::numeric_limits<double>::infinity());
            std::vector<size_t> prev(n, SIZE_MAX);
            dist[startIdx] = 0;

            auto cmp = [](const std::pair<double, size_t>& left, const std::pair<double, size_t>& right) {
                return left.first > right.first;
                };
            std::priority_queue<std::pair<double, size_t>, std::vector<std::pair<double, size_t>>, decltype(cmp)> pq(cmp);
            pq.push({ 0, startIdx });

            while (!pq.empty()) {
                auto current = pq.top();
                pq.pop();
                size_t u = current.second;

                if (current.first > dist[u]) continue;

                for (size_t v = 0; v < n; ++v) {
                    double edgeWeight = adjMatrix[u][v];
                    if (edgeWeight < std::numeric_limits<double>::infinity()) {
                        double newDist = dist[u] + edgeWeight;
                        if (newDist < dist[v]) {
                            dist[v] = newDist;
                            prev[v] = u;
                            pq.push({ newDist, v });
                        }
                    }
                }
            }

            // 10. Восстановление пути
            if ( abs(dist[endIdx] - std::numeric_limits<double>::infinity())<EPS ) {
                throw std::runtime_error("Path not found");
            }

            std::vector<PointD> path;
            for (size_t at = endIdx; at != SIZE_MAX; at = prev[at]) {
                path.push_back(midPoints[at]);
                if (path.size() > n) break; // Защита от зацикливания
            }
            std::reverse(path.begin(), path.end());

            // 11. Отрисовка пути (красным с толщиной 2 пикселя)
            Color pathColor = { 255, 0, 0 }; // Красный цвет для пути
            for (size_t i = 0; i < path.size() - 1; ++i) {
                // Отрисовка с утолщением
                resultField.draw_line(path[i].x, path[i].y, path[i + 1].x, path[i + 1].y, pathColor);
                resultField.draw_line(path[i].x + 1, path[i].y, path[i + 1].x + 1, path[i + 1].y, pathColor);
                resultField.draw_line(path[i].x, path[i].y + 1, path[i + 1].x, path[i + 1].y + 1, pathColor);
            }

            // 12. Отрисовка старта и финиша
            auto draw_point = [&](int x, int y, Color color) {
                for (int dx = -2; dx <= 2; ++dx) {
                    for (int dy = -2; dy <= 2; ++dy) {
                        if (abs(dx) + abs(dy) <= 2) { // Ромбик 5x5
                            int nx = x + dx;
                            int ny = y + dy;
                            if (nx >= 0 && nx < field.length && ny >= 0 && ny < field.width) {
                                resultField.colors[nx][ny] = color;
                            }
                        }
                    }
                }
                };

            draw_point(start.x, start.y, { 0, 255, 0 }); // Зеленый старт
            draw_point(end.x, end.y, { 0, 0, 255 });     // Синий финиш

            // 13. Подсветка пересекаемых рёбер триангуляции (желтым)
            Color highlightColor = { 255, 255, 0 }; // Желтый
            for (size_t i = 0; i < path.size() - 1; ++i) {
                for (const auto& edge : allEdges) {
                    if (doSegmentsIntersect(path[i], path[i + 1], edge.a, edge.b)) {
                        // Подсвечиваем ребро триангуляции, которое пересекает путь
                        resultField.draw_line(edge.a.x, edge.a.y, edge.b.x, edge.b.y, highlightColor);
                    }
                }
            }

            // 14. Сохранение результата
            resultField.bmp_write(filename, 3);

        }
        catch (const std::exception& e) {
            std::cerr << "Error in buildRoad: " << e.what() << std::endl;
            throw;
        }
    }
};

//212-Konchugarov-Timur

class Control { // Класс Control выполняет роль диспетчера
    std::vector<Gauss> gausses;
    Field* field = nullptr;
    Srez srez;

public:
    Server server;

    void wave_cntrl(int n) { // Алгоритм wave

        if (!field) {
            std::cerr << "Error! Field not initialized!" << std::endl;
            server.log("Control received 'bin', but field was not initialized.");
            return;
        }

        srez.wave(n);
        server.log("Control completed the wave program");
    }

    void triangulateCmd(const std::string& filename) {
        if (!field) {
            server.log("Erroe: Field not init");
            return;
        }
        srez.triangulateAndDraw(filename);
        server.log("Triangulation saved in " + filename);
    }

    void init(int length, int width) { // Инициализация поля
        if (field) {
            server.log("Control received 'init', but field was already initialized.");
            return;
        }
        field = new Field(length, width); //new возвращает указатель
        server.log("Field created with size " + std::to_string(length) + "x" + std::to_string(width));
        server.log("Control passed 'init' command to Field");
    }

    void bin_cntrl(int h) { // Разрез
        if (!field) {
            std::cerr << "Error! Field not initialized!" << std::endl;
            server.log("Control received 'bin', but field was not initialized.");
            return;
        }
        srez.bin(h, *field);
        server.log("A section on height " + std::to_string(h) + " is obtained");
    }

    void kMeans_cntrl(int k, int p) { // Алгоритм kMeans
        if (!field) {
            std::cerr << "Error! Field not initialized!" << std::endl;
            server.log("Control received 'KMEANS', but field was not initialized.");
            return;
        }
        server.log("Algotitgm KMEANS for the k =" + std::to_string(k) + " is launched");
        srez.kMeans(k, p);
        server.log("Algorithm KMEANS for the k =" + std::to_string(k) + " is completed");
    }

    void add_gauss(double h, double x0, double y0, double sigma_x, double sigma_y) { // Добавление гаусса в список

        if (!field) {
            std::cerr << "Error! Field not initialized!" << std::endl;
            server.log("Control received 'g', but field was not initialized.");
            return;
        }

        gausses.emplace_back(h, x0, y0, sigma_x, sigma_y);


        // Логирование факта передачи команды Gauss
        server.log("Control passed Gauss command to Field with (h=" + std::to_string(h) + ", x0=" + std::to_string(x0) +
            ", y0=" + std::to_string(y0) + ", sigma_x=" + std::to_string(sigma_x) + ", sigma_y=" + std::to_string(sigma_y) + ")");
    }


    void generate() { //Функция, которая добавляет список гауссов
        if (!field) {
            std::cerr << "Error! Field not initialized!" << std::endl;
            server.log("Control received 'generate', but field was not initialized.");
            return;
        }

        if (gausses.empty()) {
            server.log("No gausses to apply in 'generate'");
            return;
        }

        server.log("Control passed 'generate' command to Field");

        for (size_t i = 0; i < gausses.size(); ++i) {
            server.log("Control is applying Gauss #" + std::to_string(i + 1) + " to Field");
            field->apply_gauss(gausses[i]);
            server.log("Gauss #" + std::to_string(i + 1) + "applied");
        }

        field->normalize();
        gausses.clear();
        server.log("Gauss completed applying all Gausses");
    }


    void gnuplot(const std::string& filename) { // Гнуплот
        if (!field) {
            std::cerr << "Error! Field not initialized!" << std::endl;
            server.log("Control received 'gnuplot', but field was not initialized.");
            return;
        }

        field->save_to_gnuplot("gnuplot_" + filename + ".txt");
        server.log("Field saved to Gnuplot file: " + filename);

        std::ofstream gp_file("gnuplot_comm.txt");
        gp_file << "set view 60,30\n"; // угол в градусах по вертикальной и по горизонтальной оси
        gp_file << "set palette defined (0 \"blue\", 1 \"red\")\n";
        gp_file << "set pm3d at s\n";
        gp_file << "splot 'gnuplot_" << filename << ".txt' with lines\n";
        gp_file << "pause -1";
        gp_file.close();
        server.log("Field generated Gnuplot file: " + filename);
    }

    void bmp_write_cntrl(const std::string& filename, int k) { //Запись BMP
        if (!field) {
            std::cerr << "Error! Field not initialized!" << std::endl;
            server.log("Control received 'save bmp', but field was not initialized.");
            return;
        }
        field->bmp_write(filename, k);
        server.log("Field saved to BMP file:" + filename);
    }

    void bmp_read_cntrl(const std::string& filename) { // Чтение BMP
        if (!field) {
            std::cerr << "Error! Field not initialized!" << std::endl;
            server.log("Control received 'read bmp', but field was not initialized.");
            return;
        }

        std::pair<int, int> dimensions = field->bmp_read(filename);
        int new_length = dimensions.first;
        int new_width = dimensions.second;

        if (new_length == 0 || new_width == 0) {
            std::cerr << "Error reading BMP file. No changes made" << std::endl;
            server.log("Error reading BMP file");
        }

        delete field;
        field = new Field(new_length, new_width);

        std::ifstream bmp_file(filename, std::ios::binary);
        bmp_file.ignore(54);
        field->load_data(bmp_file, new_length, new_width);
        bmp_file.close();

        server.log("Field loaded bmp file:" + filename);
    }

    void roadCmd(int x1, int y1, int x2, int y2) {
        if (!field) {
            server.log("Ошибка: поле не инициализировано");
            return;
        }

        Point start{ x1, y1 };
        Point end{ x2, y2 };

        try {
            // Проверка корректности координат
            if (x1 < 0 || x1 >= field->length || y1 < 0 || y1 >= field->width ||
                x2 < 0 || x2 >= field->length || y2 < 0 || y2 >= field->width) {
                server.log("Wrong coordinates start's or end's points");
                std::cerr << "Error: points not in Field!" << std::endl;
                return;
            }

            srez.triangulateAndDraw("triang.bmp");
            srez.buildRoad(*field, start, end, "road.bmp");
            server.log("Маршрут успешно построен и сохранён в road.bmp");
        }
        catch (const std::exception& e) {
            server.log(std::string("Ошибка при построении маршрута: ") + e.what());
            std::cerr << "Error in roadCmd: " << e.what() << std::endl;
        }
    }

    void setMachineRadius(int r) {
        srez.setMachineRadius(r);
    }

};

//212-Konchugarov-Timur

class Interface { // Клиент (интерфейс)
    Control control;
    Client client;

public:

    std::string trim(const std::string& str) { // Функция, чтобы убирать пробелы в начале и конце строки
        size_t first = str.find_first_not_of(' ');
        if (first == std::string::npos) {
            return "";
        }
        size_t last = str.find_last_not_of(' ');
        return str.substr(first, (last - first + 1));
    }

    int run() { // функция, которая непосредственно и является интерфейсом
        std::string filename;
        std::cout << "Write name of the config" << std::endl;
        std::cin >> filename;
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');  // Очещаем символ новой строки для корректной работы Batch-pause
        std::ifstream config(filename);
        bool gaussi = false; // Индикатор, нужен для гаусса по умолчанию в конфиге
        bool batch_pause = false; // Индикатор для Batch_pause
        int n = 0; // Количество пикселей, которые мы считаем шумом

        if (!config.is_open()) {
            std::cerr << "Failed to open file:" << filename << std::endl;
            return 1;
        }

        std::cout << "Succesfully opened file:" << filename << std::endl;
        std::string line;
        bool noize = false;

        while (!noize && std::getline(config, line)) {
            std::istringstream iss(line);
            std::string key, value;

            if (std::getline(iss, key, '=') && std::getline(iss, value)) {
                key = trim(key);
                value = trim(value);

                if (key == "Noize") {
                    std::istringstream ss(value);
                    ss >> n;
                    noize = true;
                }
            }
        }

        config.clear(); //Сбрасываем флаги ошибок
        config.seekg(0, std::ios::beg); // На начало конфига

        while (std::getline(config, line)) {
            std::istringstream iss(line);
            std::string key, value;

            if (std::getline(iss, key, '=') && std::getline(iss, value)) {
                key = trim(key);
                value = trim(value);

                if (key == "Log_Server") {
                    if (value == "ON") {
                        std::string log_filename = "server_log.txt";
                        std::getline(config, line);
                        std::istringstream iss_filename(line);
                        if (std::getline(iss_filename, key, '=') && std::getline(iss_filename, value)) {
                            key = trim(key);
                            value = trim(value);
                            if (key == "Log_Server_filename") {
                                log_filename = value;
                            }
                        }
                        control.server.enable_logging(log_filename);
                        std::cout << "Server client enabled with file: " << log_filename << std::endl;
                    }
                }


                else if (key == "Log_Client") {

                    if (value == "ON") {
                        std::string log_filename = "client_log.txt";
                        std::getline(config, line);
                        std::istringstream iss_filename(line);

                        if (std::getline(iss_filename, key, '=') && std::getline(iss_filename, value)) {
                            key = trim(key);
                            value = trim(value);
                            if (key == "Log_Client_filename") {
                                log_filename = value;
                            }
                        }

                        client.enable_logging(log_filename);
                        std::cout << "Server logging enabled with file: " << log_filename << std::endl;

                        if (!noize) {
                            client.log("The noise value was not found, so we assumed that there is no noise");
                        }

                        else {
                            client.log("The noise value is set to" + std::to_string(n));
                        }
                    }
                }

                else if (key == "Radius") {
                    int r = std::stoi(value);
                    control.setMachineRadius(r);  // Проброс в Srez
                    client.log("Radius set to " + std::to_string(r));
                }


                else if (key == "Batch_pause") {
                    if (value == "ON") {
                        batch_pause = true;
                    }
                    else {
                        batch_pause = false;
                    }
                }

                else if (key == "Batch_file") {
                    if (value == "ON") {

                        std::string Batch_filename = "Batch_file.txt";
                        std::getline(config, line);
                        std::istringstream iss_filename(line);

                        if (std::getline(iss_filename, key, '=') && std::getline(iss_filename, value)) {
                            key = trim(key);
                            value = trim(value);
                            if (key == "Batch_filename") {
                                Batch_filename = value;
                            }
                        }

                        std::ifstream batch(Batch_filename);

                        if (!batch.is_open()) {
                            std::cerr << "Failed to open file:" << Batch_filename << std::endl;
                            return 1;
                        }

                        std::string batch_line, batch_key, batch_value;
                        std::getline(batch, batch_line);
                        std::istringstream iss_batch(batch_line);
                        iss_batch >> batch_key;

                        if (batch_key != "INI") {
                            std::getline(config, line);
                            iss.clear();
                            iss.str(line);
                            std::getline(iss, key, '=');
                            std::getline(iss, value);
                            key = trim(key);
                            value = trim(value);

                            if (key == "Field_DEF") {
                                int length, width;
                                std::istringstream iss_Field(value);
                                iss_Field >> length >> width;
                                client.log("The interface has set the default value of the field: length=" + std::to_string(length) + ", width=" + std::to_string(width));
                                control.init(length, width);
                            }

                            else {
                                client.log("Field initialization error");
                                return -1;
                            }

                            batch.clear(); // Сбрасываем флаги ошибок (например, EOF)
                            batch.seekg(0, std::ios::beg);
                        }

                        else {
                            batch.clear(); // Сбрасываем флаги ошибок (например, EOF)
                            batch.seekg(0, std::ios::beg);
                        }



                        while (std::getline(batch, batch_line)) {

                            if (batch_pause == true) {
                                std::cout << "Нажмите Enter, чтобы продолжить.." << std::endl;
                                std::cin.get();
                            }
                            std::istringstream iss_batch(batch_line);

                            if (iss_batch >> batch_key) {
                                std::getline(iss_batch, batch_value);
                                std::istringstream iss_info(batch_value);

                                if (batch_key == "INI") {
                                    int length, width;
                                    iss_info >> length >> width;
                                    if (batch_pause == true) {
                                        std::cout << "Interface received 'INI' command with parameters: length=" + std::to_string(length) + ", width=" + std::to_string(width) << std::endl;
                                    }
                                    client.log("Interface received 'INI' command with parameters: length=" + std::to_string(length) + ", width=" + std::to_string(width));
                                    control.init(length, width);

                                }
                                else if (batch_key == "G") {
                                    double h, x0, y0, sigma_x, sigma_y;
                                    iss_info >> h >> x0 >> y0 >> sigma_x >> sigma_y;
                                    if (batch_pause == true) {
                                        std::cout << "Interface received 'g' command with parameters: h=" + std::to_string(h) + ", x0=" + std::to_string(x0) + ", y0=" + std::to_string(y0) +
                                            ", sigma_x=" + std::to_string(sigma_x) + ", sigma_y=" + std::to_string(sigma_y) << std::endl;
                                    }
                                    client.log("Interface received 'g' command with parameters: h=" + std::to_string(h) + ", x0=" + std::to_string(x0) + ", y0=" + std::to_string(y0) +
                                        ", sigma_x=" + std::to_string(sigma_x) + ", sigma_y=" + std::to_string(sigma_y));
                                    control.add_gauss(h, x0, y0, sigma_x, sigma_y);
                                    gaussi = true;

                                }
                                else if (batch_key == "GEN") {
                                    if (gaussi == false) {
                                        std::streampos saved_position = config.tellg();
                                        while (std::getline(config, line)) {
                                            iss.str(line);
                                            iss.clear();
                                            if (std::getline(iss, key, '=') && std::getline(iss, value)) {
                                                key = trim(key);
                                                value = trim(value);
                                                if (key == "Gauss_DEF") {
                                                    double h, x0, y0, sigma_x, sigma_y;
                                                    iss.clear();
                                                    iss.str(value);
                                                    iss >> h >> x0 >> y0 >> sigma_x >> sigma_y;
                                                    if (batch_pause == true) {
                                                        std::cout << "The default Gauss value is set: h=" + std::to_string(h) + ", x0=" + std::to_string(x0) + ", y0=" + std::to_string(y0) +
                                                            ", sigma_x=" + std::to_string(sigma_x) + ", sigma_y=" + std::to_string(sigma_y) << std::endl;
                                                    }
                                                    client.log("The default Gauss value is set: h=" + std::to_string(h) + ", x0=" + std::to_string(x0) + ", y0=" + std::to_string(y0) +
                                                        ", sigma_x=" + std::to_string(sigma_x) + ", sigma_y=" + std::to_string(sigma_y));
                                                    control.add_gauss(h, x0, y0, sigma_x, sigma_y);
                                                    gaussi = true;
                                                }
                                            }
                                        }
                                        config.seekg(saved_position);
                                    }

                                    if (batch_pause == true) {
                                        std::cout << "Interface received 'generate' command." << std::endl;
                                    }
                                    client.log("Interface received 'generate' command.");
                                    control.generate();

                                }
                                else if (batch_key == "GNU") {
                                    std::string filename;
                                    iss_info >> filename;

                                    if (batch_pause == true) {
                                        std::cout << "Interface received 'gnuplot' command with filename: " + filename << std::endl;
                                    }

                                    client.log("Interface received 'gnuplot' command with filename: " + filename);
                                    control.gnuplot(filename);

                                }
                                else if (batch_key == "BMP") {
                                    std::string filename;
                                    iss_info >> filename;
                                    if (batch_pause == true) {
                                        std::cout << "Interface received 'write bmp' command with filename:" + filename << std::endl;
                                    }
                                    client.log("Interface received 'write bmp' command with filename:" + filename);
                                    control.bmp_write_cntrl(filename, 2);

                                }
                                else if (batch_key == "TRIANG") {
                                    std::string filename;
                                    iss_info >> filename;
                                    if (batch_pause == true) {
                                        std::cout << "Interface received 'TRIANG' command with filename: " + filename << std::endl;
                                    }
                                    client.log("Interface received 'TRIANG' command with filename: " + filename);
                                    control.triangulateCmd(filename);
                                }

                                else if (batch_key == "RBMP") {
                                    std::string filename;
                                    iss_info >> filename;
                                    if (batch_pause == true) {
                                        std::cout << "Interface received 'read bmp' command with filename: " + filename << std::endl;
                                    }
                                    client.log("Interface received 'read bmp' command with filename: " + filename);
                                    control.bmp_read_cntrl(filename);
                                    gaussi = true;

                                }
                                else if (batch_key == "BIN") {
                                    int h;
                                    iss_info >> h;

                                    if (batch_pause == true) {
                                        std::cout << "Interface received 'BIN' command with h = " + std::to_string(h) << std::endl;
                                    }

                                    client.log("Interface received 'BIN' command with h = " + std::to_string(h));
                                    control.bin_cntrl(h);

                                }
                                else if (batch_key == "WAVE") {

                                    if (batch_pause == true) {
                                        std::cout << "Interface received 'WAVE' command with pixel noise:" + std::to_string(n) << std::endl;
                                    }

                                    client.log("Interface received 'WAVE' command with pixel noise:" + std::to_string(n));
                                    control.wave_cntrl(n);
                                }
                                else if (batch_key == "KMEANS") {
                                    int k, p;
                                    iss_info >> k >> p;

                                    if (batch_pause == true) {
                                        std::cout << "Interface received 'KMEANDS' coommand with k =" + std::to_string(k) << std::endl;
                                    }

                                    client.log("Interface received 'KMEANDS' coommand with k =" + std::to_string(k));
                                    control.kMeans_cntrl(k, p);
                                }
                                else if (batch_key == "ROAD") {
                                    int x1, y1, x2, y2;
                                    char comma;
                                    iss_info >> x1 >> comma >> y1 >> comma >> x2 >> comma >> y2;

                                    if (batch_pause) {
                                        std::cout << "Build Road: (" << x1 << "," << y1 << ") -> ("
                                            << x2 << "," << y2 << ")" << std::endl;
                                    }

                                    client.log("Obrabotka ROAD: (" + std::to_string(x1) + "," + std::to_string(y1) +
                                        ") -> (" + std::to_string(x2) + "," + std::to_string(y2) + ")");

                                    control.roadCmd(x1, y1, x2, y2);
                                }
                            }

                        }
                    }
                }
            }
        }



        // else if (input_type == "2") {
        //     std::string command;
        //     while (true) {
        //         std::cin >> command;

        //         if (command == "init") {
        //             int length, width;
        //             std::cin >> length >> width;
        //             client.log("Interface received 'init' command with parameters: length=" + std::to_string(length) + ", width=" + std::to_string(width));
        //             control.init(length, width);

        //         } else if (command == "g") {
        //             double h, x0, y0, sigma_x, sigma_y;
        //             std::cin >> h >> x0 >> y0 >> sigma_x >> sigma_y;
        //             client.log("Interface received 'g' command with parameters: h=" + std::to_string(h) + ", x0=" + std::to_string(x0) + ", y0=" + std::to_string(y0) +
        //                 ", sigma_x=" + std::to_string(sigma_x) + ", sigma_y=" + std::to_string(sigma_y));
        //             control.add_gauss(h, x0, y0, sigma_x, sigma_y);

        //         } else if (command == "generate") {
        //             client.log("Interface received 'generate' command.");
        //             control.generate();

        //         } else if (command == "gnuplot") {
        //             std::string filename;
        //             std::cin >> filename;
        //             client.log("Interface received 'gnuplot' command with filename: " + filename);
        //             control.gnuplot(filename);

        //         } else if (command == "save") {
        //             std::string filetype;
        //             std::cin >> filetype;
        //             if (filetype == "bmp") {
        //                 client.log("Interface received 'save bmp' command.");
        //                 control.bmp_write_cntrl();
        //             }
        //         } else if (command == "exit") {
        //             break;
        //         }
        //     }
        // }
        return 0;
    }
};

//212-Konchugarov-Timur

int main() {
    Interface interface;
    interface.run();
    return 0;
}
