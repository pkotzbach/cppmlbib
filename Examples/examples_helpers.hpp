#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>

std::vector<std::vector<std::string>> loadCSV(const std::string& filename) {
    std::vector<std::vector<std::string>> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::vector<std::string> row;
        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, ',')) {
            row.push_back(cell);
        }
        data.push_back(row);
    }

    return data;
}

std::vector<float> generate_random_data(size_t size) {
    std::vector<float> data(size);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0, 1.0);
    for (size_t i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
    return data;
}