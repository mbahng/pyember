#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>

class Tensor {
private:
    std::vector<double> data;

public:
    Tensor(const std::vector<double>& input_data) : data(input_data) {}

    size_t size() const {
        return data.size();
    }

    double& operator[](size_t index) {
        if (index >= data.size()) {
            throw std::out_of_range("Index out of range");
        }
        return data[index];
    }

    const double& operator[](size_t index) const {
        if (index >= data.size()) {
            throw std::out_of_range("Index out of range");
        }
        return data[index];
    }

    std::string str() const {
        std::ostringstream oss;
        oss << "Tensor(";
        for (size_t i = 0; i < data.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << data[i];
        }
        oss << ")";
        return oss.str();
    }
};

double add(double x, double y); 

