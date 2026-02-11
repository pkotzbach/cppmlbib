#pragma once

#include <vector>
#include <string>

std::vector<double> _matmul(const std::vector<double> A, const std::vector<double> B, int K, int X, int Y, std::string device="cpu");
