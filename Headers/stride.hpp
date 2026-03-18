#pragma once
#include <vector>
#include <stdexcept>
#include <algorithm>

namespace stride {
    inline std::vector<int> calc_strides(const std::vector<int>& shape) {
        if (shape.empty()) return {};
        std::vector<int> strides(shape.size());
        strides.back() = 1;
        for (int i = (int)shape.size() - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    }

    inline std::vector<int> broadcast_shape(const std::vector<int>& a, const std::vector<int>& b) {
        int ndim = std::max((int)a.size(), (int)b.size());
        std::vector<int> result(ndim);
        for (int i = 0; i < ndim; ++i) {
            int ai = i < (int)a.size() ? a[a.size() - 1 - i] : 1;
            int bi = i < (int)b.size() ? b[b.size() - 1 - i] : 1;
            if (ai != bi && ai != 1 && bi != 1) throw std::invalid_argument("Incompatible broadcast shapes");
            result[ndim - 1 - i] = std::max(ai, bi);
        }
        return result;
    }

    inline std::vector<int> broadcast_strides(const std::vector<int>& shape, const std::vector<int>& strides, int ndim) {
        std::vector<int> result(ndim, 0);
        int offset = ndim - (int)shape.size();
        for (int i = 0; i < (int)shape.size(); ++i) {
            if (shape[i] != 1) result[offset + i] = strides[i];
        }
        return result;
    }

    inline int strided_idx(int shape_idx, const std::vector<int>& strides, const std::vector<int>& shape) {
        if (strides == shape) return shape_idx;
        
        int strided_idx = 0, temp;
        int current_idx = shape_idx;
        for (int i = (int)shape.size() - 1; i >= 0; --i) {
            temp = current_idx % shape[i];
            current_idx = current_idx / shape[i];
            strided_idx += temp * strides[i];
        }
        return strided_idx;
    }

    inline int strided_idx(const std::vector<int>& indices, const std::vector<int>& strides, const std::vector<int>& shape) {
        if (indices.size() != shape.size()) throw std::runtime_error("wrong indices vector");
        int strided_idx = 0;
        for (int i = (int)shape.size() - 1; i >= 0; --i) {
            if (indices[i] >= shape[i]) throw std::runtime_error("wrong indices vector");
            strided_idx += indices[i] * strides[i];
        }
        return strided_idx;
    }
}
