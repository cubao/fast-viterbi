#include <pybind11/pybind11.h>

#include <limits>
#include <map>
#include <vector>
#include <unordered_map>
#include <iostream>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j) { return i + j; }

namespace py = pybind11;

namespace cubao {

// https://github.com/isl-org/Open3D/blob/88693971ae7a7c3df27546ff7c5b1d91188e39cf/cpp/open3d/utility/Helper.h#L71
template <typename T>
struct hash_vector {
    std::size_t operator()(const std::vector<T> &vec) const {
        size_t hash_seed = 0;
        for (auto elem: vec) {
            hash_seed ^= std::hash<T>()(elem) + 0x9e3779b9 +
                (hash_seed << 6) + (hash_seed >> 2);
        }
        return hash_seed;
    }
};


constexpr double neg_inf = -std::numeric_limits<double>::infinity();
struct FastViterbi {
    using LayerIndex = int;
    using CandidateIndex = int;
    using NodeIndex = std::tuple<LayerIndex, CandidateIndex>;
    FastViterbi(int K, int N, const std::map<std::tuple<NodeIndex, NodeIndex>, double> &scores) : K_(K), N_(N) {
        if (K == 0 || N < 2) {
            throw std::invalid_argument("invalid K, N = " + std::to_string(K) + ", " + std::to_string(N));
        }
        links_ = std::vector<std::vector<Links>>(N - 1, std::vector<Links>(K));
        for (auto &pair : scores) {
            auto &curr = std::get<0>(pair.first);
            auto &next = std::get<1>(pair.first);
            auto lidx0 = std::get<0>(curr);
            auto cidx0 = std::get<1>(curr);
            auto lidx1 = std::get<0>(next);
            auto cidx1 = std::get<1>(next);
            double score = pair.second;
            if (lidx0 < 0) {
                if (lidx1 == 0 && cidx1 < K) {
                    heads_.push_back({cidx1, score});
                    scores_[-1][-1][cidx1] = score;
                }
                continue;
            }
            if (lidx0 >= N || lidx1 != lidx0 + 1 || lidx1 >= N) {
                continue;
            }
            if (cidx0 < 0 || cidx0 >= K || cidx1 < 0 || cidx1 >= K) {
                continue;
            }
            links_[lidx0][cidx0].push_back({cidx1, score});
            scores_[lidx0][cidx0][cidx1] = score;
        }
    }

    std::vector<double> scores(const std::vector<int> &node_path) const {
        if (node_path.size() != N_) {
            return {};
        }
        std::vector<double> ret;
        ret.reserve(N_);
        double acc = scores_.at(-1).at(-1).at(node_path[0]);
        ret.push_back(acc);
        for (int n = 0; n < N_ - 1; ++n) {
            acc += scores_.at(n).at(node_path[n]).at(node_path[n+1]);
            ret.push_back(acc);
        }
        return ret;
    }

    std::vector<int> inference() const {
        // forward
        // backward
    }

    bool setup_roads(const std::vector<std::vector<int64_t>> &roads) {
        if (roads.size() != N_) {
            return false;
        }
        roads_ = std::vector<std::vector<int64_t>>(N_, std::vector<int64_t>(K_, (int64_t)-1));
        for (int n = 0; n < N_; ++n) {
            int K = roads[n].size();
            if (K > K_) {
                roads_.clear();
                std::cerr << "invalid road ids at #layer=" << n << ", #candidates=" << K << std::endl;
                return false;
            }
            for (int k = 0; k < K; ++k) {
                roads_[n][k] = roads[n][k];
            }
        }
        return true;
    }

    std::vector<int64_t> road_path(const std::vector<int> &node_path) const {
        if (node_path.empty()) {
            std::cerr << "empty node path!" << std::endl;
            return {};
        }
        if (roads_.empty()) {
            std::cerr << "roads not inited!" << std::endl;
            return {};
        }
        std::vector<int64_t> path;
        path.push_back(roads_[0][node_path[0]]);
        for (int n = 1; n < node_path.size(); ++n) {
            auto r = roads_[0][node_path[0]];
            if (r == path.back()) {
                continue;
            }
            path.push_back(r);
        }
        return path;
    }

    std::vector<int> inference(const std::vector<int> &path) const { return {}; }

  private:
    const int K_{-1};
    const int N_{-1};
    using Links = std::vector<std::pair<int, double>>;
    // head layer: cidx, score
    Links heads_;
    // tail layers, [[cidx (in next layer), score]]
    std::vector<std::vector<Links>> links_;
    // road ids, K * N
    std::vector<std::vector<int64_t>> roads_;
    // score map, lidx -> cidx -> next_cidx -> score
    std::unordered_map<int, std::unordered_map<int, std::unordered_map<int, double>>> scores_;
};
}  // namespace cubao

PYBIND11_MODULE(_core, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: scikit_build_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def(
        "subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
