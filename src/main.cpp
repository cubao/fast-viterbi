#include <pybind11/pybind11.h>

#include <limits>
#include <map>
#include <vector>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j) { return i + j; }

namespace py = pybind11;

namespace cubao {
constexpr double neg_inf = -std::numeric_limits<double>::infinity();
struct FastViterbi {
    using LayerIndex = int;
    using CandidateIndex = int;
    using NodeIndex = std::tuple<LayerIndex, CandidateIndex>;
    FastViterbi(int K, int N, const std::map<std::tuple<NodeIndex, NodeIndex>, double> &scores) : K_(K), N_(N) {
        if (K == 0 || N < 2) {
            return;
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
        }
    }

    std::vector<int> inference() const { return {}; }

    bool setup_roads(const std::vector<std::vector<int64_t>> &roads) {
        if (roads.size() != N_) {
            roads_.clear();
            return false;
        }
        roads_ = std::vector<std::vector<int64_t>>(N_, std::vector<int64_t>(K_, (int64_t)-1));
        for (int n = 0; n < N_; ++n) {
            int K = roads[n].size();
            if (K > K_) {
                roads_.clear();
                return false;
            }
            for (int k = 0; k < K; ++k) {
                roads_[n][k] = roads[n][k];
            }
        }
        return true;
    }

    std::vector<int> inference(const std::vector<int> &path) const { return {}; }

  private:
    const int K_{-1};
    const int N_{-1};
    using Links = std::vector<std::pair<int, double>>;
    Links heads_;
    std::vector<std::vector<Links>> links_;
    std::vector<std::vector<int64_t>> roads_;
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
