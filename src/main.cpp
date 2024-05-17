#include <pybind11/pybind11.h>
#include <vector>
#include <unordered_map>

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
    FastViterbi(int K, int N, const std::unordered_map<std::tuple<NodeIndex, NodeIndex>, double> &scores) {
        if (K == 0 || N < 2) {
            return;
        }
        heads_ = std::vector<double>(K, neg_inf);
        links_ = std::vector<std::vector<Links>>(N - 1, std::vector<Links>(K));
        for (auto &pair: scores) {
            auto &curr = std::get<0>(pair.first);
            auto &next = std::get<1>(pair.first);
            auto lidx0 = std::get<0>(curr);
            auto cidx0 = std::get<1>(curr);
            auto lidx1 = std::get<0>(next);
            auto cidx1 = std::get<1>(next);
            double score = pair.second;
            if (lidx0 < 0) {
                if (lidx1 == 0 && cidx1 < K) {
                    heads_[cidx1] = score;
                }
                continue;
            }
            if (lidx0 < N && cidx0 < K && lidx1 == lidx0 + 1 && lidx1 < N && cidx1 < K) {
                links[lidx0][cidx0].push_back({cidx1, score});
            }
        }
    }
    private:
        using Links = std::vector<std::pair<int, double>>;
        std::vector<double> heads_;
        std::vector<std::vector<Links>> links_;
};
}

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
