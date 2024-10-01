#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <iostream>
#include <limits>
#include <map>
#include <unordered_map>
#include <vector>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j) { return i + j; }

namespace py = pybind11;
using namespace pybind11::literals;

template <typename T>
struct hash_vector {
    std::size_t operator()(const T &vec) const {
        size_t hash_seed = std::hash<size_t>()(vec.size());
        for (auto elem : vec) {
            hash_seed ^= std::hash<typename T::value_type>()(elem) + 0x9e3779b9 + (hash_seed << 6) + (hash_seed >> 2);
        }
        return hash_seed;
    }
};

struct Seq {
    const std::vector<int> node_path;
    const std::vector<int64_t> road_path;
    Seq() = default;
    Seq(std::vector<int> &&node_path, std::vector<int64_t> &&road_path)
        : node_path(std::move(node_path)), road_path(std::move(road_path)) {}

    bool operator==(const Seq &other) const {
        if (node_path.size() != other.node_path.size()) {
            return false;
        }
        for (int i = 0; i < node_path.size(); ++i) {
            if (node_path[i] != other.node_path[i]) {
                return false;
            }
        }
        return true;
    }

    Seq patch(const std::vector<int> &more_nodes, const std::vector<int64_t> &more_roads) const {
        auto nodes = node_path;
        nodes.reserve(nodes.size() + more_nodes.size());
        nodes.insert(nodes.end(), more_nodes.begin(), more_nodes.end());
        auto roads = road_path;
        roads.reserve(roads.size() + more_roads.size());
        roads.insert(roads.end(), more_roads.begin(), more_roads.end());
        return Seq(std::move(nodes), std::move(roads));
    }
};

using Roads = std::unordered_set<std::vector<int64_t>, hash_vector<std::vector<int64_t>>>;

namespace std {
template <>
struct hash<Seq> {
    size_t operator()(const Seq &s) const noexcept { return hash_vector<decltype(s.node_path)>()(s.node_path); }
};
}  // namespace std

inline bool __equals(const std::vector<int64_t> &seq0, int i0, int j0,  //
                     const std::vector<int64_t> &seq1, int i1, int j1) {
    if (j0 - i0 != j1 - i1) {
        return false;
    }
    while (i0 < j0) {
        if (seq0[i0] != seq1[i1]) {
            return false;
        }
        ++i0;
        ++i1;
    }
    return true;
}

namespace cubao {

// https://github.com/isl-org/Open3D/blob/88693971ae7a7c3df27546ff7c5b1d91188e39cf/cpp/open3d/utility/Helper.h#L71
constexpr double neg_inf = -std::numeric_limits<double>::infinity();
constexpr double pos_inf = std::numeric_limits<double>::infinity();

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
            acc += scores_.at(n).at(node_path[n]).at(node_path[n + 1]);
            ret.push_back(acc);
        }
        return ret;
    }

    std::tuple<double, std::vector<int>> inference() const {
        // forward
        // backward
        return std::make_tuple(pos_inf, std::vector<int>{});
    }

    bool setup_roads(const std::vector<std::vector<int64_t>> &roads) {
        if (roads.size() != N_) {
            std::cerr << "invalid roads, #layers=" << roads.size() << " != " << N_ << std::endl;
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

    bool setup_shortest_road_paths(const std::map<std::tuple<NodeIndex, NodeIndex>, std::vector<int64_t>> &sp_paths) {
        if (roads_.empty()) {
            std::cerr << "roads not setup" << std::endl;
            return false;
        }
        for (auto &pair : sp_paths) {
            auto &curr = std::get<0>(pair.first);
            auto &next = std::get<1>(pair.first);
            auto lidx0 = std::get<0>(curr);
            auto cidx0 = std::get<1>(curr);
            auto lidx1 = std::get<0>(next);
            auto cidx1 = std::get<1>(next);
            auto &path = pair.second;
            if (path.empty()) {
                std::cerr << "empty path" << std::endl;
                sp_paths_.clear();
                return false;
            }
            if (lidx0 < 0) {
                if (lidx1 == 0 && cidx1 < K_) {
                    if (path.size() == 1 && path[0] == roads_[0][cidx1]) {
                        sp_paths_[-1][-1][cidx1] = path;
                    } else {
                        std::cerr << "sp_path not match roads" << std::endl;
                        sp_paths_.clear();
                        return false;
                    }
                }
                continue;
            }
            if (lidx0 >= N_ || lidx1 != lidx0 + 1 || lidx1 >= N_) {
                continue;
            }
            if (cidx0 < 0 || cidx0 >= K_ || cidx1 < 0 || cidx1 >= K_) {
                continue;
            }
            if (path.front() == roads_[lidx0][cidx0] && path.back() == roads_[lidx1][cidx1]) {
                // scores_.at(lidx0).at(cidx0).at(cidx1)
                sp_paths_[lidx0][cidx0][cidx1] = path;
            } else {
                std::cerr << "sp_path not match roads" << std::endl;
                sp_paths_.clear();
                return false;
            }
        }
        return true;
    }

    std::vector<std::vector<int64_t>> all_road_paths() const {
        if (roads_.empty() || sp_paths_.empty()) {
            return {};
        }
        std::vector<Roads> prev_paths(K_);
        for (auto &pair : heads_) {
            auto cidx = pair.first;
            auto nidx = roads_[0][cidx];
            prev_paths[cidx].insert({nidx});
        }
        for (int n = 0; n < N_ - 1; ++n) {
            std::vector<Roads> curr_paths(K_);
            auto &paths = sp_paths_.at(n);
            auto &layer = links_[n];
            for (int i = 0; i < K_; ++i) {
                const auto &heads = prev_paths[i];
                if (heads.empty() || layer[i].empty()) {
                    continue;
                }
                auto &p = paths.at(i);
                for (auto &pair : layer[i]) {
                    int j = pair.first;
                    const auto &sig = p.at(j);
                    if (sig.size() == 1) {
                        curr_paths[j].insert(heads.begin(), heads.end());
                        continue;
                    }
                    for (auto copy : heads) {
                        copy.insert(copy.end(), sig.begin() + 1, sig.end());
                        curr_paths[j].insert(std::move(copy));
                    }
                }
            }
            prev_paths = std::move(curr_paths);
        }
        Roads ret;
        for (auto seqs : prev_paths) {
            for (auto &seq : seqs) {
                ret.insert(seq);
            }
        }
        return {ret.begin(), ret.end()};
    }

    std::tuple<double, std::vector<int>, std::vector<int64_t>> inference(const std::vector<int64_t> &road_path) const {
        if (roads_.empty() || sp_paths_.empty()) {
            return std::make_tuple(pos_inf, std::vector<int>{}, std::vector<int64_t>{});
        }
        if (road_path.empty()) {
            return std::make_tuple(pos_inf, std::vector<int>{}, std::vector<int64_t>{});
        }
        std::vector<std::unordered_set<Seq>> prev_paths(K_);
        for (auto &pair : heads_) {
            auto nid = pair.first;
            auto rid = roads_[0][nid];
            if (rid != road_path[0]) {
                continue;
            }
            prev_paths[nid].insert(Seq({nid}, {rid}));
        }
        const int N = road_path.size();
        for (int n = 0; n < N_ - 1; ++n) {
            std::vector<std::unordered_set<Seq>> curr_paths(K_);
            auto &paths = sp_paths_.at(n);
            auto &layer = links_[n];
            for (int i = 0; i < K_; ++i) {
                const auto &heads = prev_paths[i];
                if (heads.empty() || layer[i].empty()) {
                    continue;
                }
                auto &p = paths.at(i);
                for (auto &pair : layer[i]) {
                    int j = pair.first;
                    const auto &sig = p.at(j);
                    if (sig.size() == 1) {
                        for (auto &seq : heads) {
                            curr_paths[j].insert(seq.patch({j}, {}));
                        }
                        continue;
                    }
                    for (auto &seq : heads) {
                        auto &path = seq.road_path;
                        int I = path.size();
                        int J = I + sig.size() - 1;
                        if (!__equals(sig, 1, sig.size(), road_path, I, J)) {
                            continue;
                        }
                        curr_paths[j].insert(seq.patch({j}, std::vector<int64_t>(sig.begin() + 1, sig.end())));
                    }
                }
            }
            prev_paths = std::move(curr_paths);
        }

        std::vector<Seq> all_paths;
        for (auto &paths : prev_paths) {
            for (auto &path : paths) {
                if (path.road_path.size() != N) {
                    continue;
                }
                all_paths.push_back(std::move(path));
            }
        }
        if (all_paths.empty()) {
            return std::make_tuple(pos_inf, std::vector<int>{}, std::vector<int64_t>{});
        }

        double max_score = neg_inf;
        int best_path = -1;
        for (int i = 0; i < all_paths.size(); ++i) {
            double score = calc_score(all_paths[i]);
            if (score > max_score) {
                max_score = score;
                best_path = i;
            }
        }
        if (best_path < 0) {
            return std::make_tuple(pos_inf, std::vector<int>{}, std::vector<int64_t>{});
        }
        return std::make_tuple(max_score,                       //
                               all_paths[best_path].node_path,  //
                               all_paths[best_path].road_path);
    }

  private:
    const int K_{-1};
    const int N_{-1};
    using Links = std::vector<std::pair<int, double>>;
    // head layer: cidx, score
    Links heads_;
    // tail layers, [[cidx (in next layer), score]]
    std::vector<std::vector<Links>> links_;
    // score map, lidx -> cidx -> next_cidx -> score
    std::unordered_map<int, std::unordered_map<int, std::unordered_map<int, double>>> scores_;

    // road ids, K * N
    std::vector<std::vector<int64_t>> roads_;
    // sp_paths, lidx -> cidx -> next_cidx -> sp_path (road seq)
    std::unordered_map<int, std::unordered_map<int, std::unordered_map<int, std::vector<int64_t>>>> sp_paths_;

    double calc_score(const Seq &seq) const {
        auto &nodes = seq.node_path;
        if (nodes.empty()) {
            return neg_inf;
        }
        double score = scores_.at(-1).at(-1).at(nodes[0]);
        for (int n = 0; n < nodes.size() - 1; ++n) {
            int i = nodes[n];
            int j = nodes[n + 1];
            score += scores_.at(n).at(i).at(j);
        }
        return score;
    }
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

    using FastViterbi = cubao::FastViterbi;
    using NodeIndex = FastViterbi::NodeIndex;
    py::class_<FastViterbi>(m, "FastViterbi", py::module_local(), py::dynamic_attr())           //
        .def(py::init<int, int, const std::map<std::tuple<NodeIndex, NodeIndex>, double> &>(),  //
             "K"_a, "N"_a, "scores"_a,
             R"pbdoc(
             Initialize FastViterbi object.

             Args:
                 K (int): Number of nodes per layer.
                 N (int): Number of layers.
                 scores (dict): Scores for node transitions.
             )pbdoc")
        //
        .def("scores", &FastViterbi::scores, "node_path"_a,
             R"pbdoc(
             Get scores for a given node path.

             Args:
                 node_path (list): List of node indices representing a path.

             Returns:
                 float: Total score for the given path.
             )pbdoc")
        //
        .def("inference", py::overload_cast<>(&FastViterbi::inference, py::const_),
             R"pbdoc(
             Perform inference without a road path.

             Returns:
                 tuple: Best path and its score.
             )pbdoc")
        //
        .def("setup_roads", &FastViterbi::setup_roads, "roads"_a,
             R"pbdoc(
             Set up roads for the Viterbi algorithm.

             Args:
                 roads (list): List of road sequences.
             )pbdoc")
        .def("setup_shortest_road_paths", &FastViterbi::setup_shortest_road_paths, "sp_paths"_a,
             R"pbdoc(
             Set up shortest road paths.

             Args:
                 sp_paths (dict): Dictionary of shortest paths between nodes.
             )pbdoc")
        //
        .def("all_road_paths", &FastViterbi::all_road_paths,
             R"pbdoc(
             Get all road paths.

             Returns:
                 list: All road paths in the graph.
             )pbdoc")
        .def("inference", py::overload_cast<const std::vector<int64_t> &>(&FastViterbi::inference, py::const_),
             "road_path"_a, py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
             Perform inference with a given road path.

             Args:
                 road_path (list): List of road indices representing a path.

             Returns:
                 tuple: Best path and its score.
             )pbdoc")
        //
        ;

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
