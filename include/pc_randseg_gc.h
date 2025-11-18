#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

#include <geometrycentral/pointcloud/point_cloud.h>
#include <geometrycentral/pointcloud/point_position_geometry.h>

namespace pc_rwalker_gc {

using Points3d = Eigen::Matrix<double, Eigen::Dynamic, 3>;

struct SegResult {
    std::vector<std::vector<int>> segments;
    Eigen::MatrixXd probabilities;
    std::vector<int> labels;
};

namespace detail {

using namespace geometrycentral::pointcloud;

inline std::pair<std::unique_ptr<PointCloud>, std::unique_ptr<PointPositionGeometry>>
buildGeometry(const Points3d& xyz, int n_neighbors) {
    const size_t n = xyz.rows();
    auto cloud = std::make_unique<PointCloud>(n);
    PointData<geometrycentral::Vector3> positions(*cloud);

    for (Point p : cloud->points()) {
        const size_t i = p.getIndex();
        positions[p] = {xyz(i, 0), xyz(i, 1), xyz(i, 2)};
    }

    auto geom = std::make_unique<PointPositionGeometry>(*cloud, positions);
    geom->kNeighborSize = std::max(n_neighbors, 3);
    return {std::move(cloud), std::move(geom)};
}

// Maps point indices to their class labels and whether they are seeds
struct SeedInfo {
    std::vector<int> label;      // -1 if unseeded
    std::vector<int> seed_order;  // maps seeded points to column in seed matrix
    std::vector<int> unseed_order; // maps unseeded points to row in unseed matrix
    int n_seeds = 0;
};

inline SeedInfo buildSeedInfo(size_t n_points, const std::vector<std::vector<int>>& seeds) {
    SeedInfo info;
    info.label.assign(n_points, -1);
    info.seed_order.assign(n_points, -1);
    info.unseed_order.assign(n_points, -1);

    // Mark all seed points
    for (size_t c = 0; c < seeds.size(); ++c) {
        for (int i : seeds[c]) {
            if (info.label[i] >= 0) {
                throw std::invalid_argument("Seed point assigned to multiple classes");
            }
            info.label[i] = c;
            info.seed_order[i] = info.n_seeds++;
        }
    }

    // Map unseeded points
    int unseed_count = 0;
    for (size_t i = 0; i < n_points; ++i) {
        if (info.label[i] < 0) {
            info.unseed_order[i] = unseed_count++;
        }
    }

    return info;
}

inline Eigen::MatrixXd normalizeProbabilities(Eigen::MatrixXd& probs) {
    const int n_classes = probs.cols();
    for (int i = 0; i < probs.rows(); ++i) {
        double sum = probs.row(i).sum();
        if (sum > 0.0) {
            probs.row(i) /= sum;
        } else {
            probs.row(i).setConstant(1.0 / n_classes);
        }
    }
    return probs;
}

inline std::vector<int> extractLabels(const Eigen::MatrixXd& probs) {
    std::vector<int> labels(probs.rows());
    for (int i = 0; i < probs.rows(); ++i) {
        Eigen::Index argmax;
        probs.row(i).maxCoeff(&argmax);
        labels[i] = argmax;
    }
    return labels;
}

inline std::vector<std::vector<int>> groupByLabel(
    const std::vector<int>& labels,
    int n_classes,
    const SeedInfo& info
) {
    std::vector<std::vector<int>> segments(n_classes);
    for (size_t i = 0; i < labels.size(); ++i) {
        if (info.seed_order[i] < 0 && labels[i] >= 0 && labels[i] < n_classes) {
            segments[labels[i]].push_back(i);
        }
    }
    return segments;
}

} // namespace detail

inline std::vector<std::vector<int>> randomWalkerSegmentation(
    const Points3d& xyz,
    const std::vector<std::vector<int>>& seeds,
    int n_neighbors = 30
) {
    using namespace geometrycentral;

    // Build geometry and compute Laplacian
    auto [cloud, geom] = detail::buildGeometry(xyz, n_neighbors);
    geom->requireLaplacian();
    const auto& L = geom->laplacian;

    const size_t n = cloud->nPoints();
    const int k = seeds.size();
    const auto info = detail::buildSeedInfo(n, seeds);

    // Initialize probabilities: seeds have probability 1.0 for their class
    Eigen::MatrixXd probs = Eigen::MatrixXd::Zero(n, k);
    for (size_t c = 0; c < seeds.size(); ++c) {
        for (int i : seeds[c]) {
            probs(i, c) = 1.0;
        }
    }

    // Count unseeded points
    const int n_unseed = n - info.n_seeds;
    
    // If all points are seeded, return immediately
    if (n_unseed == 0) {
        SegResult result;
        result.probabilities = probs;
        result.labels = info.label;
        result.segments = seeds;
        return result.segments;
    }

    // Partition Laplacian: L = [L_uu  L_us]
    //                          [L_su  L_ss]
    // We solve: L_uu * x_u = -L_us * x_s
    std::vector<Eigen::Triplet<double>> uu_triplets, us_triplets;
    uu_triplets.reserve(L.nonZeros());
    us_triplets.reserve(L.nonZeros());

    for (int col = 0; col < L.outerSize(); ++col) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(L, col); it; ++it) {
            const int r = it.row();
            const int c = it.col();
            const bool r_unseed = (info.seed_order[r] < 0);
            const bool c_unseed = (info.seed_order[c] < 0);

            if (r_unseed && c_unseed) {
                uu_triplets.emplace_back(info.unseed_order[r], info.unseed_order[c], it.value());
            } else if (r_unseed && !c_unseed) {
                us_triplets.emplace_back(info.unseed_order[r], info.seed_order[c], it.value());
            }
        }
    }

    Eigen::SparseMatrix<double> L_uu(n_unseed, n_unseed);
    Eigen::SparseMatrix<double> L_us(n_unseed, info.n_seeds);
    L_uu.setFromTriplets(uu_triplets.begin(), uu_triplets.end());
    L_us.setFromTriplets(us_triplets.begin(), us_triplets.end());

    // Build seed indicator matrix
    Eigen::MatrixXd seed_probs = Eigen::MatrixXd::Zero(info.n_seeds, k);
    for (size_t i = 0; i < n; ++i) {
        if (info.seed_order[i] >= 0) {
            seed_probs(info.seed_order[i], info.label[i]) = 1.0;
        }
    }

    // Solve the linear system
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(L_uu);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Failed to factorize Laplacian");
    }

    Eigen::MatrixXd unseed_probs = solver.solve(-L_us * seed_probs);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Failed to solve linear system");
    }

    // Copy unseeded probabilities back
    for (size_t i = 0; i < n; ++i) {
        if (info.unseed_order[i] >= 0) {
            probs.row(i) = unseed_probs.row(info.unseed_order[i]);
        }
    }

    // Normalize and extract labels
    detail::normalizeProbabilities(probs);
    auto labels = detail::extractLabels(probs);

    // Build final segments (seeds + assigned points)
    SegResult result;
    result.probabilities = probs;
    result.labels = labels;
    result.segments = seeds;  // Start with seed points
    auto new_segments = detail::groupByLabel(labels, k, info);
    for (int c = 0; c < k; ++c) {
        result.segments[c].insert(
            result.segments[c].end(),
            new_segments[c].begin(),
            new_segments[c].end()
        );
    }

    return result.segments;
}

} // namespace pc_rwalker_gc
