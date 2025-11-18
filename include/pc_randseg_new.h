#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>
#include <omp.h>

#include <geometrycentral/pointcloud/point_cloud.h>
#include <geometrycentral/pointcloud/point_cloud_heat_solver.h>
#include <geometrycentral/pointcloud/point_position_geometry.h>

namespace pc_rwalker_new {

using Points3d = Eigen::Matrix<double, Eigen::Dynamic, 3>;

struct SegResult {
    std::vector<std::vector<int>> segments;
    Eigen::MatrixXd probabilities;
    std::vector<int> labels;
};

namespace detail {

using namespace geometrycentral;
using namespace geometrycentral::pointcloud;

inline std::tuple<std::unique_ptr<PointCloud>, std::unique_ptr<PointPositionGeometry>>
buildGeometry(const Points3d &xyz, unsigned int n_neighbors) {
    // if (xyz.cols() != 3) {
    //     throw std::invalid_argument("Point cloud input must have exactly three columns.");
    // }
    
    const size_t n_points = static_cast<size_t>(xyz.rows());
    auto cloud = std::make_unique<PointCloud>(n_points);
    PointData<Vector3> positions(*cloud);

    // Convert the xyz points to GeometryCentral's Vector3
    for (Point p : cloud->points()) {
        const size_t idx = p.getIndex();
        positions[p] = Vector3{xyz(idx, 0), xyz(idx, 1), xyz(idx, 2)};
    }

    auto geom = std::make_unique<PointPositionGeometry>(*cloud, positions);
    geom->kNeighborSize = std::max<unsigned int>(n_neighbors, 3U);
    return {std::move(cloud), std::move(geom)};
}


struct SeedLayout {
    std::vector<int> point_to_seed_col; // size = n_points, -1 if not seeded
    std::vector<int> point_to_class;    // size = n_points, -1 if not seeded
    std::vector<int> unmarked_order;    // size = n_points, -1 if seeded
    int n_marked = 0;
    int n_unmarked = 0;
};

inline SeedLayout buildSeedLayout(size_t n_points, const std::vector<std::vector<int>> &seed_indices) {
    SeedLayout layout;
    layout.point_to_seed_col.assign(n_points, -1);
    layout.point_to_class.assign(n_points, -1);
    layout.unmarked_order.assign(n_points, -1);

    int seed_col = 0;
    for (size_t class_idx = 0; class_idx < seed_indices.size(); ++class_idx) {
        for (int idx : seed_indices[class_idx]) {
            const size_t pt = static_cast<size_t>(idx);
            if (layout.point_to_class[pt] >= 0) {
                throw std::invalid_argument("A seed point cannot belong to multiple classes.");
            }
            layout.point_to_class[pt] = static_cast<int>(class_idx);
            layout.point_to_seed_col[pt] = seed_col++;
        }
    }

    layout.n_marked = seed_col;

    int unmarked_counter = 0;
    for (size_t idx = 0; idx < n_points; ++idx) {
        if (layout.point_to_seed_col[idx] < 0) {
            layout.unmarked_order[idx] = unmarked_counter++;
        }
    }
    layout.n_unmarked = unmarked_counter;
    return layout;
}

inline std::vector<std::vector<int>> assembleSegments(
    const std::vector<int> &labels,
    const std::vector<std::vector<int>> &seed_indices,
    const std::vector<int> &point_to_seed_col
) {
    std::vector<std::vector<int>> segments = seed_indices;
    for (size_t idx = 0; idx < labels.size(); ++idx) {
        if (point_to_seed_col[idx] >= 0) {
            continue; // already part of the seed list
        }
        const int label = labels[idx];
        if (label < 0 || static_cast<size_t>(label) >= segments.size()) {
            continue;
        }
        segments[label].push_back(static_cast<int>(idx));
    }
    return segments;
}

} // namespace detail

inline SegResult _random_segmentation(
    const Points3d &xyz,
    const std::vector<std::vector<int>> &seed_indices,
    const int n_neighbors
) {
    using namespace geometrycentral;
    using namespace geometrycentral::pointcloud;

    auto [cloud, geom] = detail::buildGeometry(xyz, n_neighbors);
    auto seed_layout = detail::buildSeedLayout(cloud->nPoints(), seed_indices);

    geom->requireLaplacian();
    const Eigen::SparseMatrix<double> &laplacian = geom->laplacian;

    const size_t n_points = cloud->nPoints();
    const int n_classes = static_cast<int>(seed_indices.size());
    Eigen::MatrixXd probabilities = Eigen::MatrixXd::Zero(n_points, n_classes);

    // Initialize the probabilities matrix with 1.0 for the seed points
    for (size_t class_idx = 0; class_idx < seed_indices.size(); ++class_idx) {
        for (int idx : seed_indices[class_idx]) {
            probabilities(idx, static_cast<int>(class_idx)) = 1.0;
        }
    }

    if (seed_layout.n_unmarked == 0) {
        SegResult result;
        result.labels.assign(n_points, 0);
        for (size_t idx = 0; idx < n_points; ++idx) {
            result.labels[idx] = seed_layout.point_to_class[idx];
        }
        result.segments = detail::assembleSegments(result.labels, seed_indices, seed_layout.point_to_seed_col);
        result.probabilities = probabilities;
        return result;
    }

    std::vector<Eigen::Triplet<double>> triplet_uu;
    std::vector<Eigen::Triplet<double>> triplet_us;
    triplet_uu.reserve(static_cast<size_t>(laplacian.nonZeros()));
    triplet_us.reserve(static_cast<size_t>(laplacian.nonZeros()));

    for (int col = 0; col < laplacian.outerSize(); ++col) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(laplacian, col); it; ++it) {
            const int row_idx = static_cast<int>(it.row());
            const int col_idx = static_cast<int>(it.col());
            const double value = it.value();

            const bool row_unmarked = seed_layout.point_to_seed_col[row_idx] < 0;
            const bool col_unmarked = seed_layout.point_to_seed_col[col_idx] < 0;

            if (row_unmarked && col_unmarked) {
                triplet_uu.emplace_back(
                    seed_layout.unmarked_order[row_idx],
                    seed_layout.unmarked_order[col_idx],
                    value
                );
            } else if (row_unmarked && !col_unmarked) {
                triplet_us.emplace_back(
                    seed_layout.unmarked_order[row_idx],
                    seed_layout.point_to_seed_col[col_idx],
                    value
                );
            }
        }
    }

    Eigen::SparseMatrix<double> Luu(seed_layout.n_unmarked, seed_layout.n_unmarked);
    Eigen::SparseMatrix<double> Lus(seed_layout.n_unmarked, seed_layout.n_marked);
    Luu.setFromTriplets(triplet_uu.begin(), triplet_uu.end());
    Lus.setFromTriplets(triplet_us.begin(), triplet_us.end());

    Eigen::MatrixXd seed_indicator = Eigen::MatrixXd::Zero(seed_layout.n_marked, n_classes);
    for (size_t idx = 0; idx < n_points; ++idx) {
        const int seed_col = seed_layout.point_to_seed_col[idx];
        if (seed_col >= 0) {
            const int class_idx = seed_layout.point_to_class[idx];
            seed_indicator(seed_col, class_idx) = 1.0;
        }
    }

    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(Luu);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Failed to factorize Laplacian submatrix for segmentation.");
    }

    Eigen::MatrixXd rhs = -(Lus * seed_indicator);
    Eigen::MatrixXd interior_prob = solver.solve(rhs);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Failed to solve Laplacian system for segmentation.");
    }

//#pragma omp parallel for schedule(dynamic) if(n_points > 10000)
    for (size_t idx = 0; idx < n_points; ++idx) {
        const int row = seed_layout.unmarked_order[idx];
        if (row >= 0) {
            probabilities.row(static_cast<int>(idx)) = interior_prob.row(row);
        }
    }

// #pragma omp parallel for schedule(static) if(n_points > 10000)
    for (int row = 0; row < probabilities.rows(); ++row) {
        double row_sum = probabilities.row(row).sum();
        if (row_sum <= 0.0) {
            probabilities.row(row).setConstant(1.0 / static_cast<double>(n_classes));
        } else {
            probabilities.row(row) /= row_sum;
        }
    }

    std::vector<int> labels(n_points, -1);
// #pragma omp parallel for schedule(static) if(n_points > 10000)
    for (size_t idx = 0; idx < n_points; ++idx) {
        Eigen::Index argmax;
        probabilities.row(static_cast<int>(idx)).maxCoeff(&argmax);
        labels[idx] = static_cast<int>(argmax);
    }

    SegResult result;
    result.probabilities = probabilities;
    result.labels = labels;
    result.segments = detail::assembleSegments(labels, seed_indices, seed_layout.point_to_seed_col);

    // Ensure geometry resources are released before the point cloud to avoid dangling callbacks.
    geom.reset();
    cloud.reset();

    return result;
}

inline std::vector<std::vector<int>> randomWalkerSegmentation(
    const Points3d &xyz,
    const std::vector<std::vector<int>> &seed_indices,
    const int n_neighbors,
    const double sigma1 = 1.0,
    const double sigma2 = 1.0,
    const double min_weight = 0.0001,
    const int n_proc = -1
) {
    (void)sigma1;
    (void)sigma2;
    (void)min_weight;
    (void)n_proc;

    auto result = _random_segmentation(xyz, seed_indices, n_neighbors);
    return result.segments;
}

} // namespace pc_rwalker_new