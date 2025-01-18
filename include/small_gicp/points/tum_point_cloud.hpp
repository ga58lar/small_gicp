// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <memory>
#include <vector>
#include <Eigen/Core>
#include <small_gicp/points/traits.hpp>

namespace small_gicp {

/**
 * @brief Point cloud
 */
struct TUMPointCloud {
public:
  using Ptr = std::shared_ptr<TUMPointCloud>;
  using ConstPtr = std::shared_ptr<const TUMPointCloud>;

  /// @brief Constructor
  TUMPointCloud() {}

  /// @brief Constructor
  /// @param points  Points to initialize the point cloud
  template <typename T, int D, typename Allocator>
  explicit TUMPointCloud(const std::vector<Eigen::Matrix<T, D, 1>, Allocator>& points) {
    this->resize(points.size());
    for (size_t i = 0; i < points.size(); i++) {
      this->point(i) << points[i].template cast<double>().template head<3>(), 1.0;
    }
  }

  /// @brief Destructor
  ~TUMPointCloud() {}

  /// @brief Number of points.
  size_t size() const { return points.size(); }

  /// @brief Check if the point cloud is empty.
  bool empty() const { return points.empty(); }

  /// @brief Resize point/normal/cov/label buffers.
  /// @param n  Number of points
  void resize(size_t n) {
    points.resize(n);
    normals.resize(n);
    covs.resize(n);
    labels.resize(n);
  }

  /// @brief Get i-th point.
  Eigen::Vector4d& point(size_t i) { return points[i]; }

  /// @brief Get i-th normal.
  Eigen::Vector4d& normal(size_t i) { return normals[i]; }

  /// @brief Get i-th covariance.
  Eigen::Matrix4d& cov(size_t i) { return covs[i]; }

  /// @brief Get i-th label.
  int& label(size_t i) { return labels[i]; }

  /// @brief Get i-th point (const).
  const Eigen::Vector4d& point(size_t i) const { return points[i]; }

  /// @brief Get i-th normal (const).
  const Eigen::Vector4d& normal(size_t i) const { return normals[i]; }

  /// @brief Get i-th covariance (const).
  const Eigen::Matrix4d& cov(size_t i) const { return covs[i]; }

  /// @brief Get i-th label (const).
  const int& label(size_t i) const { return labels[i]; }

public:
  std::vector<Eigen::Vector4d> points;   ///< Point coordinates (x, y, z, 1)
  std::vector<Eigen::Vector4d> normals;  ///< Point normals (nx, ny, nz, 0)
  std::vector<Eigen::Matrix4d> covs;     ///< Point covariances (3x3 matrix) + zero padding
  std::vector<int> labels;              ///< Point labels
};

namespace traits {

template <>
struct Traits<TUMPointCloud> {
  using Points = TUMPointCloud;

  static size_t size(const Points& points) { return points.size(); }

  static bool has_points(const Points& points) { return !points.points.empty(); }
  static bool has_normals(const Points& points) { return !points.normals.empty(); }
  static bool has_covs(const Points& points) { return !points.covs.empty(); }
  static bool has_labels(const Points& points) { return !points.labels.empty(); }

  static const Eigen::Vector4d& point(const Points& points, size_t i) { return points.point(i); }
  static const Eigen::Vector4d& normal(const Points& points, size_t i) { return points.normal(i); }
  static const Eigen::Matrix4d& cov(const Points& points, size_t i) { return points.cov(i); }
  static const int& label(const Points& points, size_t i) { return points.label(i); }

  static void resize(Points& points, size_t n) { points.resize(n); }
  static void set_point(Points& points, size_t i, const Eigen::Vector4d& pt) { points.point(i) = pt; }
  static void set_normal(Points& points, size_t i, const Eigen::Vector4d& n) { points.normal(i) = n; }
  static void set_cov(Points& points, size_t i, const Eigen::Matrix4d& cov) { points.cov(i) = cov; }
  static void set_label(Points& points, size_t i, const int& l) { points.label(i)= l; }
};

}  // namespace traits

}  // namespace small_gicp
