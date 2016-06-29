//
// Created by hannes on 2016-06-29.
//

#ifndef MINIMAL_SOPHUS_CERES_CIRCULARCONSTRAINT_H
#define MINIMAL_SOPHUS_CERES_CIRCULARCONSTRAINT_H

#include <vector>

#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>
#include <sophus/se3.hpp>

#include "se3_spline.h"

struct CircularSplineConstraint {
    CircularSplineConstraint(UniformSpline<double>& spline, std::vector<double>& eval_times)
            : eval_times_(eval_times), spline_dt_(spline.get_dt()), spline_offset_(spline.get_offset()), num_knots_(spline.num_knots()){
        // Constant angle between evaluations
        expected_angular_difference_ = 2 * M_PI / eval_times_.size();
    };

    template<typename T>
    bool operator()(T const* const* parameters, T* residuals) const {
        // Map spline
        UniformSpline<T> spline(spline_dt_, spline_offset_);
        for (size_t i=0; i < num_knots_; ++i) {
            spline.add_knot((T*) &parameters[i][0]);
        }

        Eigen::Matrix<T, 3, 1> landmark(T(5.0), T(1.0), T(3.0));
        Eigen::Matrix<T, 3, 1> expected(T(0.0), T(0.0), T(1.0));


        // Distance to center
        Sophus::SE3Group<T> P;
        Eigen::Matrix<T, 4, 4> dP, d2P;
        for (size_t i=0; i < eval_times_.size(); ++i) {
            spline.evaluate(T(eval_times_[i]), P, dP, d2P);
            Eigen::Matrix<T, 3, 1> X_spline = P.inverse() * landmark;
            Eigen::Matrix<T, 3, 1> diff = X_spline - expected;
            residuals[3*i + 0] = diff(0);
            residuals[3*i + 1] = diff(1);
            residuals[3*i + 2] = diff(2);
        }

        const size_t poff = 3 * eval_times_.size();

        // Constant angular difference
        for (size_t i=1; i < eval_times_.size(); ++i) {
            SE3Group<T> P0, P1, delta;
            spline.evaluate(T(eval_times_[i-1]), P0, dP, d2P);
            spline.evaluate(T(eval_times_[i]), P1, dP, d2P);
            delta = P0.inverse() * P1;
            Eigen::Quaternion<T> q = delta.unit_quaternion();
            residuals[poff + 4*(i - 1) + 0] = T(5.0) * (q.w() - T(cos(0.5 * expected_angular_difference_)));
            residuals[poff + 4*(i - 1) + 1] = T(5.0) * (q.x() - T(sin(0.5 * expected_angular_difference_)));
            residuals[poff + 4*(i - 1) + 2] = T(5.0) * q.y(); //- 0;
            residuals[poff + 4*(i - 1) + 3] = T(5.0) * q.z(); //- 0;
        }

        return true;
    }

    std::vector<double>& eval_times_;
    double spline_offset_;
    double spline_dt_;
    double expected_angular_difference_;
    size_t num_knots_;
};


UniformSpline<double> create_zero_spline(size_t num_knots) {
    UniformSpline<double> spline(1.0, 0.0);
    spline.zero_knots(num_knots);
    return spline;
}

std::vector<double> create_eval_times(UniformSpline<double>& spline, size_t num_eval) {
    std::vector<double> eval_times;
    double t = spline.min_time();
    double delta = (spline.max_time() - spline.min_time()) / num_eval;
    for (size_t i=0; i < num_eval; ++i) {
        eval_times.push_back(t);
        t += delta;
    }

    return eval_times;
}

#endif //MINIMAL_SOPHUS_CERES_CIRCULARCONSTRAINT_H
