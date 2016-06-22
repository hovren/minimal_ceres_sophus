#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include <ceres/ceres.h>

#include "se3_spline.h"
#include "local_parameterization_se3.hpp"

using std::cout;
using std::endl;

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


UniformSpline<double> create_zero_spline() {
    UniformSpline<double> spline(1.0, 0.0);
    const size_t num_knots = 10;
    spline.zero_knots(num_knots);
    return spline;
}

template<typename T>
void run_solver(T cost_func, UniformSpline<double>& spline, ceres::Solver::Options& solver_options, size_t num_eval) {

    // One parameter block per spline knot
    std::vector<double*> parameter_blocks;
    for (size_t i=0; i < spline.num_knots(); ++i) {
        parameter_blocks.push_back(spline.get_knot_data(i));
        cost_func->AddParameterBlock(Sophus::SE3d::num_parameters);
    }

    // Set residual count
    cost_func->SetNumResiduals(3*num_eval + 4*(num_eval - 1));

    ceres::Problem problem;

    // Local parameterization
    for (size_t i=0; i < spline.num_knots(); ++i) {
        problem.AddParameterBlock(spline.get_knot_data(i),
                                  Sophus::SE3d::num_parameters,
                                  new Sophus::test::LocalParameterizationSE3);
    }

    problem.AddResidualBlock(cost_func, NULL, parameter_blocks);

    ceres::Solver::Summary summary;
    cout << "Solving..." << endl;
    ceres::Solve(solver_options, &problem, &summary);
    cout << summary.FullReport() << endl;
}

int main(int argc, char** argv) {

    // Solver options
    ceres::Solver::Options solver_options;
    solver_options.max_solver_time_in_seconds = 30;
    solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
    solver_options.minimizer_progress_to_stdout = true;
    solver_options.parameter_tolerance = 1e-4;


    // Evaluation times
    // Create dummy spline to be sure to use correct values
    UniformSpline<double> dummy_spline = create_zero_spline();
    double t = dummy_spline.min_time();
    size_t num_eval = 30;
    std::vector<double> eval_times;
    double delta = (dummy_spline.max_time() - dummy_spline.min_time()) / num_eval;
    for (size_t i=0; i < num_eval; ++i) {
        eval_times.push_back(t);
        t += delta;
    }


    // 1) Numeric differentiation
    cout << "\n\n------------------------- NUMERIC ------------------------------------" << endl;
    UniformSpline<double> spline = create_zero_spline();
    CircularSplineConstraint* constraint = new CircularSplineConstraint(spline, eval_times);
    auto cost_func_numeric = new ceres::DynamicNumericDiffCostFunction<CircularSplineConstraint>(constraint);
    run_solver(cost_func_numeric, spline, solver_options, num_eval);

    // 2) Auto differentiation
    cout << "\n\n------------------------- AUTO ---------------------------------------" << endl;
    spline = create_zero_spline();
    constraint = new CircularSplineConstraint(spline, eval_times);
    const size_t kStride = 4;
    auto cost_func_auto = new ceres::DynamicAutoDiffCostFunction<CircularSplineConstraint, kStride>(constraint);
    run_solver(cost_func_auto, spline, solver_options, num_eval);

    return 0;
}