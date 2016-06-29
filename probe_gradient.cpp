//
// Created by hannes on 2016-06-27.
//

#include <iostream>

#include <ceres/ceres.h>
#include <ceres/gradient_checker.h>
#include <sophus/se3.hpp>

#include <eigen3/Eigen/Dense>

#include <glog/logging.h>

#include "se3_spline.h"
#include "circularconstraint.h"
#include "local_parameterization_se3.hpp"

using std::cout;
using std::endl;

// Find a probing point by letting the optimizer do one iteration
std::vector<double*> find_probe_point(size_t num_eval) {
    std::vector<double*> probe_point;

    auto spline = create_zero_spline(5); // MUST be 5
    auto eval_times = create_eval_times(spline, num_eval);
    auto constraint = new CircularSplineConstraint(spline, eval_times);
    auto cost_func = new ceres::DynamicNumericDiffCostFunction<CircularSplineConstraint>(constraint);

    std::vector<double*> parameter_blocks;
    for (size_t i=0; i < spline.num_knots(); ++i) {
        parameter_blocks.push_back(spline.get_knot_data(i));
        cost_func->AddParameterBlock(Sophus::SE3d::num_parameters);
    }

    // Set residual count
    cost_func->SetNumResiduals(3*num_eval + 4*(num_eval - 1));

    ceres::Problem problem;
    // Local parameterization
    ceres::LocalParameterization *se3_parameterization = new Sophus::test::LocalParameterizationSE3;
    for (size_t i=0; i < spline.num_knots(); ++i) {
        problem.AddParameterBlock(spline.get_knot_data(i),
                                  Sophus::SE3d::num_parameters,
                                  se3_parameterization);
    }

    problem.AddResidualBlock(cost_func, NULL, parameter_blocks);

    ceres::Solver::Summary summary;

    ceres::Solver::Options solver_options;
    solver_options.max_num_iterations = 1;
    solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
    solver_options.minimizer_progress_to_stdout = true;
    solver_options.parameter_tolerance = 1e-4;

    ceres::Solver solver;
    solver.Solve(solver_options, &problem, &summary);

    for (int i=0; i < 5; ++i) {
        probe_point.push_back(spline.get_knot_data(i));
        //cout << spline.get_knot(i).matrix() << endl;
    }

    return probe_point;
}

template<int kStride, size_t num_eval>
void run_gradient_checker(std::vector<double*>& probe_point) {
    cout << "\n\n------------------------- CHECK kStride=" << kStride << " ---------------------------------------" << endl;
    typedef ceres::DynamicAutoDiffCostFunction<CircularSplineConstraint, kStride> CheckFuncType;
    auto spline = create_zero_spline(5); // MUST have 5 knots since the checker can handle at most 5 parameter blocks
    const size_t M = 3 * num_eval + 4 * (num_eval - 1);
    const size_t N = Sophus::SE3d::num_parameters;
    typedef ceres::GradientChecker<CheckFuncType, M, N, N, N, N, N> CheckerType;

    auto eval_times = create_eval_times(spline, num_eval);

    auto constraint = new CircularSplineConstraint(spline, eval_times);
    CheckFuncType* cost_func_check = new CheckFuncType(constraint);
    cost_func_check->SetNumResiduals(M);
    for (int i = 0; i < spline.num_knots(); ++i) {
        cost_func_check->AddParameterBlock(N);
    }

    CheckerType* checker;
    typename CheckerType::GradientCheckResults gradient_result;
    double errtol = 1e-10;
    bool result = checker->Probe(probe_point.data(), errtol, cost_func_check, &gradient_result);
    cout << "Error kStride=" << kStride << ": " << gradient_result.error_jacobians << endl;
}

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);

    cout << "\n============= GRADIENT CHECKER ====================" << endl;
    cout << "Reports the gradient_result.error_jacobians value for different values of kStride." << endl;
    cout << "Run with --logtostderr=1 to see log output from GradientChecker." << endl;

    const size_t num_eval = 8;
    auto probe_point = find_probe_point(num_eval);
    run_gradient_checker<1, num_eval>(probe_point);
    run_gradient_checker<2, num_eval>(probe_point);
    run_gradient_checker<3, num_eval>(probe_point);
    run_gradient_checker<4, num_eval>(probe_point);
    run_gradient_checker<5, num_eval>(probe_point);
    run_gradient_checker<6, num_eval>(probe_point);
    run_gradient_checker<7, num_eval>(probe_point);
    run_gradient_checker<8, num_eval>(probe_point);
}