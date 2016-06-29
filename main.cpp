#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <ceres/gradient_checker.h>
#include <glog/logging.h>

#include "se3_spline.h"
#include "local_parameterization_se3.hpp"
#include "circularconstraint.h"

using std::cout;
using std::endl;


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
#if 0
    ceres::LocalParameterization *se3_parameterization = new ceres::AutoDiffLocalParameterization<Sophus::test::SophusSE3Plus,Sophus::SE3::num_parameters, Sophus::SE3::DoF>;
    cout << "Using AutoDiff Local Parameterization" << endl;
#else
    ceres::LocalParameterization *se3_parameterization = new Sophus::test::LocalParameterizationSE3;
        cout << "Using analytical Local Parameterization" << endl;
#endif
    for (size_t i=0; i < spline.num_knots(); ++i) {
        problem.AddParameterBlock(spline.get_knot_data(i),
                                  Sophus::SE3d::num_parameters,
                                  se3_parameterization);
    }

    problem.AddResidualBlock(cost_func, NULL, parameter_blocks);

    ceres::Solver::Summary summary;
    cout << "Solving..." << endl;
    ceres::Solve(solver_options, &problem, &summary);
    cout << summary.FullReport() << endl;
}

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);

    size_t num_knots = atoi(argv[1]);
    size_t num_eval = atoi(argv[2]);

    // Solver options
    ceres::Solver::Options solver_options;
    solver_options.max_solver_time_in_seconds = 30;
    solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
    solver_options.minimizer_progress_to_stdout = true;
    solver_options.parameter_tolerance = 1e-4;


    // 1) Numeric differentiation
    cout << "\n\n------------------------- NUMERIC ------------------------------------" << endl;
    UniformSpline<double> spline = create_zero_spline(num_knots);
    auto eval_times = create_eval_times(spline, num_eval);
    CircularSplineConstraint* constraint = new CircularSplineConstraint(spline, eval_times);
    auto cost_func_numeric = new ceres::DynamicNumericDiffCostFunction<CircularSplineConstraint>(constraint);
    run_solver(cost_func_numeric, spline, solver_options, num_eval);

    // 2) Auto differentiation
    cout << "\n\n------------------------- AUTO ---------------------------------------" << endl;
    spline = create_zero_spline(num_knots);
    constraint = new CircularSplineConstraint(spline, eval_times);
    const size_t kStride = 4;
    auto cost_func_auto = new ceres::DynamicAutoDiffCostFunction<CircularSplineConstraint, kStride>(constraint);
    run_solver(cost_func_auto, spline, solver_options, num_eval);

    cout << "Num knots: " << num_knots << "  evaluations: " << num_eval << endl;

    return 0;
}