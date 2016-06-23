# Minimal Sophus/Ceres autodiff vs numeric example

Using Ceres to minimize the cost function, it works fine using numeric derivatives but not automatic.

## Cost function
A B-spline in SE3 defines a camera trajectory.
Now we want to find the spline (its knots) such that

1. The world point `Xw = (5, 1, 3)` has local spline/camera coordinate `Xs = (0, 0, 1)`.
1. The angular velocity is a constant rotation around the x-axis.

I.e. The cameras trajectory is circular with `Xw` as its center point.

The spline is implemented according to the description in 

    Lovegrove, S., Patron-Perez, A., & Sibley, G. (2013). Spline Fusion: A continuous-time representation for visual-inertial fusion with application to rolling shutter cameras.
    In Procedings of the British Machine Vision Conference 2013 (pp. 93.1–93.11). British Machine Vision Association. http://doi.org/10.5244/C.27.93

## Building

Except for the usual things to build a Ceres project, you need the Sophus package for SE(3) representations.
Specifically, [the version by Steven Lovegrove](https://github.com/stevenlovegrove/Sophus)