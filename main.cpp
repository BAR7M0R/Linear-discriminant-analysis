//
// Created by Dell on 06/04/2025.
//
#include "Percepton.hpp"
#include "Matrix.hpp"
#include <complex>
int main()
{
    Matrix<bool> m(2, 3, true);
    Matrix<double> n(2, 3, 1.0);

    Matrix<double> p(2, 3, 1.0);

    Matrix<double> q = p * n.transpose();
    Matrix<double> r = q * 4.0;
}