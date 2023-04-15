/*
The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <bits/stdc++.h>
#include <omp.h>
#include <limits>
#include <chrono>
#include <cmath>
#include <sys/time.h>

#include "Functions.h"
#include "Functions.cpp"
#include "Jacobi.cpp"
#include "Gauss_Seidel.cpp"
#include "Finite_Differences.cpp"

#define THREAD_COUNT 2

int N;

typedef std::numeric_limits<double> dbl;


int main(void) {
    Finite_Differences finite_differences;
    Jacobi jacobi;
    Gauss_Seidel gauss_seidel;
    
    std::cout.precision(dbl::max_digits10);

    // Max iterations and TOL
    unsigned long max_iterations = 100;
    double TOL = 0.0000000001f;


    double h = 1.0/(pow(2.0, 2.0));  // h step
    double interval[2] = {0.0, 10 + h};   // Interval
    

    double phi = 10.0 * M_PI;
    double b = std::cos(2.0 * phi);
    auto u = [phi](double x) { return std::cos(phi * x); };
    auto f = [phi](double x) { return (3.0 * std::pow(phi, 2.0) * std::cos(phi * x)); };

    double boundary_conditions[2] = {1.0, b};   // Boundary conditions

    // Create x points in the interval
    std::vector<double> X_points = finite_differences.arange(interval[0], interval[1] + h, h);

    N = X_points.size();
    finite_differences.set_N(N);


    printf("N: %d\n", N);
    printf("h: %lf\n",h);


    // Create tridiagonal matrix
    double* A = finite_differences.create_tridiagonal_matrix(h, 
        [phi](double x) { return (2.0 * (phi * phi)); }, 
        [](double x) { return x; },
        X_points);

    
    // Set initial X solution vector
    double* X = create_vector(N-2); initialize_vector(X, 0.0, N-2);

    // Create right side vector F
    double* F = finite_differences.create_vector_F(h, f, boundary_conditions, X_points);

    // Find partitions
    //std::vector<std::vector<int>> partitions = create_variable_partitions(N-2, THREAD_COUNT);

    // Print partitions
    /* for(std::vector<int> partition: partitions) {
        printf("%d - %d\n",partition[0], partition[partition.size() - 1]);
    } */

    auto t_start = std::chrono::steady_clock::now();   // Start timer

    // Parallel Methods
    //int iteration = gauss_seidel.gauss_seidel_parallel(X, F, A, TOL, max_iterations, partitions, N-2, THREAD_COUNT);
    //int iteration = jacobi.jacobi_parallel(X, F, A, TOL, max_iterations, partitions, N-2, THREAD_COUNT);

    // Serial Methods
    int iteration = gauss_seidel.gauss_seidel_serial(X, F, A, TOL, max_iterations, N-2);
    //int iteration = jacobi.jacobi_serial(X, F, A, TOL, max_iterations, N-2);

    auto t_end = std::chrono::steady_clock::now();  // End timer

    // Calculate the time elapsed and print it
    std::cout << "Computation Time: " << std::chrono::duration<double>(t_end - t_start).count() << std::endl;    

    printf("Iterations: %d\n", iteration);   // Print iterations
    
    //printf("Number of partitions: %ld\n", partitions.size());   // Print Number of partitions

    // L2 Norm Error
    std::cout << "L2 Error: " << finite_differences.L2_NormError(u, h, X, X_points) << std::endl;

    // Max Norm Error
    //std::cout << "Max Error: " << finite_differences.Max_NormError(u, h, X, X_points) << std::endl;

    // Compare the solution to the exact solution
    //finite_differences.compare_solutions(u, X, X_points, 10);


    // Deallocate memory
    free(X);
    free(F);
    free(A);
    

    return 0;
}   



