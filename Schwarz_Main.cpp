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
#include "Schwarz_DD.cpp"
#include "Finite_Differences.cpp"

#define THREAD_COUNT 4

int N;

typedef std::numeric_limits<double> dbl;


int main(void) {
    Finite_Differences finite_differences;
    Jacobi jacobi;
    Gauss_Seidel gauss_seidel;
    Schwarz_DD schwarz_dd(jacobi, gauss_seidel);
    
    std::cout.precision(dbl::max_digits10);

    // Max iterations and TOL
    unsigned long max_iterations = 1000;
    double TOL = 0.000000001f;


    double h = 1.0/(pow(2.0, 8.0));  // h step
    double interval[2] = {0.0, 10 + h};   // Interval
    

    double phi = 10.0 * M_PI;
    double b = std::cos(2.0 * phi);
    auto u = [phi](double x) { return std::cos(phi * x); };
    auto f = [phi](double x) { return (3.0 * std::pow(phi, 2.0) * std::cos(phi * x)); };

    double boundary_conditions[2] = {1.0, b};   // Boundary conditions

    // Create x points in the interval
    std::vector<double> X_points = finite_differences.arange(interval[0], interval[1] + h, h);

    N = X_points.size(); finite_differences.set_N(N);

    printf("N: %d\n", N);
    printf("h: %lf\n",h);



    int DDBounds[THREAD_COUNT][2] = {
        {0,630},
        {640,1280},
        {1260,2000},
        {1980,2559},
    };


    int block_size = (N-2) / THREAD_COUNT;
    int number_of_blocks;

    try {
        if(block_size == 0) throw std::overflow_error("");

        number_of_blocks = pow((N-2)/block_size, 2);
    }
    catch(const std::overflow_error& e) {
        printf("\nCheck the value of the variable <THREAD_COUNT>. Division by zero attempted.\n");
        std::cerr << e.what() << '\n';
        return 1;

    }

    // Create tridiagonal matrix
    double* A = finite_differences.create_tridiagonal_matrix(h, 
        [phi](double x) { return (2.0 * (phi * phi)); }, 
        [](double x) { return x; },
        X_points);


    // Set initial X solution vector
    double* X = create_vector(N-2); initialize_vector(X, 0.0, N-2);

    // Create right side vector F
    double* F = finite_differences.create_vector_F(h, f, boundary_conditions, X_points);
    

    // Create non overlapping blocks
    /* double* A_Blocks = create_non_overlapping_blocks(A, N-2, THREAD_COUNT, block_size, 1);

    if(A_Blocks == nullptr) {
        return 0;
    } */


    // Create the main inverse diagonal block of A
    //double* inv_main_block = create_inverse_diagonal_block(A_Blocks, THREAD_COUNT, (N-2)/THREAD_COUNT);



    auto t_start = std::chrono::steady_clock::now();   // Start timer

    // Swartz Methods
    //int iterations = schwarz_dd.ABJASM(X, TOL, max_iterations, N-2, THREAD_COUNT, block_size, A_Blocks, inv_main_block, F);
    //int iterations = schwarz_dd.MBJASM(X, TOL, max_iterations, N-2, THREAD_COUNT, block_size, A, F);
    int iterations = schwarz_dd.MBJRASM(X, TOL, max_iterations, N-2, THREAD_COUNT, DDBounds, A, F);
    
    auto t_end = std::chrono::steady_clock::now();  // End timer

    // Calculate the time elapsed and print it
    std::cout << "Computation Time: " << std::chrono::duration<double>(t_end - t_start).count() << std::endl;    

    printf("Iteration: %d\n", iterations);   // Print iterations
    
    // L2 Norm Error
    std::cout << "L2 Error: " << finite_differences.L2_NormError(u, h, X, X_points) << std::endl;

    // Compare the solution to the exact solution
    //finite_differences.compare_solutions(u, X, X_points, 10);


    // Deallocate memory
    free(X);
    free(F);
    free(A);
    //free(A_Blocks);
    //free(inv_main_block);

    

    return 0;
}
