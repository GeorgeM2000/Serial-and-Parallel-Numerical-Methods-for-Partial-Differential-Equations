class Finite_Differences {
    int N;

    public:

        Finite_Differences() {}

        // Set the dimension of the linear system.
        void set_N(int n) {
            N = n - 2;
        }


        // Create the tridiagonal matrix.
        template<typename T>
        double* create_tridiagonal_matrix(double h, T&& q, std::function<double(double)> p, std::vector<double> X_points) {
            double* A = create_matrix(N, N);

            h = h * h;
            for(int i=0; i<N; i++) {
                for(int j=0; j<N; j++) {
                    if(i == j) {
                        A[i * N + j] = (2.0 * 1.0) + h*q(X_points[i+1]);
                    } else if(i == j-1 || i == j+1) {
                        A[i * N + j] = -1.0 * 1.0;
                    } else {
                        A[i * N + j] = 0.0;
                    }
                }
            }
            return A;
        }

        // Arange the x points from a starting point to an end point by a step.
        std::vector<double> arange(double start, double end, double step) {
            std::vector<double> x_points;
            for(double s=start; s<end; s+=step) {
                x_points.push_back(s);
            }
        
            return x_points;
        }

        // Create the vector F(the right hand side of the linear system).
        template<typename T>
        double* create_vector_F(double h, T&& f, double* boundary_conditions, std::vector<double> X_points) {
            double* F = create_vector(N);
            h = h * h;

            F[0] = h * f(X_points[1]) + boundary_conditions[0];
            F[N-1] = h * f(X_points[N]) + boundary_conditions[1];

            for(int i=1; i<N-1; i++){
                F[i] = h * f(X_points[i+1]);
            }

            return F;
        }


        template<typename T>
        double L2_NormError(T&& u, double h, double* X, std::vector<double> X_points) {
            double error = 0.0;
            for(int i=0; i<N; i++) {
                error += pow(fabs(u(X_points[i+1]) - X[i]), 2.0);
            }
            error = h * error;
            
            return pow(error, 1.0/2.0);
        }

        template<typename T>
        double Max_NormError(T&& u, double h, double* X, std::vector<double> X_points) {
            double local_Error[N];
            for(int i=0; i<N-2; i++) {
                local_Error[i] = fabs(X[i] - u(X_points[i+1]));
            }
            double* max = std::max_element(local_Error, local_Error+N);

            return *max;
        }


        // Compare the approximate solution with the exact solution.
        template<typename T>
        void compare_solutions(T&& u, double* X, std::vector<double> x_points, int limit) {
            std::cout.precision(std::numeric_limits<double>::max_digits10);
            for(int i=0; i<limit; i++) 
                std::cout << "U:" << X[i] << std::endl << "u:" << u(x_points[i+1]) << std::endl << "e:" << fabs(X[i] - u(x_points[i+1])) << "\n\n";
        }

};