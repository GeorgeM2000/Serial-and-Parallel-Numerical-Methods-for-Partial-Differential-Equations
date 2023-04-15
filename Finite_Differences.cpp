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


class Finite_Differences {
    int N;

    public:

        Finite_Differences() {}

        void set_N(int n) {
            N = n;
        }

        template<typename T>
        double* create_tridiagonal_matrix(double h, T&& q, std::function<double(double)> p, std::vector<double> X_points) {
            double* A = create_matrix(N-2, N-2);

            h = h * h;
            for(int i=0; i<N-2; i++) {
                for(int j=0; j<N-2; j++) {
                    if(i == j) {
                        A[i * (N-2) + j] = (2.0 * 1.0) + h*q(X_points[i+1]);
                    } else if(i == j-1 || i == j+1) {
                        A[i * (N-2) + j] = -1.0 * 1.0;
                    } else {
                        A[i * (N-2) + j] = 0.0;
                    }
                }
            }
            return A;
        }

        std::vector<double> arange(double start, double end, double step) {
            std::vector<double> x_points;
            for(double s=start; s<end; s+=step) {
                x_points.push_back(s);
            }
        
            return x_points;
        }

        template<typename T>
        double* create_vector_F(double h, T&& f, double* boundary_conditions, std::vector<double> X_points) {
            double* F = create_vector(N-2);
            h = h * h;

            F[0] = h * f(X_points[1]) + boundary_conditions[0];
            F[N-3] = h * f(X_points[N-2]) + boundary_conditions[1];

            for(int i=1; i<N-3; i++){
                F[i] = h * f(X_points[i+1]);
            }

            return F;
        }

        template<typename T>
        double L2_NormError(T&& u, double h, double* X, std::vector<double> X_points) {
            double error = 0.0;
            for(int i=0; i<N-2; i++) {
                error += pow(fabs(u(X_points[i+1]) - X[i]), 2.0);
            }
            error = h * error;
            
            return pow(error, 1.0/2.0);
        }

        template<typename T>
        double Max_NormError(T&& u, double h, double* X, std::vector<double> X_points) {
            double local_Error[N-2];
            for(int i=0; i<N-2; i++) {
                local_Error[i] = fabs(X[i] - u(X_points[i+1]));
            }
            double* max = std::max_element(local_Error, local_Error+(N-2));

            return *max;
        }

        template<typename T>
        void compare_solutions(T&& u, double* X, std::vector<double> x_points, int limit) {
            std::cout.precision(std::numeric_limits<double>::max_digits10);
            for(int i=0; i<limit; i++) 
                std::cout << "U:" << X[i] << std::endl << "u:" << u(x_points[i+1]) << std::endl << "e:" << fabs(X[i] - u(x_points[i+1])) << "\n\n";
        }

};