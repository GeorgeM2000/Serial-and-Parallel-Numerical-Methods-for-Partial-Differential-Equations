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

class Jacobi {
    public:

        Jacobi() {}


        int jacobi_serial(double *X,
                          double *B, 
                          double *A, 
                          double TOL, 
                          int max_iterations,  
                          int N) {

            int iteration = 0;
            for(iteration=0; iteration<max_iterations; iteration++) {
                
                double *old_X = copy_vector(X, N);
                double local_errors[N];
                for(int i=0; i<N; i++) {

                    // Uncomment the below block of code if the matrix is dense.

                    /* double s = 0.0f;
                    for (int j = 0; j < N; j++) {
                        if (j != i) {
                            s += A[i][j] * old_X[j];
                        }
                    }
                    X[i] = (1.0f / A[i][i]) * (B[i] - s);
                    local_errors[i] = fabs(X[i] - old_X[i]); */
                    
                    // Uncomment the below block of code for tridiagonal matrices.

                    if(i > 0 && i < N-1) {
                        X[i] = (1.0f / A[i * N + i]) * (B[i] - (A[i * N + (i-1)]*old_X[i-1]) - (A[i * N + (i+1)]*old_X[i+1]));
                    } 
                    else if(i == N-1) {
                        X[i] = (1.0f / A[i * N + i]) * (B[i] - (A[i * N + (i-1)]*old_X[i-1]));
                    } 
                    else {
                        X[i] = (1.0f / A[i * N + i]) * (B[i] - (A[i * N + (i+1)]*old_X[i+1]));
                    } 

                    local_errors[i] = fabs(X[i] - old_X[i]); // Use this error when the solution is unknown
                }
                free(old_X);
                double *max_error = std::max_element(local_errors, local_errors+N);
                
                if(*max_error < TOL) 
                    break;

            }
            
            return iteration;
        }

        
        int jacobi_parallel(double *X, 
                            double *B, 
                            double *A, 
                            double TOL,
                            int max_iterations, 
                            const std::vector<std::vector<int>>& partitions, 
                            int N, 
                            const int THREAD_COUNT) {

            int iteration = 0;
            for(iteration=0; iteration<max_iterations; iteration++) {

                double *old_X = copy_vector(X, N);
                double local_errors[N];
                #pragma omp parallel num_threads(THREAD_COUNT) shared(local_errors, X, A, B)
                {
                    #pragma omp for
                    for(int p=0; p<partitions.size(); p++) {

                        for(int variable: partitions[p]) {


                            // Uncomment the below block of code if the matrix is dense. 

                            /* double s = 0.0f;
                            for (int j = 0; j < N; ++j) {
                                if (j != variable) {
                                    s += A[variable][j] * old_X[j];
                                }
                            }
                            X[variable] = (1.0f / A[variable][variable]) * (B[variable] - s);
                            local_errors[variable] = fabs(X[variable] - old_X[variable]);  */
                            
                            

                            // Uncomment the below block of code for tridiagonal matrices.

                            if(variable > 0 && variable < N-1) {
                                X[variable] = (1.0f / A[variable * N + variable]) * (B[variable] - (A[variable * N + (variable-1)]*old_X[variable-1]) - (A[variable * N + (variable+1)]*old_X[variable+1]));
                            } 
                            else if(variable == N-1) {
                                X[variable] = (1.0f / A[variable * N + variable]) * (B[variable] - (A[variable * N + (variable-1)]*old_X[variable-1]));
                            } 
                            else {
                                X[variable] = (1.0f / A[variable * N + variable]) * (B[variable] - (A[variable * N + (variable+1)]*old_X[variable+1]));
                            } 
                            local_errors[variable] = fabs(X[variable] - old_X[variable]);                  
                        }

                    }
                    #pragma omp barrier
                }
                free(old_X);
                double *max_error = std::max_element(local_errors, local_errors+N);

                if(*max_error < TOL) {
                    break;
                }  
            }
            return iteration;
        }


        

};
