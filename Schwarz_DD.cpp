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

class Schwarz_DD {

    Jacobi j;
    Gauss_Seidel gs;

    public:

        Schwarz_DD() {}

        Schwarz_DD(Jacobi jacobi, Gauss_Seidel gauss_seidel) {
            j = jacobi;
            gs = gauss_seidel; 
        }

        int ABJASM(double *X,
                    double TOL,
                    int max_iterations,
                    int N, 
                    const int THREAD_COUNT, 
                    int block_size, 
                    double* A_Blocks,
                    double* inv_main_block,
                    double* F
                    ) {
                
                int iteration = 0;
                for(iteration=0; iteration<max_iterations; iteration++) {

                    double *old_X = copy_vector(X, N);
                    double local_errors[N];

                    #pragma omp parallel num_threads(THREAD_COUNT) shared(local_errors, X, inv_main_block, A_Blocks, F)
                    {
                        #pragma omp for
                        for(int i=0; i<THREAD_COUNT; i++) {

                            double *subdomain_solution;

                            if(i == THREAD_COUNT-1) {

                                subdomain_solution = AAS_Residual_Calc(inv_main_block, F, block_size, i*block_size, 2*block_size, 1, old_X[block_size * i - 1 ] * -1.0);

                                /*
                                For general matrices, comment the above 'subdomain_solution' and use the below 'subdomain_solution'. 
                                */

                                //subdomain_solution = matvecmul(inv_main_block, subtract(sliced_vector_F[i], matvecmul(&A_Blocks[((i*THREAD_COUNT)+(i-1)) * block_size * block_size], old_X, block_size, (i-1)*block_size, block_size), block_size), block_size, 0, 2*block_size);

                            } else {

                                subdomain_solution = AAS_Residual_Calc(inv_main_block, F, block_size, i*block_size, 2*block_size, 0, old_X[block_size*(i+1)] * -1.0);
                                
                                /*
                                For general matrices, comment the above 'subdomain_solution' and use the below 'subdomain_solution'. 
                                */

                                //subdomain_solution = matvecmul(inv_main_block, subtract(sliced_vector_F[i], matvecmul(&A_Blocks[((i*THREAD_COUNT)+(i+1)) * block_size * block_size], old_X, block_size, (i+1)*block_size, block_size), block_size), block_size, 0, 2*block_size);


                            }

                            int sub_solution_index = 0;
                            for(int j=i*block_size; j<block_size*(i+1); j++) {
                                X[j] = subdomain_solution[sub_solution_index++];
                                local_errors[j] = fabs(X[j] - old_X[j]);
                            }

                            free(subdomain_solution);

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



            int MBJASM(
                    double *X,
                    double TOL,
                    int max_iterations,
                    int N, 
                    const int THREAD_COUNT, 
                    int block_size, 
                    double* A,
                    double* F
                    ) {
                
                int iteration = 0;
                for(iteration=0; iteration<max_iterations; iteration++) {

                    double *old_X = copy_vector(X, N);
                    double local_errors[N];

                    #pragma omp parallel for shared(local_errors, X, A, F, old_X)
                    for(int i=0; i<THREAD_COUNT; i++) {


                        if(i > 0 && i < (THREAD_COUNT-1)) {
                            /*
                            For tridiagonal matrices.
                            */

                            gs.gauss_seidel_serial_ASM(X, F, A, TOL, 1000, N, i*block_size, block_size*(i+1), THREAD_COUNT-1, i, {old_X[block_size * i - 1 ] * -1.0, old_X[block_size*(i+1)] * -1.0});


                        }
                        else if(i == THREAD_COUNT-1) {
                            /*
                            For tridiagonal matrices.
                            */

                            //gs.gauss_seidel_serial_ASM(X, F, A, TOL, max_iterations, N, block_size, block_size*2, 1, old_X[block_size * i - 1 ] * -1.0);
                            //printf("\n%d %d", iteration, gs.gauss_seidel_serial_ASM(X, F, A, TOL, max_iterations, N, i*block_size, block_size*(i+1), THREAD_COUNT-1, i, {old_X[block_size * i - 1 ] * -1.0, 0.0}));
                            gs.gauss_seidel_serial_ASM(X, F, A, TOL, 1000, N, i*block_size, block_size*(i+1), THREAD_COUNT-1, i, {old_X[block_size * i - 1 ] * -1.0, 0.0});

                        } else {
                            /*
                            For tridiagonal matrices.
                            */

                            //printf("\n%d %d", iteration, gs.gauss_seidel_serial_ASM(X, F, A, TOL, max_iterations, N, 0, block_size, THREAD_COUNT-1, i, {0.0, old_X[block_size*(i+1)] * -1.0}));
                            gs.gauss_seidel_serial_ASM(X, F, A, TOL, max_iterations, N, 0, block_size, THREAD_COUNT-1, i, {0.0, old_X[block_size*(i+1)] * -1.0});

                        }

                        for(int j=0; j<N; j++) {
                            local_errors[j] = fabs(X[j] - old_X[j]);
                        }
                    }
                    #pragma omp barrier

                    free(old_X);
                    double *max_error = std::max_element(local_errors, local_errors+N);

                    if(*max_error < TOL) {
                        break;
                    }

                }
                return iteration;

            }



            int MBJRASM(
                    double *X,
                    double TOL,
                    int max_iterations,
                    int N, 
                    const int THREAD_COUNT, 
                    int DDBounds[][2], 
                    double* A,
                    double* F
                    ) {
                
                int iteration = 0;
                for(iteration=0; iteration<max_iterations; iteration++) {

                    double *old_X = copy_vector(X, N);
                    double local_errors[N];

                    #pragma omp parallel for shared(local_errors, X, A, F, old_X)
                    for(int i=0; i<THREAD_COUNT; i++) {


                        if(i > 0 && i < (THREAD_COUNT-1)) {
                            /*
                            For tridiagonal matrices.
                            */

                            gs.gauss_seidel_serial_ASM(X, F, A, TOL, 1000, N, 
                                                        DDBounds[i][0], 
                                                        DDBounds[i][1] + 1, 
                                                        THREAD_COUNT-1, 
                                                        i, 
                                                        {X[DDBounds[i][0] - 1] * -1.0, X[DDBounds[i][1] + 1] * -1.0});


                        }
                        else if(i == THREAD_COUNT-1) {
                            /*
                            For tridiagonal matrices.
                            */

                            //gs.gauss_seidel_serial_ASM(X, F, A, TOL, max_iterations, N, block_size, block_size*2, 1, old_X[block_size * i - 1 ] * -1.0);
                            //printf("\n%d %d", iteration, gs.gauss_seidel_serial_ASM(X, F, A, TOL, max_iterations, N, i*block_size, block_size*(i+1), THREAD_COUNT-1, i, {old_X[block_size * i - 1 ] * -1.0, 0.0}));
                            gs.gauss_seidel_serial_ASM(X, F, A, TOL, max_iterations, N, 
                                                        DDBounds[i][0], 
                                                        DDBounds[i][1] + 1, 
                                                        THREAD_COUNT-1, i, 
                                                        {X[DDBounds[i][0] - 1] * -1.0, 0.0});

                        } else {
                            /*
                            For tridiagonal matrices.
                            */

                            //printf("\n%d %d", iteration, gs.gauss_seidel_serial_ASM(X, F, A, TOL, max_iterations, N, 0, block_size, THREAD_COUNT-1, i, {0.0, old_X[block_size*(i+1)] * -1.0}));
                            gs.gauss_seidel_serial_ASM(X, F, A, TOL, max_iterations, N, 
                                                        DDBounds[0][0], 
                                                        DDBounds[0][1] + 1, 
                                                        THREAD_COUNT-1, i, 
                                                        {0.0, X[DDBounds[i][1] + 1] * -1.0});

                        }

                        for(int j=0; j<N; j++) {
                            local_errors[j] = fabs(X[j] - old_X[j]);
                        }
                    }
                    #pragma omp barrier

                    free(old_X);
                    double *max_error = std::max_element(local_errors, local_errors+N);

                    if(*max_error < TOL) {
                        break;
                    }

                }
                return iteration;

            }

            double* AAS_Residual_Calc(double* INV, double* F, int N, int vector_begin_index, int W, int subdomain_index, double residual_component) {
                double *V = create_vector(N);
                initialize_vector(V, 0.0, N);

                if(subdomain_index == 0) {

                    int i,j,v_index;
                    for(i=0;i<N;i++){
                        v_index = vector_begin_index;
                        for(j=0;j<N-1;j++){
                            V[i] = V[i] + (F[v_index++] * INV[i * W + j]);
                        }
                        V[i] = V[i] + ((F[v_index] - residual_component) * INV[i * W + j]);
                    }

                    return V;

                } else {

                    int i,j,v_index;
                    for(i=0;i<N;i++){
                        v_index = vector_begin_index;
                        V[i] = V[i] + ((F[v_index++] - residual_component) * INV[i * W + 0]);
                        for(j=1;j<N;j++){
                            V[i] = V[i] + (F[v_index++] * INV[i * W + j]);
                        }
                    }

                    return V;

                }
            }
            

};