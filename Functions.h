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

long usecs (void);

void initialize_vector(double *V, double value, int N);

double* create_vector(int N);

void print_vector(const char* s, double *V, int N);

void initialize_matrix(double *A, double value, int N);

void print_matrix(const char* s, double *A, int rows, int cols);

double* copy_vector(double *A, int N);

int write_vector_to_file(const char* filename, double* V, int N);

double* create_non_overlapping_blocks(double *A, int N, const int THREAD_COUNT, int block_size, int number_of_blocks);

std::vector<std::vector<int>> create_variable_partitions(int N, const int THREAD_COUNT);

void print_blocks(double* Blocks, int number_of_blocks, int block_size);

double* create_inverse_diagonal_block(double* A_Blocks, const int THREAD_COUNT, int block_size);

double* matvecmul(double* A, double* X, int N, int vector_begin_index, int W);

double* subtract(double* A, double* B, int N);

double* add(double* A, double* B, int N);

double* slice(double *V, int start, int end);

void inv(double* A, int order);


