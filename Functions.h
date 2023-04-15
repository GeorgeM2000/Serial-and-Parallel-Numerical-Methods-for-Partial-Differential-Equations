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


