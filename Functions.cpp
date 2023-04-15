double* create_matrix(int cols, int rows) {
	return (double*)malloc(sizeof(double) * cols * rows);
}

double* create_3D_matrix(int cols, int rows, int number_of_blocks) {
    return (double*)malloc(sizeof(double) * cols * rows * number_of_blocks);
}

void initialize_vector(double *V, double value, int N) {
    int i;
    for(i=0; i<N; i++) 
        V[i] = value;
}

double* create_vector(int N) {
    return (double*)malloc(sizeof(double) * N);
}

void print_vector(const char* s, double *V, int N) {
    printf("%s ", s);
    std::cout.precision(std::numeric_limits<double>::max_digits10);

    int i;
    for (i = 0; i < N; ++i) {
        std::cout << V[i] << std::endl;
    }
    printf("\n");
}

void initialize_matrix(double *A, double value, int N) {
    int i,j;
    for(i=0; i<N; i++) 
        for(j=0; j<N; j++) 
            A[i * N + j] = value;
}

void print_matrix(const char* s, double *A, int rows, int cols) {
    printf("%s ", s);
    std::cout.precision(std::numeric_limits<double>::max_digits10);

    int i,j;
    for (i = 0; i < rows; ++i) {
        for (j = 0; j < cols; ++j) {
            std::cout << A[i * cols + j] << " ";
        }
        printf("\n");
    }
    printf("\n");
}

double* copy_vector(double *A, int N) {
    double *V = create_vector(N);;

    int i;
    for(i=0; i<N; i++) {
        V[i] = A[i];
    }
    return V;
}

double* copy_subvector(double *X, int start, int end) {
    double *V = create_vector(end - start);

    int i, vec_start = 0;
    for(i=start; i<end; i++) {
        V[vec_start++] = X[i];
    }
    return V;
}



int write_vector_to_file(const char* filename, double* V, int N) {
    std::ofstream fout(filename);
    if(fout.is_open()) {
        for(int i=0; i<N; i++) {
            fout << V[i] << "\n";

        }
        fout.close();
    } else {
        return 0;
    }
    return 1;
}

void print_blocks(double* Blocks, int number_of_blocks, int block_size) {
    std::cout.precision(std::numeric_limits<double>::max_digits10);

    int i,j,k;
    for(i=0; i<number_of_blocks; i++) {
        for(j=0; j<block_size; j++) {
            for(k=0; k<block_size; k++) {
                std::cout << Blocks[i * block_size * block_size + j * block_size + k] << " ";
            }
            printf("\n");
        }
        printf("\n");
    }
}


double* create_non_overlapping_blocks(double *A, int N, const int THREAD_COUNT, int block_size, int number_of_blocks) {
    if(N % THREAD_COUNT != 0) {
        std::cout << "The division between the dimension N of the matrix and the number of threads(blocks) must be an integer." << std::endl;
        return nullptr;
    }

    double *A_Blocks = create_3D_matrix(block_size, block_size, number_of_blocks);

    int i, j, z, k, block_matrix_index = 0, block_row_index, block_col_index;
    for(i=0; i<1 /*THREAD_COUNT*/; i++) {

        j = 0;
        while(j <= N-block_size) {

            block_row_index = 0;           
            for(z=i*block_size; z<(i+1)*(block_size); z++) {

                block_col_index = 0;
                for(k=j; k<j+block_size; k++) {
                    A_Blocks[block_matrix_index * block_size * block_size + block_row_index * block_size + block_col_index] = A[z * N + k];
                    block_col_index++;
                }
                block_row_index++;
            }
            j += block_size;
            block_matrix_index++;

            break;

        }
    }

    return A_Blocks;
}



std::vector<std::vector<int>> create_variable_partitions(int N, const int THREAD_COUNT) {
    int partition_size = (int)std::floor(N/THREAD_COUNT);
    std::vector<std::vector<int>> partitions;

    // Add varibales 0 - N-1 to partitions
    for(int i=1; i<=THREAD_COUNT; i++) {
        std::vector<int> partition;
        int j = partition_size*(i-1);
        for(j; j<partition_size*i; j++) {
            partition.push_back(j);
        }
        partitions.push_back(partition);
    }

    // Add remaining variables to last partition
    for(int rem=(THREAD_COUNT*partition_size); rem<N; rem++) partitions[partitions.size() - 1].push_back(rem);

    return partitions;
}

// Function to perform the inverse operation on the matrix.
void inv(double* A, int order){
    double temp;
 
    for (int i = 0; i < order; i++) {
        for (int j = 0; j < 2 * order; j++) {
            if (j == (i + order))
                A[i * (2*order) + j] = 1;
        }
    }
 
    
    for (int i = order - 1; i > 0; i--) {
        if (A[(i - 1) * order + 0] < A[i * order + 0]) {
            
            double* temp = &A[i * (2*order)];

            for(int j=0; j<2*order; j++) {
                A[i * (2*order)] = A[(i-1) * (2*order)];
            }

            for(int j=0; j<2*order; j++) {
                A[(i-1) * (2*order)] = temp[j];
            }
               
        }
    }
 
    for (int i = 0; i < order; i++) {
        for (int j = 0; j < order; j++) {
            if (j != i) {
                temp = A[j * (2*order) + i] / A[i * (2*order) + i];
                for (int k = 0; k < 2 * order; k++) {
                    A[j * (2*order) + k] -= A[i * (2*order) + k] * temp;
                }
            }
        }
    }
 
    for (int i = 0; i < order; i++) {
        temp = A[i * (2*order) + i];
        for (int j = 0; j < 2 * order; j++) {
            A[i * (2*order) + j] = A[i * (2*order) + j] / temp;
        }
    }
}


double* create_inverse_diagonal_block(double* A_Blocks, const int THREAD_COUNT, int block_size) {
    
    double *inverse_main_block = create_matrix(2*block_size, block_size);

    for(int i=0; i<block_size; i++) 
        for(int j=0; j<block_size; j++) 
            inverse_main_block[i * (2*block_size) + j] = A_Blocks[0 * block_size * block_size + i * block_size + j];
        
    
    inv(inverse_main_block, block_size);

    for(int i=0; i<block_size; i++) 
        for(int j=0; j<block_size; j++) 
            inverse_main_block[i * (2*block_size) + j] = inverse_main_block[i * (2*block_size) + (j+block_size)];
        
    
    return inverse_main_block;
}


double* matvecmul(double* A, double* X, int n, int vector_start, int w) {
    double *V = create_vector(n);
    initialize_vector(V, 0.0, n);

    int i,j,vec_start;
    for(i=0;i<n;i++){
        vec_start = vector_start;
        for(j=0;j<n;j++){
            V[i] = V[i] + (X[vec_start++] * A[i * w + j]);
        }
    }

    return V;
}




double* subtract(double* A, double* B, int N) {
    double *V = create_vector(N);
    
    int i;
    for(i=0; i<N; i++) {
        V[i] = A[i] - B[i];
    }

    return V;
}

double* add(double* A, double* B, int N) {
    double *V = create_vector(N);
    
    int i;
    for(i=0; i<N; i++) {
        V[i] = A[i] + B[i];
    }

    return V;
}


double* slice(double *V, int start, int end) {
    double *sliced_V = create_vector(end-start);

    int vector_index = 0, i;
    for(i=start; i<end; i++) {
        sliced_V[vector_index++] = V[i];
    }

    return sliced_V;

}


double** create_sliced_vectors(double* F, int block_size, const int THREAD_COUNT) {
    double** sliced_vectors = new double *[THREAD_COUNT];
    for(int row=0; row<THREAD_COUNT; row++) sliced_vectors[row] = new double[block_size];

    for(int i=0; i<THREAD_COUNT; i++) {
        sliced_vectors[i] = slice(F, i*block_size, block_size*(i+1));
    }

    return sliced_vectors;
}

