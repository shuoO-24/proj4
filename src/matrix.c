#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. `parent` should be set to NULL to indicate that
 * this matrix is not a slice. You should also set `ref_cnt` to 1.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails. Remember to set the error messages in numc.c.
 * Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    if(rows <= 0 || cols <= 0){
    	return -1;
    }
    *mat = (struct matrix *)calloc(1, sizeof(struct matrix));
    if (*mat == NULL) {
        return -2;
    }
    matrix *m = *mat;
    m->rows = rows;
    m->cols = cols;
    m->ref_cnt = 1;
    m->parent = NULL;
    m->data = (double *)calloc(rows*cols, sizeof(double)); 
    // (*mat) = m;
    if(m->data == NULL){
       return -2;
    }
    return 0;
}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data should point to the `offset`th entry of `from`'s data (you do not need to allocate memory)
 * for the data field. `parent` should be set to `from` to indicate this matrix is a slice of `from`.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Remember to set the error messages in numc.c.
 * Return 0 upon success.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    /* TODO: YOUR CODE HERE */
    if(rows <= 0 || cols <= 0 || offset + rows * cols > from->rows * from->cols){
    	return -1;
    }
    *mat = (struct matrix *)calloc(1, sizeof(struct matrix));
    if (*mat == NULL) {
        return -2;
    }
    matrix *m = *mat;
    m->rows = rows;
    m->cols = cols;
    m->ref_cnt = -1;
    m->parent = from;
    m->parent->ref_cnt +=1;
    m->data = from->data + offset;
    // (*mat) = m; 

    return 0;
}

/*
 * You need to make sure that you only free `mat->data` if `mat` is not a slice and has no existing slices,
 * or that you free `mat->parent->data` if `mat` is the last existing slice of its parent matrix and its parent matrix has no other references
 * (including itself). You cannot assume that mat is not NULL.
 */
void deallocate_matrix(matrix *mat) {
    /* TODO: YOUR CODE HERE */
    if(mat == NULL){
	    return;
    }
    
    matrix* tmp = mat;
    do{
        tmp->ref_cnt -= 1;
	    tmp = tmp->parent;
    } while(tmp != NULL);

   
    if(mat->parent == NULL){
        if (mat->ref_cnt > 0) {
            mat->ref_cnt -= 1;

            if(mat->ref_cnt == 0) {
                free(mat->data);
                mat->data = NULL;
                free(mat);
                return;
            }
        }
    }
    if (mat->parent != NULL){
        if (mat->parent->ref_cnt > 0) {
            mat->parent->ref_cnt -= 1;
            
            if(mat->parent->ref_cnt == 0){
                free(mat->parent->data);
                mat->data = NULL;
                free(mat->parent);
                return;
            }
        }
        free(mat);
	// mat = mat->parent;
    }
    return;

}

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid.
 */
double get(matrix *mat, int row, int col) {
    unsigned int index = (mat->cols)*row +col;
    return *((mat->data)+index);
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid
 */
void set(matrix *mat, int row, int col, double val) {
    /* TODO: YOUR CODE HERE */
    unsigned int index = (mat->cols)*row + col;
    *((mat->data)+index) = val;
    return;
}

/*
 * Sets all entries in mat to val
 */
void fill_matrix(matrix *mat, double val) {
    /* TODO: YOUR CODE HERE */
    for (int i = 0; i < mat->cols; ++i) {
        for (int j = 0; j < mat->rows; ++j) {
            *(mat->data + i * mat->cols + j) = val;
        }
    }
    return;
}


/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2){
    // if(mat1->rows != mat2->rows || mat1->cols != mat2->cols || result->rows != mat1->rows || result->cols != mat1->cols){
    //     return 1;
    // }
    // result->rows = mat1->rows;
    // result->cols = mat1->cols;
    // for (int index = 0; index< mat1->rows * mat2->cols; index++) {
    //     *(result->data + index) = *(mat1->data + index) + *(mat2->data + index);
	// }
    // if(!result){
	//     return 0;
    // }
    
    int stride = 16;

#pragma omp parallel num_threads(8)
{
    __m256d vector[4];
    double* mat1Addr;
    double* mat2Addr;
    double* resultAddr;
    #pragma omp for
    for(int i = 0; i < result->rows * result->cols / stride * stride; i += stride) {
        mat1Addr = mat1->data + i;
        mat2Addr = mat2->data + i;
        resultAddr = result->data + i;
        vector[0] = _mm256_add_pd(_mm256_loadu_pd(mat1Addr), _mm256_loadu_pd(mat2Addr));
        vector[1] = _mm256_add_pd(_mm256_loadu_pd(mat1Addr + 4), _mm256_loadu_pd(mat2Addr + 4));
        vector[2] = _mm256_add_pd(_mm256_loadu_pd(mat1Addr + 8), _mm256_loadu_pd(mat2Addr + 8));
        vector[3] = _mm256_add_pd(_mm256_loadu_pd(mat1Addr + 12), _mm256_loadu_pd(mat2Addr + 12));
        _mm256_storeu_pd(resultAddr, vector[0]);
        _mm256_storeu_pd(resultAddr + 4, vector[1]);
        _mm256_storeu_pd(resultAddr + 8, vector[2]);
        _mm256_storeu_pd(resultAddr + 12, vector[3]);
    }
}

    for (int i = result->rows * result->cols / stride * stride; i < result->rows * result->cols; i++) {
        *(result->data + i) = *(mat1->data + i) + *(mat2->data + i);
    }

    return 0;
}
/* 
for(int i = 0; i < i_size; ++i) {
    for(int j = 0; j < j_size; ++j)
         c[i][j] = a[i][j] + b[i][j];
}
*/


/*
 * (OPTIONAL)
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    return 0;
}

int check_dim_match_mul(matrix *result, matrix *mat1, matrix *mat2)
{
    if (mat1->cols != mat2->rows || result->rows != mat1->rows || result->cols != mat2->cols) {
        return 0;
    }

    return 1;
}

double dot_prod(double* a, double* b, int n) {
    double sum = 0;

    // Initialize to 0
    double mem_vals[4] = {0, 0, 0, 0};
    __m256d vector[4] = {
        _mm256_loadu_pd(mem_vals),
        _mm256_loadu_pd(mem_vals),
        _mm256_loadu_pd(mem_vals),
        _mm256_loadu_pd(mem_vals)
    };

    for (int i = 0; i < n / 16 * 16; i += 16) {
        double* a_this_start_addr = a + i;
        double* b_this_start_addr = b + i;

        vector[0] = _mm256_fmadd_pd(_mm256_loadu_pd(a_this_start_addr), _mm256_loadu_pd(b_this_start_addr), vector[0]);
        vector[1] = _mm256_fmadd_pd(_mm256_loadu_pd(a_this_start_addr+4), _mm256_loadu_pd(b_this_start_addr+4), vector[1]);
        vector[2] = _mm256_fmadd_pd(_mm256_loadu_pd(a_this_start_addr+8), _mm256_loadu_pd(b_this_start_addr+8), vector[2]);
        vector[3] = _mm256_fmadd_pd(_mm256_loadu_pd(a_this_start_addr+12), _mm256_loadu_pd(b_this_start_addr+12), vector[3]);
    }

    for (int i = n / 16 * 16; i < n; i++) {
        sum += a[i] * b[i];
    }

    for (int i = 0; i < 4; i++) {
        _mm256_storeu_pd(mem_vals, vector[i]);
        sum += mem_vals[0] + mem_vals[1] + mem_vals[2] + mem_vals[3];
    }

    return sum;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */

    // check if dim matches
    if (mat1->cols != mat2->rows || result->rows != mat1->rows || result->cols != mat2->cols) {
        return -1;
    }

    // result matrix is guaranteed t be a different matrix object from mat1 and mat2
    int M, K, N;
    M = mat1->rows;
    K = mat1->cols;
    N = mat2->cols;

    // transpose mat2 to offer better memory access pattern
    matrix *mat2_T;
    allocate_matrix(&mat2_T, mat2->cols, mat2->rows);

#pragma omp parallel num_threads(8)
{
    #pragma omp for
    for (int i = 0; i < mat2_T->rows; i++)
        for (int j = 0; j < mat2_T->cols; j++) {
            mat2_T->data[i*mat2_T->cols + j] = mat2->data[j*mat2->cols+i];
        }

#pragma omp barrier

    #pragma omp for
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            result->data[i*N + j] = dot_prod(mat1->data + i*K, mat2_T->data + j*K, K);
        }
}

    deallocate_matrix(mat2_T);
    return 0;
}

/* 
for(int i = 0; i < mat1->rows; ++i) {
    for(int j = 0; j < mat1->cols; ++j)
         c[i][j] = 0;

    for(int k = 0; k < k_size; ++k) {
         double aa = a[i][k];
         for(int j = 0; j < j_size; ++j)
             c[i][j] += aa*b[k][j];
    }
}
*/


// Helper function for power function
void copy_matrix(matrix *des, matrix *src) {
    int stride = 16;
    int length = src->rows * src->cols;

#pragma omp parallel num_threads(8)
{        
    double *des_addr, *src_addr;
    for (int i = 0; i < length / stride * stride; i += stride) {
        des_addr = des->data + i;
        src_addr = src->data + i;
        _mm256_storeu_pd(des_addr, _mm256_loadu_pd(src_addr));
        _mm256_storeu_pd(des_addr + 4, _mm256_loadu_pd(src_addr + 4));   
        _mm256_storeu_pd(des_addr + 8, _mm256_loadu_pd(src_addr + 8));  
        _mm256_storeu_pd(des_addr + 12, _mm256_loadu_pd(src_addr + 12));    
    }
}
    for (int i = length / stride * stride; i < length; ++i) {
        *(des->data + i) = *(src->data + i);
    }
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 */
//I revised some errors: 1. Algorithmic error (b/c this is function for matrix power, not elementwise power)
//2. undetermined variable a,b
//But using four nested loop for statement can cause significantly low performance, we should change some algorithms.
int pow_matrix(matrix *result, matrix *mat, int pow) {
    /* TODO: YOUR CODE HERE */

    matrix* tmp_mat;
    allocate_matrix(&tmp_mat, result->rows, result->cols);
    matrix* mat_ith_pow;
    allocate_matrix(&mat_ith_pow, result->rows, result->cols);

    // initial value for the result matrix: the identity matrix
    fill_matrix(result, 0);
    for (int i = 0; i < mat->rows; i++) {
        set(result, i, i, 1);
    }

    // initial value for the mat_ith_pow matrix: mat^1
    copy_matrix(mat_ith_pow, mat);

    // by matrix multiplication, logn algorithm
    int is_bit_set;
    while (pow > 0) {
        is_bit_set = pow & 0x01;
        if (is_bit_set) {
            copy_matrix(tmp_mat, result);
            mul_matrix(result, tmp_mat, mat_ith_pow);
        }

        pow = pow >> 1;
        // compute mat^{2^i} by squaring
        // 1. store old value of mat_ith_pow into tmp_mat
        copy_matrix(tmp_mat, mat_ith_pow);
        // 2. increase power by 1
        mul_matrix(mat_ith_pow, tmp_mat, tmp_mat);
    }

    deallocate_matrix(mat_ith_pow);
    deallocate_matrix(tmp_mat);

    return 0;
}

/*
 * (OPTIONAL)
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */

// I revised some minor errors : 1. undetermined variables a,c
int neg_matrix(matrix *result, matrix *mat) {
    /* TODO: YOUR CODE HERE */
    double * c = result->data;
    double *a = mat->data;
    int rows = mat->rows;
    int cols = mat->cols;
    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j) {
            (*(c + (cols * i) + j)) = (-1) * (*(a + (cols*i) +j));
        }
    }
    return 0;
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */

// I revised some minor errors (undectermined variables a,c, using conditional statements to alternate abs functioin
// which can occurs low performance)
int abs_matrix(matrix *result, matrix *mat) {
    /* TODO: YOUR CODE HERE */
    int length = result->rows * result->cols;
    int stride = 16;

#pragma omp parallel num_threads(8)
{
    __m256d vector[4];
    double* result_addr;
    double* mat_addr;
    #pragma omp for
    for (int i = 0; i < length / stride * stride; i += stride) {
        result_addr = result->data + i;
        mat_addr = mat->data + i;
        // mask off sign bit using andnot
        vector[0] = _mm256_andnot_pd(_mm256_set1_pd(-0.0f), _mm256_loadu_pd(mat_addr));
        vector[1] = _mm256_andnot_pd(_mm256_set1_pd(-0.0f), _mm256_loadu_pd(mat_addr + 4));
        vector[2] = _mm256_andnot_pd(_mm256_set1_pd(-0.0f), _mm256_loadu_pd(mat_addr + 8));
        vector[3] = _mm256_andnot_pd(_mm256_set1_pd(-0.0f), _mm256_loadu_pd(mat_addr + 12));

        _mm256_storeu_pd(result_addr, vector[0]);
        _mm256_storeu_pd(result_addr + 4, vector[1]);
        _mm256_storeu_pd(result_addr + 8, vector[2]);
        _mm256_storeu_pd(result_addr + 12, vector[3]);
    }
}
    for (int i = length / stride * stride; i < length; i++) {
        *(result->data + i) = *(mat->data + i) < 0 ? -*(mat->data + i) : *(mat->data + i);
    }

    return 0;
}
