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

// #include <execinfo.h>
// #include <signal.h>

#define BUF_LEN 100

#define LOG_INFO 0
#define LOG_WARN 1
#define LOG_ERRO 2

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
    /* TODO: YOUR CODE HERE */
    return mat->data[row * mat->cols + col];
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid
 */
void set(matrix *mat, int row, int col, double val) {
    /* TODO: YOUR CODE HERE */
    mat->data[row * mat->cols + col] = val;
}

/*
 * Sets all entries in mat to val
 */
void fill_matrix(matrix *mat, double val) {
    /* TODO: YOUR CODE HERE */
    int stride = 16;
    double vals[4] = {val, val, val, val};
    __m256d vector[4] = {_mm256_loadu_pd(vals), _mm256_loadu_pd(vals),  _mm256_loadu_pd(vals), _mm256_loadu_pd(vals)};

#pragma omp parallel num_threads(8)
{
    double* resultAddr;
    #pragma omp for
    for(int i = 0; i < mat->rows * mat->cols / stride * stride; i += stride) {
        resultAddr = mat->data + i;
        _mm256_storeu_pd(resultAddr, vector[0]);
        _mm256_storeu_pd(resultAddr + 4, vector[1]);
        _mm256_storeu_pd(resultAddr + 8, vector[2]);
        _mm256_storeu_pd(resultAddr + 12, vector[3]);
    }
}
    for (int i = mat->rows * mat->cols / stride * stride; i < mat->rows * mat->cols; i++) {
        *(mat->data + i) = val;
    }
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2){
    // if(mat1->rows != mat2->rows || mat1->cols != mat2->cols || result->rows != mat1->rows || result->cols != mat1->cols){
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
 * (OPTIONAL)
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    return 0;
}

double dot_product(double* a, double* b, int n) {
    double sum = 0;
    double vals[4] = {0, 0, 0, 0};
    __m256d vector[4] = {_mm256_loadu_pd(vals), _mm256_loadu_pd(vals),  _mm256_loadu_pd(vals), _mm256_loadu_pd(vals)};
    for (int i = 0; i < n / 16 * 16; i += 16) {
        double* vA_addr = a + i;
        double* vB_addr = b + i;
        vector[0] = _mm256_fmadd_pd(_mm256_loadu_pd(vA_addr), _mm256_loadu_pd(vB_addr), vector[0]);
        vector[1] = _mm256_fmadd_pd(_mm256_loadu_pd(vA_addr + 4), _mm256_loadu_pd(vB_addr + 4), vector[1]);
        vector[2] = _mm256_fmadd_pd(_mm256_loadu_pd(vA_addr + 8), _mm256_loadu_pd(vB_addr + 8), vector[2]);
        vector[3] = _mm256_fmadd_pd(_mm256_loadu_pd(vA_addr + 12), _mm256_loadu_pd(vB_addr + 12), vector[3]);
    }
    for (int i = n / 16 * 16; i < n; i++) {
        sum += a[i] * b[i];
    }
    for (int i = 0; i < 4; i++) {
        _mm256_storeu_pd(vals, vector[i]);
        sum += vals[0] + vals[1] + vals[2] + vals[3];
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

    if (mat1->cols != mat2->rows || result->rows != mat1->rows || result->cols != mat2->cols) {
        return -1;
    }
    int rows, K, cols;
    rows = mat1->rows;
    K = mat1->cols;
    cols = mat2->cols;

    // transposing
    matrix *mat2_transpose;
    allocate_matrix(&mat2_transpose, mat2->cols, mat2->rows);

#pragma omp parallel num_threads(8)
{
    #pragma omp for
    for (int i = 0; i < mat2_transpose->rows; i++)
        for (int j = 0; j < mat2_transpose->cols; j++) {
            mat2_transpose->data[i * mat2_transpose->cols + j] = mat2->data[j * mat2->cols + i];
        }

#pragma omp barrier
// The omp barrier directive identifies a synchronization point at which threads in a parallel region will wait until all other threads in that section reach the same point. 
// Statement execution past the omp barrier point then continues in parallel.
    #pragma omp for
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) {
            result->data[i * cols + j] = dot_product(mat1->data + i * K, mat2_transpose->data + j * K, K);
        }
}
    deallocate_matrix(mat2_transpose);
    return 0;
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    /* TODO: YOUR CODE HERE */

    matrix* tmp;
    allocate_matrix(&tmp, result->rows, result->cols);
    matrix* cur;
    allocate_matrix(&cur, result->rows, result->cols);

    fill_matrix(result, 0);
    for (int i = 0; i < mat->rows; i++) {
        set(result, i, i, 1);
    }
    memcpy(cur->data, mat->data, mat->rows * mat->cols * 8);

    int is_bit_set;
    while (pow > 0) {
        is_bit_set = pow & 0x01;
        if (is_bit_set) {
            memcpy(tmp->data, result->data, result->rows * result->cols * 8);
            mul_matrix(result, tmp, cur);
        }
        pow = pow >> 1;
        memcpy(tmp->data, result->data, result->rows * result->cols * 8);
        mul_matrix(cur, tmp, tmp);
    }

    deallocate_matrix(cur);
    deallocate_matrix(tmp);

    return 0;
}


/*
 * (OPTIONAL)
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int neg_matrix(matrix *result, matrix *mat) {
    /* TODO: YOUR CODE HERE */
    return 0;
}

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
