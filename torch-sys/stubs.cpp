// Stub implementations for tch-rs stream callbacks
// These are normally provided by Rust, but we stub them out for Idris FFI

#include <cstddef>
#include <cstdint>
#include "torch_api_generated.h"

extern "C" {

// Stream stubs matching torch_api.h declarations
bool tch_read_stream_destructor(void *stream_ptr) {
    return true;  // success
}

bool tch_read_stream_read(void *stream_ptr, uint8_t *buf, size_t size, size_t *new_pos) {
    *new_pos = 0;
    return false;  // no data
}

bool tch_read_stream_seek_start(void *stream_ptr, uint64_t pos, uint64_t *new_pos) {
    *new_pos = 0;
    return true;
}

bool tch_read_stream_seek_end(void *stream_ptr, int64_t pos, uint64_t *new_pos) {
    *new_pos = 0;
    return true;
}

bool tch_read_stream_stream_position(void *stream_ptr, uint64_t *pos) {
    *pos = 0;
    return true;
}

bool tch_write_stream_destructor(void *stream_ptr) {
    return true;
}

bool tch_write_stream_write(void *stream_ptr, const uint8_t *buf, size_t size, size_t *out_size) {
    *out_size = 0;
    return false;  // no write
}

// Idris FFI helper: read tensor pointer from out-parameter location
void* idris_read_tensor_ptr(void **out) {
    return out[0];
}

// Idris FFI helper: create zeros tensor with simple 1D shape
void idris_zeros_1d(void **out, int64_t size, int dtype, int device) {
    int64_t dims[] = {size};
    atg_zeros((tensor*)out, dims, 1, dtype, device);
}

// Idris FFI helper: create ones tensor with simple 1D shape
void idris_ones_1d(void **out, int64_t size, int dtype, int device) {
    int64_t dims[] = {size};
    atg_ones((tensor*)out, dims, 1, dtype, device);
}

// Idris FFI helper: create zeros tensor with 2D shape
void idris_zeros_2d(void **out, int64_t d0, int64_t d1, int dtype, int device) {
    int64_t dims[] = {d0, d1};
    atg_zeros((tensor*)out, dims, 2, dtype, device);
}

// Idris FFI helper: create ones tensor with 2D shape
void idris_ones_2d(void **out, int64_t d0, int64_t d1, int dtype, int device) {
    int64_t dims[] = {d0, d1};
    atg_ones((tensor*)out, dims, 2, dtype, device);
}

// Debug: simple test to check FFI parameter passing
int64_t idris_debug_echo(int64_t x) {
    return x * 2;
}

// Debug: test out-parameter pattern
void idris_debug_outptr(void **out, int64_t val) {
    if (out != nullptr) {
        *out = (void*)(val * 3);
    }
}

// ============================================================
// Tier 1: Shape Operations
// ============================================================

// REQ-T4-SHP-001: Transpose two dimensions
void idris_transpose(void **out, void *t, int64_t dim0, int64_t dim1) {
    atg_transpose((tensor*)out, (tensor)t, dim0, dim1);
}

// REQ-T4-SHP-002/003: Reshape/View with up to 4 dimensions
void idris_reshape_1d(void **out, void *t, int64_t d0) {
    int64_t dims[] = {d0};
    atg_reshape((tensor*)out, (tensor)t, dims, 1);
}

void idris_reshape_2d(void **out, void *t, int64_t d0, int64_t d1) {
    int64_t dims[] = {d0, d1};
    atg_reshape((tensor*)out, (tensor)t, dims, 2);
}

void idris_reshape_3d(void **out, void *t, int64_t d0, int64_t d1, int64_t d2) {
    int64_t dims[] = {d0, d1, d2};
    atg_reshape((tensor*)out, (tensor)t, dims, 3);
}

void idris_reshape_4d(void **out, void *t, int64_t d0, int64_t d1, int64_t d2, int64_t d3) {
    int64_t dims[] = {d0, d1, d2, d3};
    atg_reshape((tensor*)out, (tensor)t, dims, 4);
}

void idris_view_1d(void **out, void *t, int64_t d0) {
    int64_t dims[] = {d0};
    atg_view((tensor*)out, (tensor)t, dims, 1);
}

void idris_view_2d(void **out, void *t, int64_t d0, int64_t d1) {
    int64_t dims[] = {d0, d1};
    atg_view((tensor*)out, (tensor)t, dims, 2);
}

void idris_view_3d(void **out, void *t, int64_t d0, int64_t d1, int64_t d2) {
    int64_t dims[] = {d0, d1, d2};
    atg_view((tensor*)out, (tensor)t, dims, 3);
}

void idris_view_4d(void **out, void *t, int64_t d0, int64_t d1, int64_t d2, int64_t d3) {
    int64_t dims[] = {d0, d1, d2, d3};
    atg_view((tensor*)out, (tensor)t, dims, 4);
}

// ============================================================
// Tier 2: Tensor Creation
// ============================================================

// REQ-T4-CRE-003: Random normal tensor
void idris_randn_1d(void **out, int64_t d0, int dtype, int device) {
    int64_t dims[] = {d0};
    atg_randn((tensor*)out, dims, 1, dtype, device);
}

void idris_randn_2d(void **out, int64_t d0, int64_t d1, int dtype, int device) {
    int64_t dims[] = {d0, d1};
    atg_randn((tensor*)out, dims, 2, dtype, device);
}

void idris_randn_3d(void **out, int64_t d0, int64_t d1, int64_t d2, int dtype, int device) {
    int64_t dims[] = {d0, d1, d2};
    atg_randn((tensor*)out, dims, 3, dtype, device);
}

// ============================================================
// Tier 3: Shape Queries
// ============================================================

// REQ-T4-QRY-001: Get shape (returns ndim, fills dims array)
int64_t idris_shape(void *t, int64_t *dims_out) {
    tensor tens = (tensor)t;
    int64_t ndim = at_dim(tens);
    at_shape(tens, dims_out);
    return ndim;
}

// REQ-T4-QRY-002: Size at dimension
int64_t idris_size_dim(void *t, int64_t dim) {
    tensor tens = (tensor)t;
    int64_t dims[16];
    at_shape(tens, dims);
    return dims[dim];
}

// ============================================================
// Tier 4: Tensor Combination
// ============================================================

// REQ-T4-CMB-001: Concatenate tensors along dimension
void idris_cat_2(void **out, void *t0, void *t1, int64_t dim) {
    tensor tensors[] = {(tensor)t0, (tensor)t1};
    atg_cat((tensor*)out, tensors, 2, dim);
}

void idris_cat_3(void **out, void *t0, void *t1, void *t2, int64_t dim) {
    tensor tensors[] = {(tensor)t0, (tensor)t1, (tensor)t2};
    atg_cat((tensor*)out, tensors, 3, dim);
}

// REQ-T4-CMB-002: Stack tensors along new dimension
void idris_stack_2(void **out, void *t0, void *t1, int64_t dim) {
    tensor tensors[] = {(tensor)t0, (tensor)t1};
    atg_stack((tensor*)out, tensors, 2, dim);
}

void idris_stack_3(void **out, void *t0, void *t1, void *t2, int64_t dim) {
    tensor tensors[] = {(tensor)t0, (tensor)t1, (tensor)t2};
    atg_stack((tensor*)out, tensors, 3, dim);
}

// ============================================================
// Tier 5: Neural Network Primitives
// ============================================================

// REQ-T4-NN-001: Layer normalization (simplified: no weight/bias)
void idris_layer_norm_1d(void **out, void *t, int64_t norm_dim, double eps) {
    int64_t shape[] = {norm_dim};
    atg_layer_norm((tensor*)out, (tensor)t, shape, 1, nullptr, nullptr, eps, 1);
}

void idris_layer_norm_2d(void **out, void *t, int64_t d0, int64_t d1, double eps) {
    int64_t shape[] = {d0, d1};
    atg_layer_norm((tensor*)out, (tensor)t, shape, 2, nullptr, nullptr, eps, 1);
}

// REQ-T4-NN-002: Embedding lookup
void idris_embedding(void **out, void *weight, void *indices) {
    // padding_idx=-1 (none), scale_grad_by_freq=0, sparse=0
    atg_embedding((tensor*)out, (tensor)weight, (tensor)indices, -1, 0, 0);
}

// REQ-T4-NN-003: Dropout
void idris_dropout(void **out, void *t, double p, int training) {
    atg_dropout((tensor*)out, (tensor)t, p, training);
}

// REQ-T4-NN-004: GELU activation
void idris_gelu(void **out, void *t) {
    // approximate="none" (exact GELU)
    atg_gelu((tensor*)out, (tensor)t, (char*)"none", 4);
}

// ============================================================
// Tier 3 continued: Scalar extraction
// ============================================================

// REQ-T4-QRY-004: Extract scalar value (for 0-dim or 1-element tensors)
double idris_item_double(void *t) {
    tensor tens = (tensor)t;
    double val;
    at_copy_data(tens, &val, 1, sizeof(double));
    return val;
}

// ============================================================
// Buffer writing helpers for List marshalling
// ============================================================

void idris_write_int64(int64_t *buf, int64_t idx, int64_t val) {
    buf[idx] = val;
}

void idris_write_double(double *buf, int64_t idx, double val) {
    buf[idx] = val;
}

// ============================================================
// Tier 5: Data Bridge - List to Tensor conversion
// ============================================================

// REQ-T5-CRE-001: Create tensor from Int64 array
void idris_from_array_int64(void **out, int64_t *data, int64_t len) {
    int64_t dims[] = {len};
    *out = at_tensor_of_data(data, dims, 1, sizeof(int64_t), 4);  // dtype=4 (Int64)
}

// REQ-T5-CRE-002: Create tensor from Double array
void idris_from_array_double(void **out, double *data, int64_t len) {
    int64_t dims[] = {len};
    *out = at_tensor_of_data(data, dims, 1, sizeof(double), 7);  // dtype=7 (Float64)
}

// ============================================================
// Tier 5: Reduction Operations
// ============================================================

// REQ-T5-RED-001: Mean along dimension
void idris_mean_dim(void **out, void *t, int64_t dim, int keepdim) {
    int64_t dims[] = {dim};
    atg_mean_dim((tensor*)out, (tensor)t, dims, 1, keepdim, -1);  // dtype=-1 = preserve
}

// REQ-T5-RED-002: Sum along dimension
void idris_sum_dim(void **out, void *t, int64_t dim, int keepdim) {
    int64_t dims[] = {dim};
    atg_sum_dim_intlist((tensor*)out, (tensor)t, dims, 1, keepdim, -1);  // dtype=-1 = preserve
}

// ============================================================
// Tier 5: Scalar Operations
// ============================================================

// REQ-T5-SCA-001: Divide by scalar
void idris_div_scalar(void **out, void *t, double val) {
    scalar s = ats_float(val);
    atg_div_scalar((tensor*)out, (tensor)t, s);
    ats_free(s);
}

// REQ-T5-SCA-002: Multiply by scalar
void idris_mul_scalar(void **out, void *t, double val) {
    scalar s = ats_float(val);
    atg_mul_scalar((tensor*)out, (tensor)t, s);
    ats_free(s);
}

} // extern "C"
