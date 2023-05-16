//#include <torch/all.h>
//#include <torch/python.h>
//#include <c10/cuda/CUDAGuard.h>
//#include <cuda_fp16.h//#include <torch/types.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "column_remap.h"
#include "q4v2_matmul.h"
#include "q4v2_recons.h"
#include "q4v2_sequential.h"

// v1

//void vecquant4matmul_v1_cuda(torch::Tensor vec, torch::Tensor mat, torch::Tensor mul, torch::Tensor scales, torch::Tensor zeros);
//void vecquant4matmul_v1(torch::Tensor vec, torch::Tensor mat, torch::Tensor mul, torch::Tensor scales, torch::Tensor zeros)
//{
//  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
//  vecquant4matmul_v1_cuda(vec, mat, mul, scales, zeros);
//}
//
//void vecquant4recons_v1_cuda(torch::Tensor mat, torch::Tensor res, torch::Tensor scales, torch::Tensor zeros);
//void vecquant4recons_v1(torch::Tensor mat, torch::Tensor res, torch::Tensor scales, torch::Tensor zeros)
//{
//  const at::cuda::OptionalCUDAGuard device_guard(device_of(scales));
//  vecquant4recons_v1_cuda(mat, res, scales, zeros);
//}

// v2

void q4v2_matmul
(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor out,
    torch::Tensor w_scales,
    torch::Tensor w_zeros,
    torch::Tensor seq_g_idx,
    torch::Tensor x_map
)
{
    TORCH_CHECK(x.dtype() == torch::kHalf, "x must be a half tensor");
    TORCH_CHECK(w.dtype() == torch::kInt, "w must be an int (q4) tensor");
    TORCH_CHECK(out.dtype() == torch::kHalf, "out must be a half tensor");
    TORCH_CHECK(w_scales.dtype() == torch::kHalf, "w_scales must be a half tensor");
    TORCH_CHECK(w_zeros.dtype() == torch::kInt, "w_zeros must be an int tensor");
    TORCH_CHECK(x.size(1) == w.size(0) * 8, "x and w have incompatible shapes");
    TORCH_CHECK(x.size(1) % 256 == 0, "x.shape[1] must be multiple of 256");
    TORCH_CHECK(seq_g_idx.device().is_meta() || seq_g_idx.size(0) == w.size(0) * 2 * 8, "seq_g_idx and w have incompatible shapes");
    TORCH_CHECK(x_map.device().is_meta() || x_map.size(0) == w.size(0) * 8, "x_map and w have incompatible shapes");

    int groupsize = w.size(0) * 8 / w_zeros.size(0);
    TORCH_CHECK(groupsize * w_zeros.size(0) == w.size(0) * 8, "w.shape[-2] must be a multiple of zeros.shape[-2]")

    const at::cuda::OptionalCUDAGuard device_guard(device_of(w_scales));

    int height = x.size(0);
    int dim = x.size(1);
    int width = w.size(1);

    q4v2_matmul_cuda
    (
        (half*) x.data_ptr(),
        (uint32_t*) w.data_ptr(),
        (half*) out.data_ptr(),
        (half*) w_scales.data_ptr(),
        (uint32_t*) w_zeros.data_ptr(),
        height,
        dim,
        width,
        groupsize,
        seq_g_idx.device().is_meta() ? NULL : (uint16_t*) seq_g_idx.data_ptr(),
        x_map.device().is_meta() ? NULL : (uint32_t*) x_map.data_ptr()
    );
}


void q4v2_recons
(
    torch::Tensor w,
    torch::Tensor out,
    torch::Tensor w_scales,
    torch::Tensor w_zeros,
    torch::Tensor seq_g_idx
)
{
    TORCH_CHECK(w.dtype() == torch::kInt, "w must be an int (q4) tensor");
    TORCH_CHECK(out.dtype() == torch::kHalf, "out must be a half tensor");
    TORCH_CHECK(w_scales.dtype() == torch::kHalf, "w_scales must be a half tensor");
    TORCH_CHECK(w_zeros.dtype() == torch::kInt, "w_zeros must be an int tensor");

    int groupsize = w.size(0) * 8 / w_zeros.size(0);
    TORCH_CHECK(groupsize * w_zeros.size(0) == w.size(0) * 8, "w.shape[0] must be a multiple of zeros.shape[0]")

    const at::cuda::OptionalCUDAGuard device_guard(device_of(w_scales));

    int height = w.size(0);
    int width = w.size(1);

    q4v2_recons_cuda
    (
        (uint32_t*) w.data_ptr(),
        (half*) out.data_ptr(),
        (half*) w_scales.data_ptr(),
        (uint32_t*) w_zeros.data_ptr(),
        height,
        width,
        groupsize,
        seq_g_idx.device().is_meta() ? NULL : (uint16_t*) seq_g_idx.data_ptr()
    );
}


void q4v2_sequential
(
    torch::Tensor w,
    torch::Tensor g_idx,        // size: w_height * 8
    torch::Tensor seq_g_idx,    // size: w_height * 8 * 2
    torch::Tensor x_map,        // size: w_height * 8
    const int num_groups
)
{
    TORCH_CHECK(w.dtype() == torch::kInt, "w must be an int (q4) tensor");
    TORCH_CHECK(g_idx.dtype() == torch::kInt, "g_idx must be an int tensor");
    TORCH_CHECK(seq_g_idx.dtype() == torch::kShort, "seq_g_idx must be a short tensor");
    TORCH_CHECK(x_map.dtype() == torch::kInt, "x_map must be an int tensor");
    TORCH_CHECK(g_idx.size(0) == x_map.size(0), "x_map must be same shape as g_idx");
    TORCH_CHECK(seq_g_idx.size(0) == g_idx.size(0) * 2, "seq_g_idx must be twice as wide as g_idx");
    TORCH_CHECK(g_idx.size(0) == w.size(0) * 8, "g_idx and w have incompatible shapes");

    int height = w.size(0);
    int width = w.size(1);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(w));

    q4v2_sequential_cuda
    (
        (uint32_t*) w.data_ptr(),
        height,
        width,
        (uint32_t*) g_idx.data_ptr(),
        (uint16_t*) seq_g_idx.data_ptr(),
        (uint32_t*) x_map.data_ptr(),
        num_groups
    );
}


void column_remap
(
    torch::Tensor x,
    torch::Tensor x_new,
    torch::Tensor x_map
)
{
    TORCH_CHECK(x.dtype() == torch::kHalf, "x must be an half tensor");
    TORCH_CHECK(x_new.dtype() == torch::kHalf, "x_new must be half tensor");
    TORCH_CHECK(x_map.dtype() == torch::kInt, "x_map must be an int tensor");
    TORCH_CHECK(x_map.size(0) == x.size(1), "x_map and x have incompatible shapes");

    int height = x.size(0);
    int width = x.size(1);

    column_remap_cuda
    (
        (half*) x.data_ptr(),
        (half*) x_new.data_ptr(),
        height,
        width,
        (uint32_t*) x_map.data_ptr()
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
//  m.def("vecquant4matmul_v1", &vecquant4matmul_v1, "Vector 4-bit Quantized Matrix Multiplication (CUDA) v1");
//  m.def("vecquant4recons_v1", &vecquant4recons_v1, "Vector 4-bit Quantized Matrix Reconstruction (CUDA) v1");
  m.def("q4v2_matmul", &q4v2_matmul, "q4v2 matrix multiplication");
  m.def("q4v2_recons", &q4v2_recons, "q4v2 matrix reconstruction");
  m.def("q4v2_sequential", &q4v2_sequential, "q4v2 matrix serialization");
  m.def("column_remap", &column_remap, "half matrix column remapping");
}