/*
Copyright (2024) Bytedance Ltd. and/or its affiliates
*/
#include <torch/types.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

//#include "fpA_intB_gemm/launchers/fpA_intB_launcher_sm90.inl"
//#include "fpA_intB_gemm/fpA_intB_gemm_template.h"
#include "fpA_intB_gemm/fpA_intB_gemm.h"

#include "cutlass_preprocessors.h"
#include "cutlass/numeric_types.h"
#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#include <vector>
#endif  // CUDA_VERSION >= 11000

namespace th = ::torch;
using namespace ::tensorrt_llm::kernels::cutlass_kernels;

template <typename T>
inline T* get_ptr(th::Tensor& t) {
  return reinterpret_cast<T*>(t.data_ptr());
}
template <typename T>
inline const T* get_ptr(const th::Tensor& t) {
  return reinterpret_cast<const T*>(t.data_ptr());
}

class ITrtllmFpAIntBGemm {
 public:
  ITrtllmFpAIntBGemm() {}
  virtual ~ITrtllmFpAIntBGemm() {}

  // asymm
  /*
  virtual void forward_res(const th::Tensor& A, const th::Tensor& B,
                           th::Tensor& C, const th::Tensor& scale,
                           const th::Tensor& zp, const th::Tensor& bias,
                           const th::Tensor& res, const int64_t m,
                           const int64_t n, const int64_t k,
                           int group_size) = 0;

  virtual void forward_gelu(const th::Tensor& A, const th::Tensor& B,
                            th::Tensor& C, const th::Tensor& scale,
                            const th::Tensor& zp, const th::Tensor& bias,
                            const int64_t m, const int64_t n, const int64_t k,
                            int group_size) = 0;
  */
  virtual void forward(const th::Tensor& A, const th::Tensor& B, th::Tensor& C,
                       const th::Tensor& scale, const th::Tensor& zp,
                       const th::Tensor& bias, const int64_t m, const int64_t n,
                       const int64_t k, int group_size) = 0;

};

  
template <typename T, typename WeightType, cutlass::WeightOnlyQuantOp QuantOp>
class TrtllmFpAIntBGemm : public ITrtllmFpAIntBGemm {
 public:
  TrtllmFpAIntBGemm() {}

  ~TrtllmFpAIntBGemm() override {}

  // asymm funcs
  /*
  void forward_res(const th::Tensor& A, const th::Tensor& B, th::Tensor& C,
                   const th::Tensor& scale, const th::Tensor& zp,
                   const th::Tensor& bias, const th::Tensor& res,
                   const int64_t m, const int64_t n, const int64_t k,
                   int group_size) override {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const T* input_act_ptr = get_ptr<const T>(A);
    const WeightType* weight_ptr = get_ptr<const WeightType>(B);
    const T* scales_ptr = get_ptr<const T>(scale);
    const T* zp_ptr = get_ptr<const T>(zp);
    const T* bias_ptr = get_ptr<const T>(bias);
    const T* res_ptr = get_ptr<const T>(res);
    // how to use ?
    const int64_t ws_bytes = fused_gemm_dq_runner.getWorkspaceSize(m, n, k);
    auto ws_tensor =
        th::empty({ws_bytes},
                  th::dtype(th::kInt8).device(th::kCUDA).requires_grad(false));

    T* output_tensor_ptr = get_ptr<T>(C);
    char* ws_ptr = get_ptr<char>(ws_tensor);

    
    fused_gemm_dq_runner.gemm(input_act_ptr, weight_ptr, scales_ptr, zp_ptr,
                              bias_ptr, res_ptr, output_tensor_ptr, m, n, k,
                              group_size, ActivationType::Identity, ws_ptr,
                              ws_bytes, stream);
    
  }

  void forward_gelu(const th::Tensor& A, const th::Tensor& B, th::Tensor& C,
                    const th::Tensor& scale, const th::Tensor& zp,
                    const th::Tensor& bias, const int64_t m, const int64_t n,
                    const int64_t k, int group_size) override {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const T* input_act_ptr = get_ptr<const T>(A);
    const WeightType* weight_ptr = get_ptr<const WeightType>(B);
    const T* scales_ptr = get_ptr<const T>(scale);
    const T* zp_ptr = get_ptr<const T>(zp);
    const T* bias_ptr = get_ptr<const T>(bias);
    // how to use ?
    const int64_t ws_bytes = fused_gemm_dq_runner.getWorkspaceSize(m, n, k);
    auto ws_tensor =
        th::empty({ws_bytes},
                  th::dtype(th::kInt8).device(th::kCUDA).requires_grad(false));

    T* output_tensor_ptr = get_ptr<T>(C);
    char* ws_ptr = get_ptr<char>(ws_tensor);
    
    fused_gemm_dq_runner.gemm(input_act_ptr, weight_ptr, scales_ptr, zp_ptr,
                              bias_ptr, nullptr, output_tensor_ptr, m, n, k,
                              group_size, ActivationType::Gelu, ws_ptr,
                              ws_bytes, stream);
    
  }
  */
  void forward(const th::Tensor& A, const th::Tensor& B, th::Tensor& C,
               const th::Tensor& scale, const th::Tensor& zp,
               const th::Tensor& bias, const int64_t m, const int64_t n,
               const int64_t k, int group_size) override {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const T* input_act_ptr = get_ptr<const T>(A);
    const WeightType* weight_ptr = get_ptr<const WeightType>(B);
    const T* scales_ptr = get_ptr<const T>(scale);
    const T* zp_ptr = get_ptr<const T>(zp);
    const T* bias_ptr = get_ptr<const T>(bias);
    const T* res_ptr = nullptr;
    // how to use ?
    const int64_t ws_bytes = fused_gemm_dq_runner.getWorkspaceSize(m, n, k);
    auto ws_tensor =
        th::empty({ws_bytes},
                  th::dtype(th::kInt8).device(th::kCUDA).requires_grad(false));

    T* output_tensor_ptr = get_ptr<T>(C);
    char* ws_ptr = get_ptr<char>(ws_tensor);

    auto configs = fused_gemm_dq_runner.getConfigs();
    configs[0].stages = 3;

    fused_gemm_dq_runner.gemm(input_act_ptr, weight_ptr, scales_ptr, zp_ptr,
                              bias_ptr, output_tensor_ptr, m, n, k,
                              group_size, configs[0], ws_ptr,
                              ws_bytes, stream);
  }

 private:
  CutlassFpAIntBGemmRunner<T, WeightType, QuantOp> fused_gemm_dq_runner;
};

// w2 interface
th::Tensor dQ_asymm_qw2_gemm(const th::Tensor& A, const th::Tensor& B,
                                  const th::Tensor& scale, const th::Tensor& zp,
                                  const th::Tensor& bias, int group_size) {
  /*
  CHECK_TH_CUDA(A);
  CHECK_CONTIGUOUS(B);
  CHECK_TH_CUDA(B);
  CHECK_CONTIGUOUS(A);
  CHECK_TH_CUDA(scale);
  CHECK_CONTIGUOUS(scale);
  CHECK_TH_CUDA(zp);
  CHECK_CONTIGUOUS(zp);
  */

  int64_t m = 1;
  const int64_t n = B.size(1) * 4;
  const int64_t k = A.size(-1);
  std::vector<int64_t> out_shape;
  for (int64_t i = 0; i < A.dim() - 1; ++i) {
    m *= A.size(i);
    out_shape.push_back(A.size(i));
  }
  out_shape.push_back(n);

  auto compute_dtype = A.scalar_type();

  /*
  PTH_ENFORCE(compute_dtype == at::ScalarType::Half ||
                  compute_dtype == at::ScalarType::BFloat16,
              "compute type only support fp16 and bf16 !!!");
  PTH_ENFORCE(compute_dtype == scale.scalar_type(), "invalid scale dtype");
  PTH_ENFORCE(compute_dtype == zp.scalar_type(), "invalid zp dtype");
  PTH_ENFORCE(compute_dtype == bias.scalar_type(), "invalid bias dtype");
  */

  std::unique_ptr<ITrtllmFpAIntBGemm> qgemm;

  if (compute_dtype == at::ScalarType::Half) {
    qgemm = std::make_unique<TrtllmFpAIntBGemm<
                half, cutlass::uint2b_t,
                cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS>>();
  }
#if CUDA_VERSION >= 11000
  else if (compute_dtype == at::ScalarType::BFloat16) {
    qgemm = std::make_unique<TrtllmFpAIntBGemm<
                __nv_bfloat16, cutlass::uint2b_t,
                cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS>>();
  }
#endif
  else {
    std::string err_msg =
        "Unsupported compute type " + std::string(at::toString(compute_dtype));
    throw std::runtime_error(err_msg);
  }

  auto output_tensor = th::empty(
      out_shape,
      th::dtype(compute_dtype).device(th::kCUDA).requires_grad(false));
  qgemm->forward(A, B, output_tensor, scale, zp, bias, m, n, k, group_size);

  return output_tensor;
}

/*
th::Tensor dQ_asymm_qw2_gemm_res(const th::Tensor& A, const th::Tensor& B,
                                      const th::Tensor& scale,
                                      const th::Tensor& zp,
                                      const th::Tensor& bias,
                                      const th::Tensor& res, int group_size) {
  
  CHECK_TH_CUDA(A);
  CHECK_CONTIGUOUS(B);
  CHECK_TH_CUDA(B);
  CHECK_CONTIGUOUS(A);
  CHECK_TH_CUDA(scale);
  CHECK_CONTIGUOUS(scale);
  CHECK_TH_CUDA(zp);
  CHECK_CONTIGUOUS(zp);
  CHECK_TH_CUDA(res);
  CHECK_CONTIGUOUS(res);
  

  int64_t m = 1;
  const int64_t n = B.size(1) * 4;
  const int64_t k = A.size(-1);
  std::vector<int64_t> out_shape;
  for (int64_t i = 0; i < A.dim() - 1; ++i) {
    m *= A.size(i);
    out_shape.push_back(A.size(i));
  }
  out_shape.push_back(n);
  auto compute_dtype = A.scalar_type();

  
  PTH_ENFORCE(compute_dtype == at::ScalarType::Half ||
                  compute_dtype == at::ScalarType::BFloat16,
              "compute type only support fp16 and bf16 !!!");
  

  std::unique_ptr<ITrtllmFpAIntBGemm> qgemm;

  if (compute_dtype == at::ScalarType::Half) {
    qgemm  = std::make_unique<TrtllmFpAIntBGemm<
        half, cutlass::uint2b_t,
        cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS>>();
  }
#if CUDA_VERSION >= 11000
  else if (compute_dtype == at::ScalarType::BFloat16) {
    qgemm = std::make_unique<TrtllmFpAIntBGemm<
        __nv_bfloat16, cutlass::uint2b_t,
        cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS>>();
  }
#endif
  else {
    std::string err_msg =
        "Unsupported compute type " + std::string(at::toString(compute_dtype));
    throw std::runtime_error(err_msg);
  }

  auto output_tensor = th::empty(
      out_shape,
      th::dtype(compute_dtype).device(th::kCUDA).requires_grad(false));
  qgemm->forward_res(A, B, output_tensor, scale, zp, bias, res, m, n, k,
                     group_size);

  return output_tensor;
}

th::Tensor dQ_asymm_qw2_gemm_gelu(const th::Tensor& A, const th::Tensor& B,
                                       const th::Tensor& scale,
                                       const th::Tensor& zp,
                                       const th::Tensor& bias, int group_size) {
  
  CHECK_TH_CUDA(A);
  CHECK_CONTIGUOUS(B);
  CHECK_TH_CUDA(B);
  CHECK_CONTIGUOUS(A);
  CHECK_TH_CUDA(scale);
  CHECK_CONTIGUOUS(scale);
  CHECK_TH_CUDA(zp);
  CHECK_CONTIGUOUS(zp);
  

  const int64_t m = A.size(0);
  const int64_t n = B.size(1) * 4;
  const int64_t k = A.size(1);
  auto compute_dtype = A.scalar_type();

  
  PTH_ENFORCE(compute_dtype == at::ScalarType::Half ||
                  compute_dtype == at::ScalarType::BFloat16,
              "compute type only support fp16 and bf16 !!!");
  

  std::unique_ptr<ITrtllmFpAIntBGemm> qgemm;

  if (compute_dtype == at::ScalarType::Half) {
    qgemm = std::make_unique<TrtllmFpAIntBGemm<
        half, cutlass::uint2b_t,
        cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS>>();
  }
#if CUDA_VERSION >= 11000
  else if (compute_dtype == at::ScalarType::BFloat16) {
    qgemm = std::make_unique<TrtllmFpAIntBGemm<
        __nv_bfloat16, cutlass::uint2b_t,
        cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS>>();
  }
#endif
  else {
    std::string err_msg =
        "Unsupported compute type " + std::string(at::toString(compute_dtype));
    throw std::runtime_error(err_msg);
  }

  auto output_tensor = th::empty(
      {m, n}, th::dtype(compute_dtype).device(th::kCUDA).requires_grad(false));
  qgemm->forward_gelu(A, B, output_tensor, scale, zp, bias, m, n, k,
                      group_size);

  return output_tensor;
}
*/
// preprocess weight
th::Tensor dQ_preprocess_weights_int2_for_weight_only(
    const th::Tensor& in) {
  const int8_t* in_ptr = reinterpret_cast<const int8_t*>(in.data_ptr());
  auto dims = in.dim();
  std::vector<size_t> shape;
  for (int i = 0; i < dims; i++) {
    shape.push_back(in.size(i));
  }
  //auto dtype = QuantType::PACKED_INT2_WEIGHT_ONLY;
  auto dtype = QuantType::W2_A16;
  std::vector<int8_t> in_int2_pack;
  const size_t num_experts = shape.size() == 2 ? 1 : shape[0];
  const size_t num_rows = shape.size() == 2 ? shape[0] : shape[1];
  const size_t num_cols = shape.size() == 2 ? shape[1] : shape[2];

  const int bits_in_type = 2;
  const int bytes_per_out_col = num_cols * bits_in_type / 8;

  std::vector<int8_t> weight_buf(num_experts * num_rows * num_cols);
  auto unprocessed_quantized_weight = weight_buf.data();

  const int input_mat_size = num_rows * num_cols;
  const int quantized_mat_size = num_rows * bytes_per_out_col;
  const float quant_range_scale = 1.f / float(1 << (bits_in_type - 1));

  for (int expert = 0; expert < num_experts; ++expert) {
    const int8_t* current_weight = in_ptr + expert * input_mat_size;
    int8_t* current_quantized_weight =
        unprocessed_quantized_weight + expert * quantized_mat_size;
    // Finally, construct the weights.
    for (int ii = 0; ii < num_rows; ++ii) {
      int8_t* current_quantized_weight_row =
          current_quantized_weight + ii * bytes_per_out_col;
      const int8_t* current_weight_row = current_weight + ii * num_cols;
      for (int jj = 0; jj < bytes_per_out_col; ++jj) {
        // We will pack 4 int2 elements per iteration of the inner loop.
        int8_t packed_int2s = 0;
        for (int packed_idx = 0; packed_idx < 4; ++packed_idx) {
          const int input_idx = 4 * jj + packed_idx;
          if (input_idx < num_cols) {
            packed_int2s |=
                ((current_weight_row[input_idx] & 0x03) << (2 * packed_idx));
          }
        }
        current_quantized_weight_row[jj] = packed_int2s;
      }
    }
  }
  th::Tensor out =
      th::zeros({(int)num_experts, (int)num_rows, (int)num_cols / 4})
          .to(th::kInt8)
          .cpu();
  int8_t* out_ptr = reinterpret_cast<int8_t*>(out.data_ptr());
  preprocess_weights_for_mixed_gemm(out_ptr, unprocessed_quantized_weight,
                                    shape, dtype);

  if (shape.size() == 2) {
    out = out.view({(int)num_rows, (int)num_cols / 4});
  }
  return out;
}

/*
th::Tensor dQ_asymm_qw2_gemm(const th::Tensor& A, const th::Tensor& B,
                                  const th::Tensor& scale, const th::Tensor& zp,
                                  const th::Tensor& bias, int group_size)
th::Tensor dQ_preprocess_weights_int2_for_weight_only(
    const th::Tensor& in)
*/

PYBIND11_MODULE(decoupleQ_kernels, m) {
  m.def(
    "dQ_asymm_qw2_gemm",
    &dQ_asymm_qw2_gemm,
    "weight only int2 gemm for asymm quant"
  );
  m.def(
    "dQ_preprocess_weights_int2_for_weight_only",
    &dQ_preprocess_weights_int2_for_weight_only,
    "preprocess weight before weight only int2 gemm run"
  );
}
/*
PYBIND11_MODULE(DecoupleQ_kernels, m) { 
}
*/
