#include <vector>
#include <iostream>
#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
        exit(0);
        std::cout<<"123"<<std::endl;
  const Dtype* bottom_data = bottom[0]->gpu_data();
  bottom[0]->dataFloat2Half();
  const __half* bottom_data_fp16 = (const __half*)bottom[0]->gpuFp16_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  __half* top_data_fp16 = (__half*)top[0]->mutable_gpuFp16_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  this->blobs_[0]->dataFloat2Half();
  const __half* weight_fp16 =(const __half*) this->blobs_[0]->gpuFp16_data();
        std::cout<<"-----------weight_fp16---fw---------"<<std::endl;
    showDeviceHalf(weight_fp16,5);
  if (M_ == 1) {
    caffe_gpu_gemv_half(CblasNoTrans, N_, K_, 1.,
                         weight_fp16, bottom_data_fp16, 0., top_data_fp16);
    if (bias_term_)
    {
        this->blobs_[1]->dataFloat2Half();
        bias_multiplier_.dataFloat2Half();
        caffe_gpu_axpy_half(N_, ((const __half*)bias_multiplier_.gpuFp16_data())[0],
                            (const __half*)this->blobs_[1]->gpuFp16_data(), top_data_fp16);
    }
    
  } else {
    caffe_gpu_gemm_half(CblasNoTrans,
                          transpose_ ? CblasNoTrans : CblasTrans,
                          M_, N_, K_, 1.0,
                          bottom_data_fp16, weight_fp16, 0., top_data_fp16);
    if (bias_term_)
    {
        this->blobs_[1]->dataFloat2Half();
        bias_multiplier_.dataFloat2Half();
      caffe_gpu_gemm_half(CblasNoTrans, CblasNoTrans, M_, N_, 1, 1.,
                            (const __half*)bias_multiplier_.gpuFp16_data(),
                            (const __half*)this->blobs_[1]->gpuFp16_data(), 1., top_data_fp16);
    }
  }
  top[0]->dataHalf2Float();
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    const Dtype* top_diff = top[0]->gpu_diff();
    top[0]->diffFloat2Half();
    const __half* top_diff_fp16 = (const __half*)top[0]->gpuFp16_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    bottom[0]->dataFloat2Half();
    const __half* bottom_data_fp16 = (const __half*)bottom[0]->gpuFp16_data();
  if (this->param_propagate_down_[0]) {
    // Gradient with respect to weight
    if (transpose_) {
      caffe_gpu_gemm_half(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          1., bottom_data_fp16, top_diff_fp16,
          1., (__half*)this->blobs_[0]->mutable_gpuFp16_diff());
    } else {
      caffe_gpu_gemm_half(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          1., top_diff_fp16, bottom_data_fp16,
          1., (__half*)this->blobs_[0]->mutable_gpuFp16_diff());
    }
  }

  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    bias_multiplier_.dataFloat2Half();
    // Gradient with respect to bias
    caffe_gpu_gemv_half(CblasTrans, M_, N_, 1., top_diff_fp16,
        (const __half*)bias_multiplier_.gpuFp16_data(), 1.,
        (__half*)this->blobs_[1]->mutable_gpuFp16_diff());
    this->blobs_[1]->diffHalf2Float();
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_gpu_gemm_half(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          1., top_diff_fp16, (const __half*)this->blobs_[0]->gpuFp16_data(),
          0., (__half*)bottom[0]->mutable_gpuFp16_diff());
    } else {
      caffe_gpu_gemm_half(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
         1., top_diff_fp16, (const __half*)this->blobs_[0]->gpuFp16_data(),
         0., (__half*)bottom[0]->mutable_gpuFp16_diff());
    }
  }
  std::cout<<"-----------weight_fp16-DIFF-----------"<<std::endl;
    showDeviceHalf((__half*)this->blobs_[0]->mutable_gpuFp16_diff(),5);
  this->blobs_[0]->diffHalf2Float();
  bottom[0]->diffHalf2Float();
}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductLayer);

}  // namespace caffe
