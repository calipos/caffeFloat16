#include <vector>
#include <iostream>
#include "caffe/layers/convFp16_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionFp16Layer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //std::cout<<"weight[0] = "<<this->blobs_[0]->cpu_data()[2]<<std::endl;
  const Dtype* weight = this->blobs_[0]->gpu_data();
  
  this->blobs_[0]->dataFloat2Half();

  const __half* weight_fp16 = (const __half*)this->blobs_[0]->gpuFp16_data();
     //std::cout<<"-----------weight_fp16---fw---------"<<std::endl;
    //showDeviceHalf(weight_fp16,10);
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    bottom[i]->dataFloat2Half();
    const __half* bottom_data_fp16 = (const __half*)bottom[i]->gpuFp16_data();
    __half* top_data = (__half*)top[i]->mutable_gpuFp16_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm_half(bottom_data_fp16 + n * this->bottom_dim_, weight_fp16,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->blobs_[1]->dataFloat2Half();
        const __half* bias_fp16 = (const __half*)this->blobs_[1]->gpuFp16_data();
        this->forward_gpu_bias_half(top_data + n * this->top_dim_, bias_fp16);
      }
    }
    top[i]->dataHalf2Float();
  }
}

template <typename Dtype>
void ConvolutionFp16Layer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
      //std::cout<<"================Backward_gpu=============="<<std::endl;
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const __half* weight_fp16 = (const __half*)this->blobs_[0]->gpuFp16_data();
  //std::cout<<"-----------weight_fp16------------"<<std::endl;
    //showDeviceHalf(weight_fp16,this->blobs_[0]->count());
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  __half* weight_diff_fp16 = (__half*)this->blobs_[0]->mutable_gpuFp16_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    top[i]->diffFloat2Half();
    const __half* top_diff_fp16 = (const __half*)top[i]->gpuFp16_diff();
    //std::cout<<"-----------top_diff_fp16------------"<<std::endl;
    //showDeviceHalf(top_diff_fp16,top[i]->count());

    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      __half* bias_diff_fp16 = (__half*)this->blobs_[1]->mutable_gpuFp16_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias_half(bias_diff_fp16, top_diff_fp16 + n * this->top_dim_);
      }
      //std::cout<<"-----------bias_diff_fp16 out------------"<<std::endl;
    //showDeviceHalf(bias_diff_fp16,this->blobs_[1]->count());
      this->blobs_[1]->diffHalf2Float();
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      //bottom[i]->dataFloat2Half();
      const __half* bottom_data_fp16 = (const __half*)bottom[i]->gpuFp16_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      __half* bottom_diff_fp16 = (__half*)bottom[i]->mutable_gpuFp16_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm_half(bottom_data_fp16 + n * this->bottom_dim_,
              top_diff_fp16 + n * this->top_dim_, weight_diff_fp16);
        this->blobs_[0]->diffHalf2Float();
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm_half(top_diff_fp16 + n * this->top_dim_, weight_fp16,
              bottom_diff_fp16 + n * this->bottom_dim_);
        }
      }
    //std::cout<<"-----------weight_diff_fp16 out------------"<<std::endl;
    //showDeviceHalf(weight_diff_fp16,10);
    }
    this->blobs_[0]->diffHalf2Float();
    bottom[i]->diffHalf2Float();
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionFp16Layer);

}  // namespace caffe
