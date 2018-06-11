#ifndef CAFFE_CONV_LATER_LAYER_HPP_
#define CAFFE_CONV_LATER_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_conv_layer.hpp"

namespace caffe {


template <typename Dtype>
class ConvolutionLaterLayer : public BaseConvolutionLayer<Dtype> {
 public:

  explicit ConvolutionLaterLayer(const LayerParameter& param)
      : BaseConvolutionLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "ConvolutionLater"; }

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  
 protected:
  Blob<Dtype> bias_multiplier_;
  Blob<Dtype> output_noTranspose;
  Blob<Dtype> weight1;
  Blob<Dtype> weight2;
  std::vector<int> posStrat;
  std::vector<int> posLength;
  std::vector<int> weightType;
  std::vector<const Dtype*> weightPtr;
  Dtype*weight0;//只有后半截
  bool isWeight12Init;
  int kernel_w;
  int kernel_h;
  int stride_w;
  int stride_h;
  int pad_w;
  int pad_h;
  int num_output;
  int planeCnt;
  bool bias_term;
  vector<int> weight_shape;
  vector<int> weight12_shape;
  vector<int> bias_shape;
  
  int dim3;//用以对top transpose
  int dim2;
  int dim1;
 
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual inline bool reverse_dimensions() { return false; }
  virtual void compute_output_shape();
};

}  // namespace caffe

#endif  // CAFFE_CONV_LAYER_HPP_
