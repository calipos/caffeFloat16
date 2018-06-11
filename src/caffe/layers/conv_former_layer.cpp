#include <vector>
#include <iostream>


#include "caffe/filler.hpp"
#include "caffe/layers/conv_former_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionFormerLayer<Dtype>::compute_output_shape() {
//not needed
}

template <typename Dtype>
void ConvolutionFormerLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.

  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  bool force_nd_im2col = conv_param.force_nd_im2col();
  CHECK(0==force_nd_im2col);
  CHECK(bottom[0]->num()==1);    //only support
  CHECK(1==bottom[0]->CanonicalAxisIndex(conv_param.axis()));
  CHECK(1==conv_param.has_kernel_h());
  CHECK(1==conv_param.has_kernel_w());
  CHECK(0==conv_param.kernel_size_size());
  kernel_h=conv_param.kernel_h();
  kernel_w=conv_param.kernel_w();
  CHECK(kernel_h==3);    //only support
  CHECK(kernel_w==1);    //only support
  CHECK(1==conv_param.has_stride_h());
  CHECK(1==conv_param.has_stride_w());
  CHECK(0==conv_param.stride_size());
  stride_h=conv_param.stride_h();
  stride_w=conv_param.stride_w();
  CHECK(stride_h==1 || stride_h==2);//only support
  CHECK(stride_w==1);//only support
  CHECK(1==conv_param.has_pad_h());
  CHECK(1==conv_param.has_pad_w());
  CHECK(0==conv_param.pad_size());
  pad_h=conv_param.pad_h();
  pad_w=conv_param.pad_w();
  CHECK(pad_h==1 || pad_h==0);//only support
  CHECK(pad_w==0);
  CHECK(1==this->layer_param_.convolution_param().group());
  CHECK(conv_param.dilation_size()==0);
  num_output=this->layer_param_.convolution_param().num_output();
  bias_term = this->layer_param_.convolution_param().bias_term(); 
  LOG(INFO)<<bias_term;
  CHECK(!bias_term);

  weight_shape.resize(4);//weight的nchw排列方式并没有变化，只是里面的数据排列其实是[nhcw]

  weight_shape[0]=num_output;
  weight_shape[1]=bottom[0]->channels();
  weight_shape[2]=kernel_h;
  weight_shape[3]=1;


 if (this->blobs_.size() > 0) {
    CHECK_EQ(1 + bias_term, this->blobs_.size())<< "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      LOG(FATAL) << "Incorrect weight shape: expected shape ("<< weight_shape[0]<<" "<< weight_shape[1]<<" "<< weight_shape[2]<<" "<< weight_shape[3]<< "); instead, shape was "<< this->blobs_[0]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
  }

 
}

template <typename Dtype>
void ConvolutionFormerLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    std::vector<int> top_shape{(int)bottom.size()};
    top_shape.push_back(this->num_output);
    int input_H=bottom[0]->height();
    int input_W=bottom[0]->width();
    int outH=(input_H+2*this->pad_h-this->kernel_h)/this->stride_h+1;
    top_shape.push_back(outH);
    top_shape.push_back(input_W);//必然和input保持一致
    
    

  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }

  this->weight1.Reshape(1, bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
  this->weight2.Reshape(1, bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
  this->weight3.Reshape(this->blobs_[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
  
  
}





template <typename Dtype>
void ConvolutionFormerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
		LOG(ERROR)<<"not implemented!";
}

template <typename Dtype>
void ConvolutionFormerLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		LOG(ERROR)<<"not implemented!";
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionFormerLayer);
#endif

INSTANTIATE_CLASS(ConvolutionFormerLayer);
REGISTER_LAYER_CLASS(ConvolutionFormer);
}  // namespace caffe
