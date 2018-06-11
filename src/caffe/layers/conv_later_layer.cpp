#include <vector>
#include <iostream>


#include "caffe/filler.hpp"
#include "caffe/layers/conv_later_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLaterLayer<Dtype>::compute_output_shape() {
//not needed
}

template <typename Dtype>
void ConvolutionLaterLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  this->isWeight12Init=false;
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
  CHECK(kernel_h==1);
  CHECK(kernel_w==3);    //only support
  CHECK(1==conv_param.has_stride_h());
  CHECK(1==conv_param.has_stride_w());
  CHECK(0==conv_param.stride_size());
  stride_h=conv_param.stride_h();
  stride_w=conv_param.stride_w();
  CHECK(stride_h==1);
  CHECK(stride_w==1 || stride_w==2);//only support
  CHECK(1==conv_param.has_pad_h());
  CHECK(1==conv_param.has_pad_w());
  CHECK(0==conv_param.pad_size());
  pad_h=conv_param.pad_h();
  pad_w=conv_param.pad_w();
  CHECK(pad_h==0);
  CHECK(pad_w==1 || pad_w==0);//only support
  CHECK(1==this->layer_param_.convolution_param().group());
  CHECK(conv_param.dilation_size()==0);
  num_output=this->layer_param_.convolution_param().num_output();
  bias_term = this->layer_param_.convolution_param().bias_term(); 

  weight_shape.resize(4);//weight的nchw排列方式并没有变化，只是里面的数据排列其实是[nhcw]
  bias_shape.resize(2);
  weight_shape[0]=num_output;
  weight_shape[1]=bottom[0]->channels();
  weight_shape[2]=1;
  weight_shape[3]=kernel_w;
  
  weight12_shape.resize(4);//weight的nchw排列方式并没有变化，只是里面的数据排列其实是[nhcw]
  weight12_shape[0]=num_output;
  weight12_shape[1]=bottom[0]->channels();
  weight12_shape[2]=1;
  weight12_shape[3]=kernel_w/3*2;
  
  bias_shape[0]=bias_term;
  bias_shape[1]=num_output;
 if (this->blobs_.size() > 0) {
    CHECK_EQ(1 + bias_term, this->blobs_.size())<< "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      LOG(FATAL) << "Incorrect weight shape: expected shape ("<< weight_shape[0]<<" "<< weight_shape[1]<<" "<< weight_shape[2]<<" "<< weight_shape[3]<< "); instead, shape was "<< this->blobs_[0]->shape_string();
    }
    if (bias_term && bias_shape != this->blobs_[1]->shape()) {
      LOG(FATAL) << "Incorrect bias shape: expected shape （"<<  bias_shape[0]<<" "<< bias_shape[1] << "); instead, shape was "<< this->blobs_[1]->shape_string();
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

    if (bias_term) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }

 
}

template <typename Dtype>
void ConvolutionLaterLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    std::vector<int> top_shape{(int)bottom.size(),this->num_output};
    int input_H=bottom[0]->height();
    int input_W=bottom[0]->width();
    top_shape.push_back(input_H);//必然和input保持一致
    int outW=(input_W+2*this->pad_w-this->kernel_w)/this->stride_w+1;
    top_shape.push_back(outW);
    
    this->dim3=top_shape[3];
    this->dim2=top_shape[2]*this->dim3;
    this->dim1=top_shape[1]*this->dim2;

  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }
  this->output_noTranspose.Reshape(top_shape);
  //weight has changed  (NCHW)  TO (NHWC)//里面的数据排列变了，但是维度的意义 没有变
  int totalKernelLen=this->blobs_[0]->count();
  int kernelLen3=this->blobs_[0]->count()/3;
  int kernelLen23=kernelLen3*2;
  
  //input has changed  (NCHW)  TO (NHWC)//里面的数据排列变了，但是维度的意义 没有变
  int inputSliceLen=bottom[0]->width()*bottom[0]->channels();//输入张量的一层，每一层的start要重算，后面的只需要在前面的基础上 +  this->blobs_[0]->count()/3;
  int step=this->blobs_[0]->count(1)/3*(this->stride_w);
  int pad_step=this->blobs_[0]->count(1)/3*(this->pad_w);
  //LOG(INFO)<<"*"<<input_H;
  //LOG(INFO)<<"*"<<outW;
    weightType.clear();//用后半截的权值
    posStrat.clear();
    posLength.clear();
  for(int j=0;j<input_H;j++)
  {
    for(int i=0;i<outW;i++)
    {
        if(i==0)
        {
            if(this->pad_w==1)
            {
                weightType.push_back(1);//用后半截的权值
                posStrat.push_back(inputSliceLen*j);
                posLength.push_back(kernelLen23);
            }
            else if(this->pad_w==0)
            {
                weightType.push_back(0);
                posStrat.push_back(inputSliceLen*j);
                posLength.push_back(totalKernelLen);
            }
        }
        else 
        {
            if(this->pad_w==1)
            {
                posStrat.push_back(inputSliceLen*j+i*step-pad_step);
                if((i*step-pad_step)+totalKernelLen/this->blobs_[0]->num()>inputSliceLen)
                {
                    weightType.push_back(2);
                    posLength.push_back(kernelLen23);
                }
                else
                {
                    weightType.push_back(0);
                    posLength.push_back(totalKernelLen);                
                }
            }
            else if(this->pad_w==0)
            {
            weightType.push_back(0);
            posStrat.push_back(inputSliceLen*j+i*step);
            posLength.push_back(totalKernelLen);
            }
        }
    }
  }
    // for(int i=0;i<weightType.size();i++)
    // {
        // if(i%5==0) std::cout<<std::endl;
        // std::cout<<weightType[i]<<" ";
    // }
    // LOG(INFO)<<"-----";
    // for(int i=0;i<weightType.size();i++)
    // {
        // if(i%5==0) std::cout<<std::endl;
        // std::cout<<posStrat[i]<<" ";
    // }
        // LOG(INFO)<<"-----";
    // for(int i=0;i<weightType.size();i++)
    // {
        // if(i%5==0) std::cout<<std::endl;
        // std::cout<<posLength[i]<<" ";
    // }
   // LOG(INFO)<<"*"<<weightType.size();
    // LOG(INFO)<<"*"<<posStrat.size();
    // LOG(INFO)<<"*"<<posLength.size();
    // exit(0);
  
  planeCnt=top[0]->height()*top[0]->width();
  if (this->bias_term) 
  {
    vector<int> bias_multiplier_shape(1, planeCnt);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1.0),
        bias_multiplier_.mutable_cpu_data());
  }
  weight1.Reshape(weight12_shape);
  weight2.Reshape(weight12_shape);
  
}





template <typename Dtype>
void ConvolutionLaterLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
		LOG(ERROR)<<"not implemented!";
}

template <typename Dtype>
void ConvolutionLaterLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		LOG(ERROR)<<"not implemented!";
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLaterLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLaterLayer);
REGISTER_LAYER_CLASS(ConvolutionLater);
}  // namespace caffe
