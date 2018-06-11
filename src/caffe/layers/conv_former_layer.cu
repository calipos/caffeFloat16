#include <vector>
//#include <iostream>
#include "caffe/layers/conv_former_layer.hpp"
#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"




namespace caffe {
//这里的weight都是事先transpos了
template <typename Dtype>
__global__ void generWeight12_h(const Dtype*  src, const int count,const int N, const int H, const int C,
                                    Dtype* const dest1_data,Dtype*const dest2_data) 
{
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < (count); index += blockDim.x * gridDim.x)
   {
       //H 一直是1
        int temp_idx = index;
        int c=temp_idx%C;
        temp_idx/=C;
        int h=temp_idx%H;
        temp_idx/=H;
        int n=temp_idx%N;
        
        int dim2=C;
        int dim1=dim2*H/3*2;
        
        if(h==0)
        {
            int idx2=n*dim1+c;
            dest2_data[idx2]=src[index];
        }
        else if(h==1)
        {
            int idx1=n*dim1+c;
            int idx2=n*dim1+dim2+c;
            dest1_data[idx1]=src[index];
            dest2_data[idx2]=src[index];
        }
        else if(h==2)
        {
            int idx1=n*dim1+dim2+c;
            dest1_data[idx1]=src[index];
        }
   }
}


template <typename Dtype>
__global__ void transposeKernel0321(const Dtype* tmp, const int count,const int N, const int H, const int W, const int C, 
                                    const int dim1, const int dim2, const int dim3, 
                                    Dtype* const top_data) 
{
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < (count); index += blockDim.x * gridDim.x)
   {
        int temp_idx = index;

        int w=temp_idx%W;        temp_idx/=W;
        int h=temp_idx%H;        temp_idx/=H;
        int c=temp_idx%C;        temp_idx/=C;
        int n=temp_idx%N;

        int newIdx=n*dim1+w*dim2+h*dim3+c;
        top_data[newIdx] = tmp[index];
   }
}

template <typename Dtype>
void ConvolutionFormerLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
          
    Dtype* tmp_data = top[0]->mutable_gpu_data();

	for(int i=0;i<this->blobs_[0]->num();i++)	{
		caffe_gpu_gemm<Dtype>(CblasTrans,CblasNoTrans,
		bottom[0]->channels(),bottom[0]->height()*bottom[0]->width(),bottom[0]->channels(),
		(Dtype)1.,bottom[0]->gpu_data(),this->weight1.gpu_data(),(Dtype)0.,
		tmp_data);
		caffe_gpu_gemm<Dtype>(CblasTrans,CblasNoTrans,
		bottom[0]->channels(),bottom[0]->height()*bottom[0]->width(),bottom[0]->channels(),
		(Dtype)1.,bottom[0]->gpu_data(),this->weight2.gpu_data(),(Dtype)0.,
		tmp_data);
		caffe_gpu_gemm<Dtype>(CblasTrans,CblasNoTrans,
		bottom[0]->channels(),bottom[0]->height()*bottom[0]->width(),bottom[0]->channels(),
		(Dtype)1.,bottom[0]->gpu_data(),this->weight3.gpu_data(),(Dtype)0.,
		tmp_data);
	}


    
	}


template <typename Dtype>
void ConvolutionFormerLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		//std::cout<<"not implemented!"<<std::endl;
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionFormerLayer);

}  // namespace caffe
