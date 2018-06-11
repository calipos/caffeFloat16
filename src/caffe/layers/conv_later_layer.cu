#include <vector>
#include <iostream>
#include "caffe/layers/conv_later_layer.hpp"
#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"




namespace caffe {
//这里的weight都是事先transpos了
template <typename Dtype>
__global__ void generWeight12(const Dtype*  src, const int count,const int N, const int W, const int C,
                                    Dtype* const dest1_data,Dtype*const dest2_data) 
{
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < (count); index += blockDim.x * gridDim.x)
   {
       //H 一直是1
        int temp_idx = index;
        int c=temp_idx%C;
        temp_idx/=C;
        int w=temp_idx%W;
        temp_idx/=W;
        int n=temp_idx%N;
        
        int dim2=C;
        int dim1=dim2*W/3*2;
        
        if(w==0)
        {
            int idx2=n*dim1+c;
            dest2_data[idx2]=src[index];
        }
        else if(w==1)
        {
            int idx1=n*dim1+c;
            int idx2=n*dim1+dim2+c;
            dest1_data[idx1]=src[index];
            dest2_data[idx2]=src[index];
        }
        else if(w==2)
        {
            int idx1=n*dim1+dim2+c;
            dest1_data[idx1]=src[index];
        }
   }
}


template <typename Dtype>
__global__ void transposeKernel0312(Dtype* const tmp, const int count,const int N, const int H, const int W, const int C, 
                                    const int dim1, const int dim2, const int dim3, 
                                    Dtype* const top_data) 
{
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < (count); index += blockDim.x * gridDim.x)
   {
        int temp_idx = index;
        int c=temp_idx%C;
        temp_idx/=C;
        int w=temp_idx%W;
        temp_idx/=W;
        int h=temp_idx%H;
        temp_idx/=H;
        int n=temp_idx%N;
        int newIdx=n*dim1+c*dim2+h*dim3+w;
        top_data[newIdx] = tmp[index];
   }
}

template <typename Dtype>
void ConvolutionLaterLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {


    const Dtype* weight = this->blobs_[0]->gpu_data();
    int weightCount=this->blobs_[0]->count();
    int weightNum=this->blobs_[0]->num();
    int weightWidth=this->blobs_[0]->width();
    int weightChannel=this->blobs_[0]->channels();
    if(this->isWeight12Init==false)
    {
        generWeight12<Dtype><<<CAFFE_GET_BLOCKS(weightCount), CAFFE_CUDA_NUM_THREADS>>>
    (weight, weightCount,weightNum , weightWidth, weightChannel,this->weight1.mutable_gpu_data(),this->weight2.mutable_gpu_data());
    }
    
    
    int kernelLen3=this->blobs_[0]->count()/3;
    std::vector<const Dtype*> weightSet(0);
    weightSet.push_back(weight);
    weightSet.push_back(this->weight1.gpu_data());
    weightSet.push_back(this->weight2.gpu_data());
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* tmp_data = this->output_noTranspose.mutable_gpu_data();

    
    
    
    for(int i=0;i<this->posStrat.size();i++)
    { 
        int weightCol=this->posLength[i]/this->blobs_[0]->num();
        caffe_gpu_gemv<Dtype>(CblasNoTrans, this->blobs_[0]->num(), weightCol,(Dtype)1., weightSet[this->weightType[i]] , bottom_data+posStrat[i], (Dtype)0., tmp_data+i*this->blobs_[0]->num());
        //caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->blobs_[0]->num(),1, weightCol,(Dtype)1., weightSet[this->weightType[i]], bottom_data+posStrat[i],(Dtype)0., tmp_data+i*this->blobs_[0]->num());

        // float*toShow=(float*)malloc(this->blobs_[0]->num()*sizeof(float));
        // cudaMemcpy(toShow,this->output_noTranspose.gpu_data()+i*this->blobs_[0]->num(),this->blobs_[0]->num()*sizeof(float),cudaMemcpyDeviceToHost);
        // for(int k=0;k<this->blobs_[0]->num();k++)
            // std::cout<<toShow[k]<<"  ";
        // std::cout<<"*"<<std::endl;        
        // float*toShowWeight=(float*)malloc(this->posLength[i]*sizeof(float));
        // cudaMemcpy(toShowWeight,weightSet[this->weightType[i]],this->posLength[i]*sizeof(float),cudaMemcpyDeviceToHost);
        // for(int k=0;k<this->posLength[i];k++)
            // std::cout<<toShowWeight[k]<<"  ";
        // std::cout<<"*"<<std::endl;        
        // float*toShowInput=(float*)malloc(weightCol*sizeof(float));
        // cudaMemcpy(toShowInput,bottom_data+posStrat[i],weightCol*sizeof(float),cudaMemcpyDeviceToHost);
        // for(int k=0;k<weightCol;k++)
            // std::cout<<toShowInput[k]<<"  ";
        // std::cout<<"*"<<std::endl;        
        // exit(0);
    }
//std::cout<<"*"<<this->output_noTranspose.shape_string()<<std::endl;
//std::cout<<"*"<<this->output_noTranspose.count()<<std::endl;
    int N=top[0]->num();
    int C=top[0]->channels();
    int H=top[0]->height();
    int W=top[0]->width();
    int count = top[0]->count();
    transposeKernel0312<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
    (tmp_data, count, N, H, W, C, this->dim1, this->dim2, this->dim3, top[0]->mutable_gpu_data());
//std::cout<<"transpose done"<<std::endl;
//std::cout<<"*"<<top[0]->shape_string()<<std::endl;
    CUDA_POST_KERNEL_CHECK;
//std::cout<<"this->bias_term : "<<this->bias_term<<std::endl;
    if (this->bias_term) 
    {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->num_output,
      this->planeCnt, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
      (Dtype)1., top[0]->mutable_gpu_data());
    }
//std::cout<<"Forward_gpu done"<<std::endl;
}


template <typename Dtype>
void ConvolutionLaterLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		LOG(ERROR)<<"not implemented!";
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLaterLayer);

}  // namespace caffe
