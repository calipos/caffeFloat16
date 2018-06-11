#ifdef WITH_PYTHON_LAYER
#include "boost/python.hpp"
namespace bp = boost::python;
#endif

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <tuple>
#include <fstream>
#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"
#include "caffe/util/math_functions.hpp"
using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using caffe::NetParameter;
using caffe::LayerParameter;
using caffe::caffe_scal;
using caffe::caffe_powx;
using caffe::caffe_mul;
using caffe::caffe_add;
using caffe::caffe_div;
using caffe::caffe_copy;
using caffe::caffe_add_scalar;

using std::tuple;
using std::make_tuple;
using std::ostringstream;




int merge_bn()
{

	  Caffe::set_mode(Caffe::CPU);
      std::vector<std::string> list= caffe::LayerRegistry<float>::LayerTypeList();
      for(auto d : list) std::cout<<d<<std::endl;

	return 0;
}

int main()
{
	merge_bn();
	return 0;
}
