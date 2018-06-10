# caffeFloat16
私人代码，caffe中float16的训练/推断

* 版本1是将float16的运算嵌入原layers中
* 版本2是将float16的运算独立成另一个layer

它们分别用来干相应的事情：版本1主要用来训练，而版本2则用来refine，针对movidus的项目

没有实现cpu的版本，仅仅gpu。