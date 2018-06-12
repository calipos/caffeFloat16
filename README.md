# caffeFloat16
给自己用的，caffe中float16的训练/推断

* branch1: not pushed yet, 是将float16的运算嵌入原layers中.
* branch2: forMovidus, 是将float16的运算独立成另一个layer.

它们分别用来干相应的事情：版本1主要用来训练，而版本2则用来refine，针对movidus的项目

没有实现cpu的版本，仅仅gpu。

Requires sm_53+ (Pascal TitanX, GTX 1080,1070,1060,1050..， Tesla P4, P40 and others).（GeForce GTX TITAN donot satisfy it.)
