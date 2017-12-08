#import os
#os.chdir('/home/ubuntu/Yongchao/yongchao/lib/python3.6/site-packages');
import mxnet as mx

from mxnet import nd

# print([mxnet.cpu, mxnet.gpu(), mxnet.gpu(1)])

print(mx.__version__);

a = nd.zeros((3,4));
print(a);
a = nd.array([1,2,3], ctx=mx.gpu())
b = nd.zeros((3,2), ctx=mx.gpu())
c = nd.random.uniform(shape=(2,3), ctx=mx.gpu())
print(a,b,c)
b = nd.zeros((3,4), ctx=mx.cpu());
print(b);
