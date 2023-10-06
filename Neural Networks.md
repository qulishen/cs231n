## Lecture 4 Introduction to Neural Networks

1. 有限差分近似

$$
u'(x) = \lim_{\substack{\Delta x \to a}} \frac{u(x+\Delta x)-u(x)}{\Delta x} \approx \frac{u(x+\Delta x)-u(x)}{\Delta x}
$$

2. 边缘填充的0的output（卷积向下取整，池化向上取整）

   $$
   output = (size + 2*pad - filter)/stride + 1
   $$
3. 零填充的意义是为了保持相同的空间大小，过滤器 F = 3 就填充1圈，F=5填充2圈，F=7填充三圈
4. Input volume: 32×32×3

10 5×5 filters with stride1, pad 2

number of parameters in this layer?

->each filter has 5×5×3 +1 =76  (+1 for bias) (3 个通道，每个通道都有一个filter)

->76 * 10 =760

5. 每一个卷积核具有和输入同样的通道数，一个卷积核可以输出一个featuremap；
6. sigmoid函数

   $$
   \sigma(x) = \frac{1}{1+e^x}
   $$
7. Leaky ReLU or PReLU

   $$
   f(x) = max(0.01x,x)
   f(x) = max(\alpha x,x)
   $$
8. PCA
9. Whitening
10. Batch Normalization

    BN一般放在Sigmoid，Tanh之前，这样可以解决饱和区的问题，因为可以分布在0周围，可以缓解梯度消失问题；

    原论文BN放在ReLU之前，但是现在主流放到后面，避免数据在激活函数之前被转化成相似的模式
11. Group Normalization
12.
