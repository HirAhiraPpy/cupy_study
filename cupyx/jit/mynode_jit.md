# My Note
## _compile.py

jit首先要获取到装饰器wrap的函数体代码，然后暴力翻译成cuda c代码（构造一个[`Generated`对象](https://github.com/HirAhiraPpy/cupy_study/blob/bydong/code_reading/cupyx/jit/_compile.py#L156)）

似乎支持nvcc和nvrtc(jitify)两种编译方式/backend。

装饰器把函数转换成[`_JitRawKernal`](https://github.com/HirAhiraPpy/cupy_study/blob/bydong/code_reading/cupyx/jit/_interface.py#L42)对象。
调用[`code.compile_with_cache()`](https://github.com/HirAhiraPpy/cupy_study/blob/bydong/code_reading/cupyx/jit/_interface.py#L111C27-L111C45)（实现在[此处](https://github.com/HirAhiraPpy/cupy_study/blob/bydong/code_reading/cupy/_core/core.pyx#L2217)）完成编译+缓存。

实际执行编译的是[`cupy.cuda.compile._compile_module_with_cache`](https://github.com/HirAhiraPpy/cupy_study/blob/bydong/code_reading/cupy/cuda/compiler.py#L508)

推测`@cupyx.jit.RawKernal()`的流程是：
1. 拿到函数体文本，解析函数为cuda代码（不知道怎么解析调用其他函数的）
2. 调用编译器，编译为kernel module，加上hash后缓存
3. 判断缓存是否和hash一致，拿到编译后的module，调用函数名对应的符号
