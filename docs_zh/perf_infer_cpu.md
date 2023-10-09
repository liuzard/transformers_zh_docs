<!--
版权所有2022年The HuggingFace团队。保留所有权利。

根据Apache License 2.0许可（“许可证”），您不得使用此文件，除非符合许可证的规定。
您可以在以下网址获得许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是按"原样"分发的，
没有任何明示或暗示的担保或条件。请参阅许可证了解具体条款和条件。

⚠️请注意，此文件是Markdown格式，但包含了特定语法，用于我们的文档构建器（类似于MDX），可能无法在您的Markdown查看器中正确渲染。
-->

# 在CPU上高效运行推断

本指南关注的是在CPU上高效推理大型模型的方法。

## 更快的推理使用`BetterTransformer`

我们最近在文本、图像和音频模型的CPU上集成了`BetterTransformer`以实现更快的推理。有关这种集成的文档，请单击[此处](https://huggingface.co/docs/optimum/bettertransformer/overview)获取更多详情。

## PyTorch JIT模式（TorchScript）
TorchScript是一种从PyTorch代码创建可序列化和可优化模型的方法。任何TorchScript程序都可以保存在Python进程中，并加载到没有Python依赖项的进程中。
与默认的即时执行模式相比，PyTorch中的JIT模式通常通过操作融合等优化方法在模型推理方面提供更好的性能。

有关TorchScript的简要介绍，请参阅[PyTorch TorchScript教程的介绍部分](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html#tracing-modules)。

### 使用JIT模式进行的IPEX图优化
Intel® PyTorch扩展（Intel® Extension for PyTorch）为Transformers系列模型的JIT模式提供了进一步的优化。强烈建议用户在JIT模式下充分利用Intel® PyTorch扩展的优势。Intel® PyTorch扩展在JIT模式下已经支持了一些Transformers模型的常用操作模式的融合，如多头注意力融合、连接线性层、线性+加法、线性+Gelu、加法+层归一化等。这些融合模式已经启用，并具有良好的性能。融合的收益以透明的方式提供给用户。根据分析，对于问题回答、文本分类和标记分类中的约70％的最流行的自然语言处理任务，这些融合模式都可以在Float32精度和BFloat16混合精度上带来性能优势。

请点击[此处](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/features/graph_optimization.html)查看更详细的IPEX图优化信息。

#### IPEX安装：

IPEX发布版本与PyTorch保持一致，请查看[IPEX安装方法](https://intel.github.io/intel-extension-for-pytorch/)。

### JIT模式的使用
要在Trainer中启用JIT模式进行评估或预测，用户应在Trainer命令参数中添加`jit_mode_eval`。

<Tip warning={true}>

对于PyTorch >= 1.14.0，JIT模式可以使任何模型在预测和评估方面受益，因为jit.trace支持字典输入。

对于PyTorch < 1.14.0，JIT模式可以使与jit.trace中的元组输入顺序匹配的模型受益，例如问题回答模型。
当前jit.trace无法处理正向参数顺序与jit.trace中的元组输入顺序不匹配的情况，例如文本分类模型。我们在此处通过异常捕获来进行回退，并使用日志记录通知用户。

</Tip>

以[Transformers问题回答](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering)为例，下面是使用示例：

- 在CPU上使用jit模式进行推理：
<pre>python run_qa.py \
--model_name_or_path csarron/bert-base-uncased-squad-v1 \
--dataset_name squad \
--do_eval \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir /tmp/ \
--no_cuda \
<b>--jit_mode_eval </b></pre> 

- 在CPU上使用IPEX和jit模式进行推理：
<pre>python run_qa.py \
--model_name_or_path csarron/bert-base-uncased-squad-v1 \
--dataset_name squad \
--do_eval \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir /tmp/ \
--no_cuda \
<b>--use_ipex \</b>
<b>--jit_mode_eval</b></pre> 
