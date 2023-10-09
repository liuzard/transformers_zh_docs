<!--版权 2021 NVIDIA Corporation和HuggingFace团队。保留所有权利。

根据Apache License第2版（“许可证”），您不得使用此文件，除非符合许可证。您可以在以下位置获取许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是按“原样”基础分发的，不附带任何形式的明示或暗示的担保或条件。请参阅许可证的特定语言，以了解许可证下的特定权限和限制。

⚠️ 请注意，此文件采用Markdown格式，但包含特定于我们的文档构建器（类似于MDX）的语法，您的Markdown查看器可能无法正确渲染。

-->

# QDQBERT

## 概述

QDQBERT模型可以在[Hao Wu, Patrick Judd, Xiaojie Zhang, Mikhail Isaev和Paulius Micikevicius的《整数量化用于深度学习推理的原理和经验评估》](https://arxiv.org/abs/2004.09602)中找到。

论文的摘要如下：

*量化技术可以通过利用高吞吐量整数指令来减小深度神经网络的尺寸，提高推理延迟和吞吐量。在本文中，我们回顾了量化参数的数学方面，并对不同应用领域（包括视觉、语音和语言）的各种神经网络模型的选择进行了评估。我们重点研究了适合于高吞吐量整数数学流水线加速的量化技术。我们还提出了一种8位量化的工作流程，该工作流程能够在所有研究的网络上保持浮点基线的精度在1%以内，包括更难量化的模型，如MobileNets和BERT-large。*

提示：

- QDQBERT模型在BERT模型中的(i)线性层输入和权重、(ii)矩阵乘法输入、(iii)残差相加输入上添加伪量化操作（对）。 

- QDQBERT需要[Pytorch量化工具包](https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization)的依赖项。安装使用`pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com`

- QDQBERT模型可以从HuggingFace BERT模型（例如*bert-base-uncased*）的任何检查点加载，并进行量化感知训练/后向量化。

- 使用QDQBERT模型执行SQUAD任务的量化感知训练和后向量化的完整示例可在[transformers/examples/research_projects/quantization-qdqbert/](examples/research_projects/quantization-qdqbert/)中找到。

此模型由[shangz](https://huggingface.co/shangz)提供。


### 设置默认量化器

QDQBERT模型通过[NVIDIA的Pytorch量化工具包](https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization)中的`TensorQuantizer`添加伪量化操作（QuantizeLinear/DequantizeLinear操作对）到BERT中。 `TensorQuantizer`是用于对张量进行量化的模块，其中`QuantDescriptor`定义了张量的量化方式。详细信息请参阅[Pytorch量化工具包用户指南](https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/userguide.html)。

在创建QDQBERT模型之前，需要设置默认的`QuantDescriptor`以定义默认的张量量化器。

示例：

```python
>>> import pytorch_quantization.nn as quant_nn
>>> from pytorch_quantization.tensor_quant import QuantDescriptor

>>> # 将默认张量量化器设置为使用最大校准方法
>>> input_desc = QuantDescriptor(num_bits=8, calib_method="max")
>>> # 将默认张量量化器设置为对权重进行每通道量化
>>> weight_desc = QuantDescriptor(num_bits=8, axis=((0,)))
>>> quant_nn.QuantLinear.set_default_quant_desc_input(input_desc)
>>> quant_nn.QuantLinear.set_default_quant_desc_weight(weight_desc)
```

### 校准

校准是将数据样本传递给量化器，并确定张量的最佳缩放因子的术语。在设置了张量量化器之后，可以使用以下示例来对模型进行校准：

```python
>>> # 查找TensorQuantizer并启用校准
>>> for name, module in model.named_modules():
...     if name.endswith("_input_quantizer"):
...         module.enable_calib()
...         module.disable_quant()  # 使用全精度数据进行校准

>>> # 提供数据样本
>>> model(x)
>>> # ...

>>> # 完成校准
>>> for name, module in model.named_modules():
...     if name.endswith("_input_quantizer"):
...         module.load_calib_amax()
...         module.enable_quant()

>>> # 如果在GPU上运行，因为校准过程会创建新的张量，所以需要再次调用.cuda（）
>>> model.cuda()

>>> # 继续运行量化模型
>>> # ...
```

### 导出到ONNX

导出到ONNX的目标是通过[TensorRT](https://developer.nvidia.com/tensorrt)部署推理。伪量化将被拆分成一对QuantizeLinear/DequantizeLinear ONNX操作。在设置TensorQuantizer的静态成员以使用Pytorch自己的伪量化函数之后，可以将伪量化模型导出到ONNX，按照[torch.onnx](https://pytorch.org/docs/stable/onnx.html)中的说明执行。示例：

```python
>>> from pytorch_quantization.nn import TensorQuantizer

>>> TensorQuantizer.use_fb_fake_quant = True

>>> # 加载校准过的模型
>>> ...
>>> # 进行ONNX导出
>>> torch.onnx.export(...)
```

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)
- [因果语言模型任务指南](../tasks/language_modeling)
- [掩码语言模型任务指南](../tasks/masked_language_modeling)
- [多选任务指南](../tasks/multiple_choice)

## QDQBertConfig

[[autodoc]] QDQBertConfig

## QDQBertModel

[[autodoc]] QDQBertModel
    - forward

## QDQBertLMHeadModel

[[autodoc]] QDQBertLMHeadModel
    - forward

## QDQBertForMaskedLM

[[autodoc]] QDQBertForMaskedLM
    - forward

## QDQBertForSequenceClassification

[[autodoc]] QDQBertForSequenceClassification
    - forward

## QDQBertForNextSentencePrediction

[[autodoc]] QDQBertForNextSentencePrediction
    - forward

## QDQBertForMultipleChoice

[[autodoc]] QDQBertForMultipleChoice
    - forward

## QDQBertForTokenClassification

[[autodoc]] QDQBertForTokenClassification
    - forward

## QDQBertForQuestionAnswering

[[autodoc]] QDQBertForQuestionAnswering
    - forward