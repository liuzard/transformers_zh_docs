<!--
版权所有2023年HuggingFace团队保留。

根据Apache许可证第2.0版（“许可证”）授权；除非符合许可证的规定，否则不得使用此文件。你可以在以下网址获得许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是“按原样”分发的，不提供任何明示或暗示的保证或条件。请参阅许可证获取具体的语言许可及限制。

注意，此文件是Markdown格式，但包含了特定的语法，以用于我们的文档构建器（类似于MDX），可能在你的Markdown查看器中无法正确显示。

-->

# Persimmon

## 概述

Persimmon模型由[ADEPT](https://www.adept.ai/blog/persimmon-8b)创建，作者是Erich Elsen，Augustus Odena，Maxwell Nye，Sağnak Taşırlar，Tri Dao，Curtis Hawthorne，Deepak Moparthi，Arushi Somani。

作者介绍了Persimmon-8B，这是一个基于经典transformers架构、具有查询和键归一化的解码器模型。Persimmon-8B是一个完全具备许可的模型，拥有大约80亿个参数，根据Apache许可证发布。Persimmon-8B的一些关键特性包括长的上下文大小（16K）、性能和多模态扩展的能力。

作者展示了他们的模型评估方法，重点是实践中的文本生成，模拟用户与语言模型的交互方式。该研究还包括了对比分析，将Persimmon-8B与其他知名模型（MPT 7B Instruct和Llama 2 Base 7B 1-Shot）在各种评估任务上进行对比。结果表明，尽管训练数据有限，Persimmon-8B的竞争性能表现出色。

在模型细节方面，该研究概述了Persimmon-8B的架构和训练方法，提供了有关其设计选择、序列长度和数据集组成的见解。作者提供了一个快速的推理代码，通过运算符融合和CUDA图形利用超越了传统的实现，同时保持代码的一致性。他们表示期待社区如何利用这一贡献推动创新，并暗示将作为持续发展系列的一部分推出更多即将到来的版本。

<Tip warning={true}>

`Persimmon`模型是使用`bfloat16`进行训练的，但原始推理使用的是`float16`。上传到hub上的检查点使用`torch_dtype = 'float16'`，`AutoModel` API将把检查点从`torch.float32`转换为`torch.float16`。

在线权重的`dtype`通常是不相关的，除非在初始化模型时使用`torch_dtype="auto"`。原因是模型首先将被下载（使用在线检查点的`dtype`），然后会转换为`torch`的默认`dtype`（变为`torch.float32`）。用户应该指定他们想要的`torch_dtype`，如果不指定，将使用`torch.float32`。

不建议在`float16`下微调模型，由于已知会产生`nan`，因此模型应该在`bfloat16`下进行微调。

</Tip>

提示：

- 转换模型需要使用`git clone https://github.com/persimmon-ai-labs/adept-inference`克隆原始仓库，然后获取检查点：

```bash
git clone https://github.com/persimmon-ai-labs/adept-inference
wget https://axtkn4xl5cip.objectstorage.us-phoenix-1.oci.customer-oci.com/n/axtkn4xl5cip/b/adept-public-data/o/8b_base_model_release.tar
tar -xvf 8b_base_model_release.tar
python src/transformers/models/persimmon/convert_persimmon_weights_to_hf.py  --input_dir /path/to/downloaded/persimmon/weights/ --output_dir /output/path \
    --pt_model_path /path/to/8b_chat_model_release/iter_0001251/mp_rank_00/model_optim_rng.pt
    --ada_lib_path /path/to/adept-inference
```

对于聊天模型：
```bash
wget https://axtkn4xl5cip.objectstorage.us-phoenix-1.oci.customer-oci.com/n/axtkn4xl5cip/b/adept-public-data/o/8b_chat_model_release.tar
tar -xvf 8b_base_model_release.tar
```

之后，可以通过以下方式加载模型：

```py
from transformers import PersimmonForCausalLM, PersimmonTokenizer

model = PersimmonForCausalLM.from_pretrained("/output/path")
tokenizer = PersimmonTokenizer.from_pretrained("/output/path")
```

此模型由[ArthurZ](https://huggingface.co/ArthurZ)贡献。原始代码可以在[这里](https://github.com/persimmon-ai-labs/adept-inference)找到。

- Perismmon使用基于`sentencepiece`的tokenizer，使用`Unigram`模型。它支持bytefallback，在`tokenizers==0.14.0`版本的快速tokenizer中才可用。
`LlamaTokenizer`被用作它是一个标准的sentencepiece包装器。`chat`模板将在后续的PR中使用模板函数进行更新！

- 作者建议在聊天模式下使用以下提示格式：`f"human: {prompt}\n\nadept:"`

## PersimmonConfig

[[autodoc]] PersimmonConfig

## PersimmonModel

[[autodoc]] PersimmonModel
    - forward

## PersimmonForCausalLM

[[autodoc]] PersimmonForCausalLM
    - forward

## PersimmonForSequenceClassification

[[autodoc]] PersimmonForSequenceClassification
    - forward