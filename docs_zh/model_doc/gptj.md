<!--版权所有2021年HuggingFace团队。保留所有权利。

根据Apache许可证第2.0版（“许可证”）许可；除非符合许可证的规定，
否则您不得使用此文件。您可以在以下位置获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则在许可证下分发的软件是基于“按原样”basis提供的，
没有任何明示或暗示的担保或条件。详见许可证中的特定语言，
以及许可证下的限制。

⚠️请注意，此文件为Markdown格式，但包含了我们的doc-builder（类似于MDX）的特定语法，
在您的Markdown查看器中可能无法正常显示。

-->

# GPT-J

## 概览

GPT-J模型由Ben Wang和Aran Komatsuzaki在[kingoflolz/mesh-transformer-jax](https://github.com/kingoflolz/mesh-transformer-jax)代码库中发布。它是一个基于[GPT-2](https://pile.eleuther.ai/)数据集训练的因果语言模型。

此模型由[Stella Biderman](https://huggingface.co/stellaathena)贡献。

提示：

- 要以float32加载[GPT-J](https://huggingface.co/EleutherAI/gpt-j-6B)，至少需要2倍的模型大小的RAM：
   1倍用于初始化权重，另外1倍用于加载检查点。因此，加载模型至少需要48GB的RAM。为了减少RAM的使用，
   可以使用`torch_dtype`参数仅在CUDA设备上以半精度初始化模型。还有一个fp16分支可存储fp16权重，
   可以用于进一步减少RAM的使用：

```python
>>> from transformers import GPTJForCausalLM
>>> import torch

>>> device = "cuda"
>>> model = GPTJForCausalLM.from_pretrained(
...     "EleutherAI/gpt-j-6B",
...     revision="float16",
...     torch_dtype=torch.float16,
... ).to(device)
```

- 模型应适用于16GB GPU用于推断。对于训练/微调，需要更多GPU RAM。例如，Adam优化器会生成模型的四个副本：
  模型本身、梯度、梯度的平均值和平方平均值。因此，即使使用混合精度，梯度更新仍以fp32表示，
  仍需要至少4倍模型大小的GPU内存。这还不包括激活和数据批处理，这些也需要一些额外的GPU RAM。
  因此，建议使用DeepSpeed等解决方案来训练/微调模型。另一个选项是使用原始代码库在TPU上训练/微调模型，
  然后将模型转换为Transformers格式以进行推断。有关此过程的说明可以在[此处](https://github.com/kingoflolz/mesh-transformer-jax/blob/master/howto_finetune.md)找到。

- 尽管嵌入矩阵的大小为50400，但GPT-2分词器仅使用50257个词条。这些额外的词条是为了在TPUs上提高效率而添加的。
  为了避免嵌入矩阵大小与词汇表大小不匹配，[GPT-J](https://huggingface.co/EleutherAI/gpt-j-6B)的分词器包含了额外的143个词条
  `<|extratoken_1|>... <|extratoken_143|>`，因此分词器的`vocab_size`也变为50400。

### 生成

可以使用[`~generation.GenerationMixin.generate`]方法使用GPT-J模型生成文本。

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
>>> tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

>>> prompt = (
...     "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
...     "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
...     "researchers was the fact that the unicorns spoke perfect English."
... )

>>> input_ids = tokenizer(prompt, return_tensors="pt").input_ids

>>> gen_tokens = model.generate(
...     input_ids,
...     do_sample=True,
...     temperature=0.9,
...     max_length=100,
... )
>>> gen_text = tokenizer.batch_decode(gen_tokens)[0]
```

...或以float16精度执行：

```python
>>> from transformers import GPTJForCausalLM, AutoTokenizer
>>> import torch

>>> device = "cuda"
>>> model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", torch_dtype=torch.float16).to(device)
>>> tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

>>> prompt = (
...     "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
...     "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
...     "researchers was the fact that the unicorns spoke perfect English."
... )

>>> input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

>>> gen_tokens = model.generate(
...     input_ids,
...     do_sample=True,
...     temperature=0.9,
...     max_length=100,
... )
>>> gen_text = tokenizer.batch_decode(gen_tokens)[0]
```

## 资源

以下是官方Hugging Face资源和社区资源（由🌎标识），可帮助您开始使用GPT-J。如果您有兴趣提交资源以包含在此处，请随时打开一个Pull Request，我们将进行审核！资源应该是展示新内容而不是重复现有资源的理想选择。

<PipelineTag pipeline="text-generation"/>

- [GPT-J](https://huggingface.co/EleutherAI/gpt-j-6B)的描述。
- 如何使用Hugging Face Transformers和Amazon SageMaker部署GPT-J 6B进行推断的博客。
- 如何在GPU上使用DeepSpeed-Inference加速GPT-J推断的博客。
- 介绍[GPT-J-6B: 6B JAX-Based Transformer](https://arankomatsuzaki.wordpress.com/2021/06/04/gpt-j/)的博客帖子。 🌎
- [GPT-J-6B推断演示](https://colab.research.google.com/github/kingoflolz/mesh-transformer-jax/blob/master/colab_demo.ipynb)的notebook。 🌎
- 另一个演示[GPT-J-6B推断](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/GPT-J-6B/Inference_with_GPT_J_6B.ipynb)的notebook。  
-  🤗 Hugging Face课程中的[因果语言建模](https://huggingface.co/course/en/chapter7/6?fw=pt#training-a-causal-language-model-from-scratch)章节。
- [`GPTJForCausalLM`]由此[因果语言建模示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#gpt-2gpt-and-causal-language-modeling)、[文本生成示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-generation)和[notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb)支持。
- [`TFGPTJForCausalLM`]由此[因果语言建模示例脚本](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_clmpy)和[notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb)支持。
- [`FlaxGPTJForCausalLM`]由此[因果语言建模示例脚本](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#causal-language-modeling)和[notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/causal_language_modeling_flax.ipynb)支持。

**文档资源**
- [文本分类任务指南](../tasks/sequence_classification)
- [问答任务指南](../tasks/question_answering)
- [因果语言建模任务指南](../tasks/language_modeling)

## GPTJConfig

[[autodoc]] GPTJConfig
    - 所有

## GPTJModel

[[autodoc]] GPTJModel
    - forward

## GPTJForCausalLM

[[autodoc]] GPTJForCausalLM
    - forward

## GPTJForSequenceClassification

[[autodoc]] GPTJForSequenceClassification
    - forward

## GPTJForQuestionAnswering

[[autodoc]] GPTJForQuestionAnswering
    - forward

## TFGPTJModel

[[autodoc]] TFGPTJModel
    - call

## TFGPTJForCausalLM

[[autodoc]] TFGPTJForCausalLM
    - call

## TFGPTJForSequenceClassification

[[autodoc]] TFGPTJForSequenceClassification
    - call

## TFGPTJForQuestionAnswering

[[autodoc]] TFGPTJForQuestionAnswering
    - call

## FlaxGPTJModel

[[autodoc]] FlaxGPTJModel
    - __call__

## FlaxGPTJForCausalLM

[[autodoc]] FlaxGPTJForCausalLM
    - __call__