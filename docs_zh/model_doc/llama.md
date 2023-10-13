<!--版权© 2022 HuggingFace团队。保留所有权利。

根据Apache License, Version 2.0许可证（"许可证"）进行许可；除非符合许可证的要求，否则不能使用此文件。你可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件基于"按原样"的基础分发，没有任何明示或暗示的保证或条件。请参阅许可证了解特定语言下的权限和限制。

⚠️注意，此文件采用Markdown格式，但包含特定语法以用于文档构建器（类似于MDX），可能无法在Markdown查看器中正确显示。-->

# LLaMA

## 概述

LLaMA模型是由Hugo Touvron、Thibaut Lavril、Gautier Izacard、Xavier Martinet、Marie-Anne Lachaux、Timothée Lacroix、Baptiste Rozière、Naman Goyal、Eric Hambro、Faisal Azhar、Aurelien Rodriguez、Armand Joulin、Edouard Grave和Guillaume Lample在论文[LLaMA：Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)中提出的。 它是一个包括从7B到65B参数的基础语言模型的集合。

论文中的摘要如下：

*我们介绍了LLaMA，这是一个包括从7B到65B参数的基础语言模型的集合。我们使用公开可用的数据集进行模型训练，而不使用专有和不可访问的数据集，并证明可以训练出最先进的模型。特别是，LLaMA-13B在大多数基准测试中胜过GPT-3（175B），LLaMA-65B与最佳模型Chinchilla-70B和PaLM-540B相当。我们将所有的模型都发布给研究社区。*

提示：

- 可以通过填写[此表单](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform?usp=send_form)获取LLaMA模型的权重。
- 下载权重后，需要使用[转换脚本](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py)将其转换为Hugging Face Transformers格式。可以使用以下命令调用脚本（示例）：

```bash
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
```

- 转换后，可以通过以下方式加载模型和分词器：

```python
from transformers import LlamaForCausalLM, LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained("/output/path")
model = LlamaForCausalLM.from_pretrained("/output/path")
```

请注意，执行脚本需要足够的CPU RAM以存储整个模型的float16精度（即使最大版本是由多个检查点组成的，每个检查点都包含模型的一部分权重，因此我们需要加载它们的全部内容）。对于65B模型，需要130GB的RAM。

- LLaMA分词器是基于[sentencepiece](https://github.com/google/sentencepiece)的BPE模型。sentencepiece的一个特殊之处是，在解码序列时，如果第一个标记是单词的开头（例如"Banana"），分词器不会在字符串前添加前缀空格。

此模型由[zphang](https://huggingface.co/zphang)贡献，并得到[BlackSamorez](https://huggingface.co/BlackSamorez)的贡献。Hugging Face实现的代码基于GPT-NeoX的代码[此处](https://github.com/EleutherAI/gpt-neox)。原始作者的代码可以在[此处](https://github.com/facebookresearch/llama)找到。


基于原始LLaMA模型，Meta AI发布了一些后续作品：

- **Llama2**：Llama2是Llama的改进版本，具有一些架构调整（Grouped Query Attention），并预训练2万亿个标记。详细信息可参见[Llama2](llama2)的文档。

## 资源

以下是官方Hugging Face和社区（由🌎标示）资源列表，可帮助你入门LLaMA。如果你有兴趣提供资源以包含在此处，请随时发起拉取请求，我们将进行评审！该资源应该最好展示一些新的东西，而不是重复现有的资源。

<PipelineTag pipeline="text-classification"/>

- 一份关于如何使用prompt tuning来适应LLaMA模型进行文本分类任务的[笔记本](https://colab.research.google.com/github/bigscience-workshop/petals/blob/main/examples/prompt-tuning-sst2.ipynb#scrollTo=f04ba4d2)。🌎

<PipelineTag pipeline="question-answering"/>

- [StackLLaMA：使用RLHF训练LLaMA的实践指南](https://huggingface.co/blog/stackllama#stackllama-a-hands-on-guide-to-train-llama-with-rlhf)，一篇关于如何使用RLHF在[Stack Exchange](https://stackexchange.com/)上训练LLaMA来回答问题的博文。

⚗️ 优化
- 一份关于如何使用xturing库在有限的GPU内存上微调LLaMA模型的[笔记本](https://colab.research.google.com/drive/1SQUXq1AMZPSLD4mk3A3swUIc6Y2dclme?usp=sharing)。🌎

⚡️ 推理
- 一份关于如何使用🤗PEFT库中的PeftModel运行LLaMA模型的[笔记本](https://colab.research.google.com/github/DominguesM/alpaca-lora-ptbr-7b/blob/main/notebooks/02%20-%20Evaluate.ipynb)。🌎
- 一份关于如何使用LangChain加载PEFT适配器LLaMA模型的[笔记本](https://colab.research.google.com/drive/1l2GiSSPbajVyp2Nk3CFT4t3uH6-5TiBe?usp=sharing)。🌎

🚀 部署
- 一份关于如何使用🤗PEFT库通过LoRA方法微调LLaMA模型的[笔记本](https://colab.research.google.com/github/lxe/simple-llama-finetuner/blob/master/Simple_LLaMA_FineTuner.ipynb#scrollTo=3PM_DilAZD8T)。🌎
- 一份关于如何在Amazon SageMaker上部署用于文本生成的Open-LLaMA模型的[笔记本](https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/jumpstart-foundation-models/text-generation-open-llama.ipynb)。🌎

## LlamaConfig

[[autodoc]] LlamaConfig


## LlamaTokenizer

[[autodoc]] LlamaTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## LlamaTokenizerFast

[[autodoc]] LlamaTokenizerFast
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - update_post_processor
    - save_vocabulary

## LlamaModel

[[autodoc]] LlamaModel
    - forward


## LlamaForCausalLM

[[autodoc]] LlamaForCausalLM
    - forward

## LlamaForSequenceClassification

[[autodoc]] LlamaForSequenceClassification
    - forward