<!--版权 2023 HuggingFace 团队。保留所有权利。

根据 Apache 许可证，版本 2.0 (the "License")，除非符合许可证中的要求，否则不得使用此文件。您可以在以下链接获得许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证发行的软件以"按原样"的方式分发，
无论是明示的还是隐含的，但不限于对适销性和特定用途的适用性的保证。详细了解许可证的特定语言版本，
请参阅许可证下的限制和权利。

⚠️ 请注意，该文件是以 Markdown 格式编写的，但包含我们的 doc-builder 的特定语法（类似于 MDX），
这可能无法在您的 Markdown 查看器中正确渲染。

-->

# Llama2

## 概览

Llama2 模型是由 Hugo Touvron、Louis Martin、Kevin Stone、Peter Albert、Amjad Almahairi、Yasmine Babaei、Nikolay Bashlykov、Soumya Batra、Prajjwal Bhargava、Shruti Bhosale、Dan Bikel、Lukas Blecher、Cristian Canton Ferrer、Moya Chen、Guillem Cucurull、David Esiobu、Jude Fernandes、Jeremy Fu、Wenyin Fu、Brian Fuller、Cynthia Gao、Vedanuj Goswami、Naman Goyal、Anthony Hartshorn、Saghar Hosseini、Rui Hou、Hakan Inan、Marcin Kardas、Viktor Kerkez、Madian Khabsa、Isabel Kloumann、Artem Korenev、Punit Singh Koura、Marie-Anne Lachaux、Thibaut Lavril、Jenya Lee、Diana Liskovich、Yinghai Lu、Yuning Mao、Xavier Martinet、Todor Mihaylov、Pushkar Mishra、Igor Molybog、Yixin Nie、Andrew Poulton、Jeremy Reizenstein、Rashi Rungta、Kalyan Saladi、Alan Schelten、Ruan Silva、Eric Michael Smith、Ranjan Subramanian、Xiaoqing EllenTan、Binh Tang、Ross Taylor、Adina Williams、Jian Xiang Kuan、Puxin Xu、Zheng Yan、Illian Zarov、Yuchen Zhang、Angela Fan、Melanie Kambadur、Sharan Narang、Aurelien Rodriguez、Robert Stojnic、Sergey Edunov 和 Thomas Scialom 提出的，它是一个包含从 70B 到 7B 参数的基础语言模型的集合，通过微调用于对话应用的检查点！

论文摘要如下：

*在这项工作中，我们开发并发布了 Llama 2，这是一系列预训练和微调过的大型语言模型（LLMs）。我们的微调 LLMs，称为 Llama 2-Chat，针对对话使用案例进行了优化。在我们测试的大多数基准中，我们的模型表现优于开源聊天模型，并且根据我们的人工评估，对于有用性和安全性，它们可能是对封闭源模型的合适替代。为了使社区能够在我们的工作基础上进行构建并推动 LLMs 的负责任开发，我们提供了关于我们微调 Llama 2-Chat 和安全性改进的详细说明。*

在此处查看所有 Llama2 模型 [here](https://huggingface.co/models?search=llama2)

<Tip warning={true}>

`Llama2` 模型是使用 `bfloat16` 进行训练的，但原始推断使用的是 `float16`。在 hub 上上传的检查点使用 `torch_dtype = 'float16'`， `AutoModel` API 将使用它将检查点从 `torch.float32` 转换为 `torch.float16`。

在线权重的 `dtype` 大多不相关，除非在初始化模型时使用了 `torch_dtype="auto"`，例如 `model = AutoModelForCausalLM.from_pretrained("path", torch_dtype="auto")`。原因是模型首先会被下载（使用在线检查点的 `dtype`），然后将其转换为 `torch` 的默认 `dtype`（即 `torch.float32`），最后如果配置中提供了 `torch_dtype`，则使用该 `dtype`。

不建议使用 `float16` 对模型进行训练，已知会产生 `nan`，因此建议在 `bfloat16` 上训练模型。

</Tip>

提示：

- 可以通过填写[此表格](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)以获取 Llama2 模型的权重。
- 该架构与第一个 Llama 非常相似，只是增加了 Grouped Query Attention (GQA)，来自这篇[论文](https://arxiv.org/pdf/2305.13245.pdf)。
- 将 `config.pretraining_tp` 设置为与 1 不同的值将激活更准确但较慢的线性层计算，这应该更好地匹配原始对数值。
- 原始模型使用 `pad_id = -1` ，这意味着没有填充标记。我们不能使用相同的逻辑，确保使用 `tokenizer.add_special_tokens({"pad_token":"<pad>"})` 添加一个填充标记并相应调整标记嵌入。您还应该设置 `model.config.pad_token_id`。模型的 `embed_tokens` 层通过 `self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.config.padding_idx)` 初始化，以确保编码填充标记输出为零，因此建议在初始化时传递它。
- 填写表单并获得模型检查点的访问权限后，应能够使用转换后的检查点。否则，如果要转换自己的模型，可以使用[转换脚本](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py)。该脚本可以使用以下（示例）命令调用：

```bash
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
```

- 转换后，可以通过以下方式加载模型和 tokenizer：

```python
from transformers import LlamaForCausalLM, LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained("/output/path")
model = LlamaForCausalLM.from_pretrained("/output/path")
```

请注意，执行脚本需要足够的 CPU RAM 来容纳 float16 精度的整个模型（即使最大版本分为多个检查点，每个检查点都包含部分模型权重，因此我们需要将它们全部加载到 RAM 中）。对于75B模型，需要 145GB RAM。

- LLaMA 分词器是基于 [sentencepiece](https://github.com/google/sentencepiece) 的 BPE 模型。sentencepiece 的一个特殊之处在于，当解码序列时，如果第一个标记是单词的开头（例如"Banana"），分词器不会在字符串前面加上前缀空格。

此模型由 [Arthur Zucker](https://huggingface.co/ArthurZ) 贡献，[Lysandre Debut](https://huggingface.co/lysandre) 也做出了贡献。Hugging Face 中的实现代码基于 GPT-NeoX [here](https://github.com/EleutherAI/gpt-neox)。作者的原始代码可以在 [here](https://github.com/facebookresearch/llama) 找到。

## 资源

以下是 Hugging Face 官方资源和社区（由 🌎 表示）资源列表，可帮助您快速开始使用 LLaMA2。如果您有兴趣提交要包含在此处的资源，请随时提出拉取请求，我们将对其进行评审！该资源应该展示出一些新内容，而不是重复现有资源。

- [Llama 2 is here - get it on Hugging Face](https://huggingface.co/blog/llama2)：一篇关于 Llama 2 及如何使用它与 🤗 Transformers 和 🤗 PEFT 的博客文章。
- [LLaMA 2 - Every Resource you need](https://www.philschmid.de/llama-2)：编译了一些与 LLaMA 2 和如何快速入门有关的相关资源。

<PipelineTag pipeline="text-generation"/>

- 一个[笔记本](https://colab.research.google.com/drive/1PEQyJO1-f6j0S_XJ8DV50NkpzasXkrzd?usp=sharing) ，介绍了如何在 Google Colab 中使用 QLoRA 和 4 位精度微调 Llama 2。🌎
- 一个[笔记本](https://colab.research.google.com/drive/134o_cXcMe_lsvl15ZE_4Y75Kstepsntu?usp=sharing) ，介绍了如何使用 4 位 QLoRA 对 "Llama-v2-7b-guanaco" 模型进行微调并从 PDF 生成问答数据集。🌎

<PipelineTag pipeline="text-classification"/>

- 一个[笔记本](https://colab.research.google.com/drive/1ggaa2oRFphdBmqIjSEbnb_HGkcIRC2ZB?usp=sharing) ，介绍了如何使用 QLoRa、TRL 和韩文文本分类数据集对 Llama 2 模型进行微调。🌎🇰🇷

⚗️ 优化
- [用 DPO 微调 Llama 2](https://huggingface.co/blog/dpo-trl)：使用 TRL 库的 DPO 方法微调 Llama 2 的指南。
- [扩展指南：指令微调 Llama 2](https://www.philschmid.de/instruction-tune-llama-2)：将 Llama 2 训练为根据输入生成指令，将模型从指令遵循转变为指令给予。
- 一个[笔记本](https://colab.research.google.com/drive/1SYpgFpcmtIUzdE7pxqknrM4ArCASfkFQ?usp=sharing) ，介绍了如何使用 QLoRa 和 TRL 在个人计算机上对 Llama 2 模型进行微调。🌎

⚡️ 推理
- 一个[笔记本](https://colab.research.google.com/drive/1TC56ArKerXUpbgRy5vM3woRsbTEVNq7h?usp=sharing) ，介绍了如何使用 AutoGPTQ 库的 GPTQ 对 Llama 2 模型进行量化。🌎
- 一个[笔记本](https://colab.research.google.com/drive/1X1z9Q6domMKl2CnEM0QGHNwidLfR4dW2?usp=sharing) ，介绍了如何在本地计算机或 Google Colab 上使用 4 位量化运行 Llama 2 Chat Model。🌎

🚀 部署
- [在 Amazon SageMaker 上对 LLaMA 2 (7-70B) 进行微调](https://www.philschmid.de/sagemaker-llama2-qlora)：从设置到 QLoRA 微调和部署 Amazon SageMaker 的完整指南。
- [在 Amazon SageMaker 上部署 Llama 2 7B/13B/70B](https://www.philschmid.de/sagemaker-llama-llm)：使用 Hugging Face 的 LLM DLC 容器安全且可扩展地部署。
