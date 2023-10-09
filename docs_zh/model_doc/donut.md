<!--版权2022年HuggingFace团队。 版权所有。

根据Apache License, Version 2.0 （"许可证"）许可；您除了遵守
许可证，您不得使用此文件。您可以在下面获得许可证副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则在许可证下分发的软件是在
"按原样"的基础上分发的，不附带任何明示或默示的担保或条件。可以查看许可证以获得

⚠️ 请注意，该文件是以Markdown格式编写的，但包含特定的句法，用于我们的doc-builder（类似于MDX），它可能在您的Markdown查看器中无法正常渲染。

特定语言的权限和限制。-->

# Donut

## 概述

Donut模型在[OCR-free Document Understanding Transformer](https://arxiv.org/abs/2111.15664)一文中提出，作者为Geewook Kim、Teakgyu Hong、Moonbin Yim、Jeongyeon Nam、Jinyoung Park、Jinyeong Yim、Wonseok Hwang、Sangdoo Yun、Dongyoon Han和Seunghyun Park。Donut由图像Transformer编码器和自回归文本Transformer解码器组成，用于执行诸如文档图像分类、表单理解和视觉问答等文档理解任务。

论文中的摘要如下：

*理解文档图像（例如发票）是一个核心但具有挑战性的任务，因为它涉及到诸多复杂功能，如文本阅读和对文档的整体理解。当前的视觉文档理解（Visual Document Understanding，VDU）方法将文本阅读任务外包给成熟的光学字符识别（Optical Character Recognition，OCR）引擎，并专注于使用OCR输出进行理解任务。虽然这种基于OCR的方法已经显示出有希望的性能，但它们存在以下问题：1）使用OCR的计算成本高；2）OCR模型在语言或文档类型上的灵活性有限；3）OCR错误向后续过程传播。为了解决这些问题，我们在本文中介绍了一种名为Donut的新型无OCR的VDU模型，Donut代表文档理解转换器。作为无OCR的VDU研究的第一步，我们提出了一个简单的架构（即Transformer），并引入了一个预训练目标（即交叉熵损失）。Donut在概念上简单但有效。通过广泛的实验和分析，我们展示了一个简单的无OCR VDU模型Donut，在速度和准确性方面在各种VDU任务上取得了最先进的性能。此外，我们提供了一个合成数据生成器，帮助模型的预训练能够适用于不同的语言和领域。*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/donut_architecture.jpg"
alt="drawing" width="600"/>

<small> Donut的高级概览。来自<a href="https://arxiv.org/abs/2111.15664">原始论文</a>。 </small>

该模型由[nielsr](https://huggingface.co/nielsr)贡献。原始代码可以在[这里](https://github.com/clovaai/donut)找到。

提示：

- 开始使用Donut的最快方法是检查教程Notebook，展示了如何在推断时使用模型以及在自定义数据上进行微调。
- Donut始终在[VisionEncoderDecoder](vision-encoder-decoder)框架内使用。

## 推理

Donut的[`VisionEncoderDecoder`]模型接受图像作为输入，并使用[`~generation.GenerationMixin.generate`]来自回归地生成给定输入图像的文本。

[`DonutImageProcessor`]类负责对输入图像进行预处理，
[`XLMRobertaTokenizer`/`XLMRobertaTokenizerFast`]解码生成的目标令牌为目标字符串。[`DonutProcessor`]封装了[`DonutImageProcessor`]和[`XLMRobertaTokenizer`/`XLMRobertaTokenizerFast`]，成为一个单一实例，用于提取输入特征并解码预测的令牌id。

- 逐步进行文档图像分类

```py
>>> import re

>>> from transformers import DonutProcessor, VisionEncoderDecoderModel
>>> from datasets import load_dataset
>>> import torch

>>> processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")
>>> model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")

>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> model.to(device)  # doctest: +IGNORE_RESULT

>>> # 读取文档图像
>>> dataset = load_dataset("hf-internal-testing/example-documents", split="test")
>>> image = dataset[1]["image"]

>>> # 准备解码器输入
>>> task_prompt = "<s_rvlcdip>"
>>> decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

>>> pixel_values = processor(image, return_tensors="pt").pixel_values

>>> outputs = model.generate(
...     pixel_values.to(device),
...     decoder_input_ids=decoder_input_ids.to(device),
...     max_length=model.decoder.config.max_position_embeddings,
...     pad_token_id=processor.tokenizer.pad_token_id,
...     eos_token_id=processor.tokenizer.eos_token_id,
...     use_cache=True,
...     bad_words_ids=[[processor.tokenizer.unk_token_id]],
...     return_dict_in_generate=True,
... )

>>> sequence = processor.batch_decode(outputs.sequences)[0]
>>> sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
>>> sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # 移除第一个任务起始标记
>>> print(processor.token2json(sequence))
{'class': 'advertisement'}
```

- 逐步进行文档解析

```py
>>> import re

>>> from transformers import DonutProcessor, VisionEncoderDecoderModel
>>> from datasets import load_dataset
>>> import torch

>>> processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
>>> model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> model.to(device)  # doctest: +IGNORE_RESULT

>>> # 读取文档图像
>>> dataset = load_dataset("hf-internal-testing/example-documents", split="test")
>>> image = dataset[2]["image"]

>>> # 准备解码器输入
>>> task_prompt = "<s_cord-v2>"
>>> decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

>>> pixel_values = processor(image, return_tensors="pt").pixel_values

>>> outputs = model.generate(
...     pixel_values.to(device),
...     decoder_input_ids=decoder_input_ids.to(device),
...     max_length=model.decoder.config.max_position_embeddings,
...     pad_token_id=processor.tokenizer.pad_token_id,
...     eos_token_id=processor.tokenizer.eos_token_id,
...     use_cache=True,
...     bad_words_ids=[[processor.tokenizer.unk_token_id]],
...     return_dict_in_generate=True,
... )

>>> sequence = processor.batch_decode(outputs.sequences)[0]
>>> sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
>>> sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # 移除第一个任务起始标记
>>> print(processor.token2json(sequence))
{'menu': {'nm': 'CINNAMON SUGAR', 'unitprice': '17,000', 'cnt': '1 x', 'price': '17,000'}, 'sub_total': {'subtotal_price': '17,000'}, 'total': {'total_price': '17,000', 'cashprice': '20,000', 'changeprice': '3,000'}}
```

- 逐步进行文档视觉问答（DocVQA）

```py
>>> import re

>>> from transformers import DonutProcessor, VisionEncoderDecoderModel
>>> from datasets import load_dataset
>>> import torch

>>> processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
>>> model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> model.to(device)  # doctest: +IGNORE_RESULT

>>> # 从DocVQA数据集中读取文档图像
>>> dataset = load_dataset("hf-internal-testing/example-documents", split="test")
>>> image = dataset[0]["image"]

>>> # 准备解码器输入
>>> task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
>>> question = "When is the coffee break?"
>>> prompt = task_prompt.replace("{user_input}", question)
>>> decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids

>>> pixel_values = processor(image, return_tensors="pt").pixel_values

>>> outputs = model.generate(
...     pixel_values.to(device),
...     decoder_input_ids=decoder_input_ids.to(device),
...     max_length=model.decoder.config.max_position_embeddings,
...     pad_token_id=processor.tokenizer.pad_token_id,
...     eos_token_id=processor.tokenizer.eos_token_id,
...     use_cache=True,
...     bad_words_ids=[[processor.tokenizer.unk_token_id]],
...     return_dict_in_generate=True,
... )

>>> sequence = processor.batch_decode(outputs.sequences)[0]
>>> sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
>>> sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # 移除第一个任务起始标记
>>> print(processor.token2json(sequence))
{'question': 'When is the coffee break?', 'answer': '11-14 to 11:39 a.m.'}
```

请访问[模型中心](https://huggingface.co/models?filter=donut)查找Donut检查点。

## 训练

请参阅[教程Notebook](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Donut)。

## DonutSwinConfig

[[autodoc]] DonutSwinConfig

## DonutImageProcessor

[[autodoc]] DonutImageProcessor
    - preprocess

## DonutFeatureExtractor

[[autodoc]] DonutFeatureExtractor
    - __call__

## DonutProcessor

[[autodoc]] DonutProcessor
    - __call__
    - from_pretrained
    - save_pretrained
    - batch_decode
    - decode

## DonutSwinModel

[[autodoc]] DonutSwinModel
    - forward