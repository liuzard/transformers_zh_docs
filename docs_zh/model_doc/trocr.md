<!--版权所有2021年HuggingFace团队。保留所有权利。

根据Apache许可证2.0版（“许可证”），除非符合该许可，否则不得使用此文件。
你可以在以下位置获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是按原样分发的，
没有任何明示或暗示的担保或条件。请参阅许可证的具体语言以及许可的权限和限制。

⚠️ 请注意，这个文件是Markdown格式的，但包含了我们文档生成器的特定语法（类似于MDX），
可能无法在你的Markdown查看器中正确显示。

在许可证下，特定语言管理权限和限制。 -->

# TrOCR

## 概述

TrOCR模型是由Minghao Li、Tengchao Lv、Lei Cui、Yijuan Lu、Dinei Florencio、Cha Zhang、Zhoujun Li、Furu Wei在
论文[TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://arxiv.org/abs/2109.10282)中
提出的。TrOCR包含一个图像Transformer编码器和一个自回归文本Transformer解码器，
用于执行[光学字符识别（OCR）](https://en.wikipedia.org/wiki/Optical_character_recognition)。

论文摘要如下：

*文本识别是一项长期存在的文件数字化研究问题。现有的文本识别方法通常基于卷积神经网络（CNN）进行图像理解，基于循环神经网络（RNN）进行字符级文本生成。
此外，在整体准确性上通常需要另一个语言模型作为后处理步骤来改进。在本文中，我们提出了一种使用预训练图像Transformer和文本Transformer模型的端到端文本识别方法，称为TrOCR，
它利用Transformer架构进行图像理解和字级文本生成。TrOCR模型简单而有效，并且可以使用大规模合成数据进行预训练，并使用人工标记的数据集进行微调。实验证明，TrOCR模型在印刷和手写文本识别任务上性能优于当前最先进的模型。*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/trocr_architecture.jpg"
alt="drawing" width="600"/>

<small> TrOCR架构. 来自<a href="https://arxiv.org/abs/2109.10282">原始论文</a>。 </small>

请参阅[`VisionEncoderDecoder`]类中如何使用此模型。

此模型由[nielsr](https://huggingface.co/nielsr)贡献。原始代码可以在[这里](https://github.com/microsoft/unilm/tree/6f60612e7cc86a2a1ae85c47231507a587ab4e01/trocr)找到。

提示：

- 快速开始使用TrOCR的方法是查看[教程笔记本](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/TrOCR)，
其中展示了如何在推理时使用模型以及在自定义数据上进行微调。
- TrOCR在被微调之前会经过2个阶段的预训练。它在印刷（例如[SROIE数据集](https://paperswithcode.com/dataset/sroie)）和手写（例如[IAM Handwriting 数据集](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database>)）文本识别任务上取得了最先进的结果。更多信息请参阅[官方模型](https://huggingface.co/models?other=trocr>)。
- TrOCR始终与[VisionEncoderDecoder](vision-encoder-decoder)框架一起使用。

## 资源

官方Hugging Face和社区（由🌎表示）资源列表，以帮助你开始使用TrOCR。如果你有兴趣提交资源并包含在此处，请随时提交Pull Request，我们将进行审核！该资源应该展示一些新的东西，而不是重复现有的资源。

<PipelineTag pipeline="text-classification"/>

- 有关如何加速文档AI的博客文章[Accelerating Document AI](https://huggingface.co/blog/document-ai)与TrOCR。
- 如何使用TrOCR博客文章[Document AI](https://github.com/philschmid/document-ai-transformers)与TrOCR。
- 如何[使用Seq2SeqTrainer在IAM Handwriting Database上微调TrOCR](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_Seq2SeqTrainer.ipynb)的笔记本。
- 使用TrOCR进行推理的笔记本[Inference with TrOCR](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Inference_with_TrOCR_%2B_Gradio_demo.ipynb)和Gradio演示。
- [使用原生PyTorch在IAM Handwriting Database上微调TrOCR](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_native_PyTorch.ipynb)的笔记本。
- [评估IAM测试集上的TrOCR](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Evaluating_TrOCR_base_handwritten_on_the_IAM_test_set.ipynb)的笔记本。

<PipelineTag pipeline="text-generation"/>

- [Casual language modeling](https://huggingface.co/docs/transformers/tasks/language_modeling)任务指南。

⚡️ 推理

- [TrOCR手写字符识别的交互式演示](https://huggingface.co/spaces/nielsr/TrOCR-handwritten)。

## 推理

TrOCR的[`VisionEncoderDecoder`]模型接受图像作为输入，并利用[`~generation.GenerationMixin.generate`]将输入图像自回归地生成文本。

[`ViTImageProcessor`/`DeiTImageProcessor`]类负责预处理输入图像，[`RobertaTokenizer`/`XLMRobertaTokenizer`]将生成的目标token解码为目标字符串。
[`TrOCRProcessor`]将[`ViTImageProcessor`/`DeiTImageProcessor`]和[`RobertaTokenizer`/`XLMRobertaTokenizer`]封装为单个实例，用于提取输入特征和解码预测的tokenid。

- 逐步光学字符识别（OCR）

``` py
>>> from transformers import TrOCRProcessor, VisionEncoderDecoderModel
>>> import requests
>>> from PIL import Image

>>> processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
>>> model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

>>> # 从IAM数据集加载图像
>>> url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

>>> pixel_values = processor(image, return_tensors="pt").pixel_values
>>> generated_ids = model.generate(pixel_values)

>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

请参阅[模型中心](https://huggingface.co/models?filter=trocr)查找TrOCR检查点。

## TrOCRConfig

[[autodoc]] TrOCRConfig

## TrOCRProcessor

[[autodoc]] TrOCRProcessor
    - __call__
    - from_pretrained
    - save_pretrained
    - batch_decode
    - decode

## TrOCRForCausalLM

[[autodoc]] TrOCRForCausalLM
     - forward
