<!--版权所有2022年HuggingFace团队。保留所有权利。

根据Apache许可证第2.0版（“许可证”），除非符合以下规定，否则不得使用此文件；
您可以在以下的网址上获得许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”提供的，
没有任何明示或暗示的担保或条件。请参阅许可证以了解许可证下的特定语言的权限和限制。

⚠️注意，此文件采用Markdown格式，但包含特定于我们doc-builder（类似于MDX）的语法，可能无法在您的Markdown查看器中正确显示。-->

# Data2Vec

## 概览

Data2Vec模型是由Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu和Michael Auli在[《data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language》](https://arxiv.org/pdf/2202.03555)中提出的。
Data2Vec提供了一个统一的框架，用于跨不同数据形式（文本、音频和图像）的无监督学习。
重要的是，预训练的目标是输入的上下文化的潜在表示，而不是特定于形态的、上下文无关的目标。

论文摘要如下：

*虽然无监督学习的一般思想在各种形态之间是一样的，但实际的算法和目标却有很大的不同，因为它们是根据单一形态开发的。为了让我们更接近通用的无监督学习，我们提出了data2vec，这是一个框架，可以使用相同的学习方法进行语音、自然语言处理（NLP）或计算机视觉学习。核心思想是使用标准Transformer架构，在自蒸馏环境中基于输入的蒙版视图来预测完整输入数据的潜在表示。data2vec不是预测类似于单词、视觉标记或人类语音单元等局部内在目标数据，而是预测包含完整输入信息的上下文化的潜在表示。在语音识别、图像分类和自然语言理解的主要基准测试中的实验证明了与主流方法相比的新的最先进性能。模型和代码可以在www.github.com/pytorch/fairseq/tree/master/examples/data2vec获取。*

提示：

- Data2VecAudio、Data2VecText和Data2VecVision都使用了相同的自监督学习方法训练。
- 对于Data2VecAudio，预处理与[`Wav2Vec2Model`]相同，包括特征提取。
- 对于Data2VecText，预处理与[`RobertaModel`]相同，包括标记化。
- 对于Data2VecVision，预处理与[`BeitModel`]相同，包括特征提取。

此模型由[edugp](https://huggingface.co/edugp)和[patrickvonplaten](https://huggingface.co/patrickvonplaten)贡献。
[sayakpaul](https://github.com/sayakpaul)和[Rocketknight1](https://github.com/Rocketknight1)为TensorFlow的Data2Vec视觉贡献了代码。

NLP和语音的原始代码可以在[此处](https://github.com/pytorch/fairseq/tree/main/examples/data2vec)找到。
视觉的原始代码可以在[此处](https://github.com/facebookresearch/data2vec_vision/tree/main/beit)找到。


## 资源

以下是官方Hugging Face资源和社区资源（由🌎表示），可以帮助您开始使用Data2Vec。

<PipelineTag pipeline="image-classification"/>

- 可通过此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)使用[`Data2VecVisionForImageClassification`]。
- 要在自定义数据集上微调[`TFData2VecVisionForImageClassification`]，请参见[此笔记本](https://colab.research.google.com/github/sayakpaul/TF-2.0-Hacks/blob/master/data2vec_vision_image_classification.ipynb)。

**Data2VecText文档资源**
- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)
- [因果语言建模任务指南](../tasks/language_modeling)
- [遮蔽语言建模任务指南](../tasks/masked_language_modeling)
- [多项选择任务指南](../tasks/multiple_choice)

**Data2VecAudio文档资源**
- [音频分类任务指南](../tasks/audio_classification)
- [自动语音识别任务指南](../tasks/asr)

**Data2VecVision文档资源**
- [图像分类](../tasks/image_classification)
- [语义分割](../tasks/semantic_segmentation)

如果您有兴趣提交资源以包含在此处，请随时提出拉取请求，我们将会进行审核！资源理想情况下应该展示出一些新内容，而不是重复现有资源。

## Data2VecTextConfig

[[autodoc]] Data2VecTextConfig

## Data2VecAudioConfig

[[autodoc]] Data2VecAudioConfig

## Data2VecVisionConfig

[[autodoc]] Data2VecVisionConfig


## Data2VecAudioModel

[[autodoc]] Data2VecAudioModel
    - forward

## Data2VecAudioForAudioFrameClassification

[[autodoc]] Data2VecAudioForAudioFrameClassification
    - forward

## Data2VecAudioForCTC

[[autodoc]] Data2VecAudioForCTC
    - forward

## Data2VecAudioForSequenceClassification

[[autodoc]] Data2VecAudioForSequenceClassification
    - forward

## Data2VecAudioForXVector

[[autodoc]] Data2VecAudioForXVector
    - forward

## Data2VecTextModel

[[autodoc]] Data2VecTextModel
    - forward

## Data2VecTextForCausalLM

[[autodoc]] Data2VecTextForCausalLM
    - forward

## Data2VecTextForMaskedLM

[[autodoc]] Data2VecTextForMaskedLM
    - forward

## Data2VecTextForSequenceClassification

[[autodoc]] Data2VecTextForSequenceClassification
    - forward

## Data2VecTextForMultipleChoice

[[autodoc]] Data2VecTextForMultipleChoice
    - forward

## Data2VecTextForTokenClassification

[[autodoc]] Data2VecTextForTokenClassification
    - forward

## Data2VecTextForQuestionAnswering

[[autodoc]] Data2VecTextForQuestionAnswering
    - forward

## Data2VecVisionModel

[[autodoc]] Data2VecVisionModel
    - forward

## Data2VecVisionForImageClassification

[[autodoc]] Data2VecVisionForImageClassification
    - forward

## Data2VecVisionForSemanticSegmentation

[[autodoc]] Data2VecVisionForSemanticSegmentation
    - forward

## TFData2VecVisionModel

[[autodoc]] TFData2VecVisionModel
    - call

## TFData2VecVisionForImageClassification

[[autodoc]] TFData2VecVisionForImageClassification
    - call

## TFData2VecVisionForSemanticSegmentation

[[autodoc]] TFData2VecVisionForSemanticSegmentation
    - call
