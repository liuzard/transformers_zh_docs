<!--版权所有2020 The HuggingFace团队。保留所有权利。

根据Apache许可证第2.0版（“许可证”）的规定，在遵守许可证的前提下，
您不得使用此文件。您可以获得许可证的副本。

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”基础上的，
没有明示或暗示的保证或条件。请参阅许可证以获取特定语言的权限和限制。

⚠️ 请注意，此文件是Markdown格式，但包含我们doc-builder的特定语法（类似于MDX），
可能在您的Markdown查看器中无法正确显示。

-->

# 自动类

在许多情况下，您要使用的架构可以从预训练模型的名称或路径中猜测出来，
您将此名称/路径提供给`from_pretrained()`方法即可自动检索出相关的模型、权重、配置文件和词汇表。

实例化[`AutoConfig`]、[`AutoModel`]和
[`AutoTokenizer`]中的任何一个类都将直接创建相应架构的类。例如


```python
model = AutoModel.from_pretrained("bert-base-cased")
```

将创建一个[`BertModel`]的实例作为模型。

每个任务和每个后端（PyTorch、TensorFlow或Flax）都有一个`AutoModel`类。

## 扩展自动类

每个自动类都有一个方法可以用于扩展您的自定义类。例如，如果您定义了一个自定义模型类`NewModel`，
请确保有一个`NewModelConfig`，然后可以像这样将其添加到自动类中：

```python
from transformers import AutoConfig, AutoModel

AutoConfig.register("new-model", NewModelConfig)
AutoModel.register(NewModelConfig, NewModel)
```

然后，您就可以像通常使用自动类一样使用它们！

<Tip warning={true}>

如果`NewModelConfig`是[`~transformer.PretrainedConfig`]的子类，请确保其
`model_type`属性设置为在注册配置文件时使用的相同键（在这里为`"new-model"`）。

同样，如果`NewModel`是[`PreTrainedModel`]的子类，请确保其
`config_class`属性设置为注册模型时使用的相同类（在这里为
`NewModelConfig`）。

</Tip>

## AutoConfig

[[autodoc]] AutoConfig

## AutoTokenizer

[[autodoc]] AutoTokenizer

## AutoFeatureExtractor

[[autodoc]] AutoFeatureExtractor

## AutoImageProcessor

[[autodoc]] AutoImageProcessor

## AutoProcessor

[[autodoc]] AutoProcessor

## 通用模型类

以下自动类可用于实例化无特定头部的基本模型类。

### AutoModel

[[autodoc]] AutoModel

### TFAutoModel

[[autodoc]] TFAutoModel

### FlaxAutoModel

[[autodoc]] FlaxAutoModel

## 通用预训练类

以下自动类可用于实例化带有预训练头部的模型。

### AutoModelForPreTraining

[[autodoc]] AutoModelForPreTraining

### TFAutoModelForPreTraining

[[autodoc]] TFAutoModelForPreTraining

### FlaxAutoModelForPreTraining

[[autodoc]] FlaxAutoModelForPreTraining

## 自然语言处理

以下自动类可用于以下自然语言处理任务。

### AutoModelForCausalLM

[[autodoc]] AutoModelForCausalLM

### TFAutoModelForCausalLM

[[autodoc]] TFAutoModelForCausalLM

### FlaxAutoModelForCausalLM

[[autodoc]] FlaxAutoModelForCausalLM

### AutoModelForMaskedLM

[[autodoc]] AutoModelForMaskedLM

### TFAutoModelForMaskedLM

[[autodoc]] TFAutoModelForMaskedLM

### FlaxAutoModelForMaskedLM

[[autodoc]] FlaxAutoModelForMaskedLM

### AutoModelForMaskGeneration

[[autodoc]] AutoModelForMaskGeneration

### TFAutoModelForMaskGeneration

[[autodoc]] TFAutoModelForMaskGeneration

### AutoModelForSeq2SeqLM

[[autodoc]] AutoModelForSeq2SeqLM

### TFAutoModelForSeq2SeqLM

[[autodoc]] TFAutoModelForSeq2SeqLM

### FlaxAutoModelForSeq2SeqLM

[[autodoc]] FlaxAutoModelForSeq2SeqLM

### AutoModelForSequenceClassification

[[autodoc]] AutoModelForSequenceClassification

### TFAutoModelForSequenceClassification

[[autodoc]] TFAutoModelForSequenceClassification

### FlaxAutoModelForSequenceClassification

[[autodoc]] FlaxAutoModelForSequenceClassification

### AutoModelForMultipleChoice

[[autodoc]] AutoModelForMultipleChoice

### TFAutoModelForMultipleChoice

[[autodoc]] TFAutoModelForMultipleChoice

### FlaxAutoModelForMultipleChoice

[[autodoc]] FlaxAutoModelForMultipleChoice

### AutoModelForNextSentencePrediction

[[autodoc]] AutoModelForNextSentencePrediction

### TFAutoModelForNextSentencePrediction

[[autodoc]] TFAutoModelForNextSentencePrediction

### FlaxAutoModelForNextSentencePrediction

[[autodoc]] FlaxAutoModelForNextSentencePrediction

### AutoModelForTokenClassification

[[autodoc]] AutoModelForTokenClassification

### TFAutoModelForTokenClassification

[[autodoc]] TFAutoModelForTokenClassification

### FlaxAutoModelForTokenClassification

[[autodoc]] FlaxAutoModelForTokenClassification

### AutoModelForQuestionAnswering

[[autodoc]] AutoModelForQuestionAnswering

### TFAutoModelForQuestionAnswering

[[autodoc]] TFAutoModelForQuestionAnswering

### FlaxAutoModelForQuestionAnswering

[[autodoc]] FlaxAutoModelForQuestionAnswering

### AutoModelForTextEncoding

[[autodoc]] AutoModelForTextEncoding

### TFAutoModelForTextEncoding

[[autodoc]] TFAutoModelForTextEncoding

## 计算机视觉

以下自动类可用于以下计算机视觉任务。

### AutoModelForDepthEstimation

[[autodoc]] AutoModelForDepthEstimation

### AutoModelForImageClassification

[[autodoc]] AutoModelForImageClassification

### TFAutoModelForImageClassification

[[autodoc]] TFAutoModelForImageClassification

### FlaxAutoModelForImageClassification

[[autodoc]] FlaxAutoModelForImageClassification

### AutoModelForVideoClassification

[[autodoc]] AutoModelForVideoClassification

### AutoModelForMaskedImageModeling

[[autodoc]] AutoModelForMaskedImageModeling

### TFAutoModelForMaskedImageModeling

[[autodoc]] TFAutoModelForMaskedImageModeling

### AutoModelForObjectDetection

[[autodoc]] AutoModelForObjectDetection

### AutoModelForImageSegmentation

[[autodoc]] AutoModelForImageSegmentation

### AutoModelForSemanticSegmentation

[[autodoc]] AutoModelForSemanticSegmentation

### TFAutoModelForSemanticSegmentation

[[autodoc]] TFAutoModelForSemanticSegmentation

### AutoModelForInstanceSegmentation

[[autodoc]] AutoModelForInstanceSegmentation

### AutoModelForUniversalSegmentation

[[autodoc]] AutoModelForUniversalSegmentation

### AutoModelForZeroShotImageClassification

[[autodoc]] AutoModelForZeroShotImageClassification

### TFAutoModelForZeroShotImageClassification

[[autodoc]] TFAutoModelForZeroShotImageClassification

### AutoModelForZeroShotObjectDetection

[[autodoc]] AutoModelForZeroShotObjectDetection

## 音频

以下自动类可用于以下音频任务。

### AutoModelForAudioClassification

[[autodoc]] AutoModelForAudioClassification

### AutoModelForAudioFrameClassification

[[autodoc]] TFAutoModelForAudioClassification

### TFAutoModelForAudioFrameClassification

[[autodoc]] AutoModelForAudioFrameClassification

### AutoModelForCTC

[[autodoc]] AutoModelForCTC

### AutoModelForSpeechSeq2Seq

[[autodoc]] AutoModelForSpeechSeq2Seq

### TFAutoModelForSpeechSeq2Seq

[[autodoc]] TFAutoModelForSpeechSeq2Seq

### FlaxAutoModelForSpeechSeq2Seq

[[autodoc]] FlaxAutoModelForSpeechSeq2Seq

### AutoModelForAudioXVector

[[autodoc]] AutoModelForAudioXVector

### AutoModelForTextToSpectrogram

[[autodoc]] AutoModelForTextToSpectrogram

### AutoModelForTextToWaveform

[[autodoc]] AutoModelForTextToWaveform

## 多模态

以下自动类可用于以下多模态任务。

### AutoModelForTableQuestionAnswering

[[autodoc]] AutoModelForTableQuestionAnswering

### TFAutoModelForTableQuestionAnswering

[[autodoc]] TFAutoModelForTableQuestionAnswering

### AutoModelForDocumentQuestionAnswering

[[autodoc]] AutoModelForDocumentQuestionAnswering

### TFAutoModelForDocumentQuestionAnswering

[[autodoc]] TFAutoModelForDocumentQuestionAnswering

### AutoModelForVisualQuestionAnswering

[[autodoc]] AutoModelForVisualQuestionAnswering

### AutoModelForVision2Seq

[[autodoc]] AutoModelForVision2Seq

### TFAutoModelForVision2Seq

[[autodoc]] TFAutoModelForVision2Seq

### FlaxAutoModelForVision2Seq

[[autodoc]] FlaxAutoModelForVision2Seq