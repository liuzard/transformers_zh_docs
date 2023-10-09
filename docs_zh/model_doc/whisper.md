<!-- 版权 2022 年的 HuggingFace 团队。保留所有权利。

根据 Apache 许可证第 2.0 版（“许可证”）许可；除非符合许可证，否则您不得使用此文件。
您可以在以下位置获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，软件根据许可证以“按原样”
分发，不附带任何形式的明示或暗示的保证或条件。请参阅许可证以获取
特定语言下的权限和限制。

⚠️请注意，此文件是 Markdown 格式，但包含特定语法以供我们的文档生成器(类似于 MDX)使用，可能无法在您的 Markdown 查看器中正确渲染。-->

# Whisper

## 概览

Whisper 模型是由 Alec Radford、Jong Wook Kim、Tao Xu、Greg Brockman 和 Christine McLeavey 提出的[《Robust Speech Recognition via Large-Scale Weak Supervision》](https://cdn.openai.com/papers/whisper.pdf)中提出的。 
该论文的摘要如下：

*我们研究了仅通过训练来预测因特网上大量音频文本的语音处理系统的能力。当扩展至 68 万小时的多语言和多任务监督时，得到的模型在标准基准测试中具有良好的泛化性能，并且通常与以前的完全监督结果相竞争，但在无需任何微调的零样本迁移设置下。与人类相比，这些模型接近其准确性和鲁棒性。我们发布模型和推断代码作为鲁棒语音处理进一步研究的基础。*


提示：

- 该模型通常在无需任何微调的情况下表现良好。
- 该架构遵循经典的编码器-解码器架构，这意味着它依赖[`~generation.GenerationMixin.generate`]函数进行推理。
- 目前仅实现了短形式的推理，即音频被预分段为<=30s的段落。长形式（包括时间戳）将在将来的版本中实现。
- 可以使用[`WhisperProcessor`]将音频准备好供模型使用，并将预测的 ID 解码回文本。

此模型由[Arthur Zucker](https://huggingface.co/ArthurZ)提供。该模型的 TensorFlow 版本由[amyeroberts](https://huggingface.co/amyeroberts)提供。
原始代码可以在[这里](https://github.com/openai/whisper)找到。


## WhisperConfig

[[autodoc]] WhisperConfig

## WhisperTokenizer

[[autodoc]] WhisperTokenizer
    - set_prefix_tokens
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## WhisperTokenizerFast

[[autodoc]] WhisperTokenizerFast
    - set_prefix_tokens
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## WhisperFeatureExtractor

[[autodoc]] WhisperFeatureExtractor
    - __call__

## WhisperProcessor

[[autodoc]] WhisperProcessor
    - __call__
    - from_pretrained
    - save_pretrained
    - batch_decode
    - decode

## WhisperModel

[[autodoc]] WhisperModel
    - forward
    - _mask_input_features

## WhisperForConditionalGeneration

[[autodoc]] WhisperForConditionalGeneration
    - forward

## WhisperForAudioClassification

[[autodoc]] WhisperForAudioClassification
    - forward


## TFWhisperModel

[[autodoc]] TFWhisperModel
    - call

## TFWhisperForConditionalGeneration

[[autodoc]] TFWhisperForConditionalGeneration
    - call


## FlaxWhisperModel

[[autodoc]] FlaxWhisperModel
    - __call__

## FlaxWhisperForConditionalGeneration

[[autodoc]] FlaxWhisperForConditionalGeneration
    - __call__

## FlaxWhisperForAudioClassification

[[autodoc]] FlaxWhisperForAudioClassification
    - __call__