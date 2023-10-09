<!--版权2020 The HuggingFace Team。版权所有。

根据Apache License，版本2.0许可证（“许可证”）；除非符合许可证的规定，
否则您不得使用此文件。您可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律或书面同意，否则根据许可证分布的软件是基于
“按原样”分发，不附带任何明示或暗示的保证或条件。请参阅许可证
以了解许可证下的特定语言的权限和限制。

⚠️注意，此文件是Markdown文件，但包含了特定于我们文档构建器（类似于MDX）的语法，
可能在您的Markdown查看器中无法正确渲染。

-->

# 生成工具

该页面列出了由[`~generation.GenerationMixin.generate`]，
[`~generation.GenerationMixin.greedy_search`]，
[`~generation.GenerationMixin.contrastive_search`]，
[`~generation.GenerationMixin.sample`]，
[`~generation.GenerationMixin.beam_search`]，
[`~generation.GenerationMixin.beam_sample`]，
[`~generation.GenerationMixin.group_beam_search`]和
[`~generation.GenerationMixin.constrained_beam_search`]使用的所有实用函数。

这些函数大多数只在您研究库中的生成方法时才有用。

## 生成输出

[`~generation.GenerationMixin.generate`]的输出是[`~utils.ModelOutput`]的子类的实例。
此输出是一个数据结构，包含所有由[`~generation.GenerationMixin.generate`]返回的信息，
但也可以用作元组或字典。

以下是一个示例：

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

inputs = tokenizer("Hello, my dog is cute and ", return_tensors="pt")
generation_output = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)
```

`generation_output`对象是[`~generation.GreedySearchDecoderOnlyOutput`]，正如我们可以在
下面该类的文档中看到的，它具有以下属性：

- `sequences`：生成的标记序列
- `scores`（可选）：语言建模头的预测分数，每一代的步骤
- `hidden_states`（可选）：模型的隐藏状态，每一代的步骤
- `attentions`（可选）：模型的注意力权重，每一代的步骤

这里我们有`scores`，因为我们传递了`output_scores=True`，但我们没有`hidden_states`和
`attentions`，因为我们没有传递`output_hidden_states=True`或`output_attentions=True`。

您可以像通常那样访问每个属性，如果该属性未被模型返回，您将获得`None`。例如，在这里，
`generation_output.scores`是语言模型头的所有生成预测分数，
`generation_output.attentions`是`None`。

将我们的`generation_output`对象用作元组时，它仅保留没有`None`值的属性。
例如，在这里，它有两个元素，`loss`然后`logits`，所以

```python
generation_output[:2]
```

将返回元组`(generation_output.sequences, generation_output.scores)`。

将我们的`generation_output`对象用作字典时，它仅保留没有`None`值的属性。
例如，在这里，它有两个键，即`sequences`和`scores`。

我们在此处记录了所有输出类型。

### PyTorch

[[autodoc]] generation.GreedySearchEncoderDecoderOutput

[[autodoc]] generation.GreedySearchDecoderOnlyOutput

[[autodoc]] generation.SampleEncoderDecoderOutput

[[autodoc]] generation.SampleDecoderOnlyOutput

[[autodoc]] generation.BeamSearchEncoderDecoderOutput

[[autodoc]] generation.BeamSearchDecoderOnlyOutput

[[autodoc]] generation.BeamSampleEncoderDecoderOutput

[[autodoc]] generation.BeamSampleDecoderOnlyOutput

[[autodoc]] generation.ContrastiveSearchEncoderDecoderOutput

[[autodoc]] generation.ContrastiveSearchDecoderOnlyOutput

### TensorFlow

[[autodoc]] generation.TFGreedySearchEncoderDecoderOutput

[[autodoc]] generation.TFGreedySearchDecoderOnlyOutput

[[autodoc]] generation.TFSampleEncoderDecoderOutput

[[autodoc]] generation.TFSampleDecoderOnlyOutput

[[autodoc]] generation.TFBeamSearchEncoderDecoderOutput

[[autodoc]] generation.TFBeamSearchDecoderOnlyOutput

[[autodoc]] generation.TFBeamSampleEncoderDecoderOutput

[[autodoc]] generation.TFBeamSampleDecoderOnlyOutput

[[autodoc]] generation.TFContrastiveSearchEncoderDecoderOutput

[[autodoc]] generation.TFContrastiveSearchDecoderOnlyOutput

### FLAX

[[autodoc]] generation.FlaxSampleOutput

[[autodoc]] generation.FlaxGreedySearchOutput

[[autodoc]] generation.FlaxBeamSearchOutput

## LogitsProcessor

[`LogitsProcessor`]用于修改生成模型头的预测分数。

### PyTorch

[[autodoc]] AlternatingCodebooksLogitsProcessor
    - __call__

[[autodoc]] ClassifierFreeGuidanceLogitsProcessor
    - __call__

[[autodoc]] EncoderNoRepeatNGramLogitsProcessor
    - __call__

[[autodoc]] EncoderRepetitionPenaltyLogitsProcessor
    - __call__

[[autodoc]] EpsilonLogitsWarper
    - __call__

[[autodoc]] EtaLogitsWarper
    - __call__

[[autodoc]] ExponentialDecayLengthPenalty
    - __call__

[[autodoc]] ForcedBOSTokenLogitsProcessor
    - __call__

[[autodoc]] ForcedEOSTokenLogitsProcessor
    - __call__

[[autodoc]] ForceTokensLogitsProcessor
    - __call__

[[autodoc]] HammingDiversityLogitsProcessor
    - __call__

[[autodoc]] InfNanRemoveLogitsProcessor
    - __call__

[[autodoc]] LogitNormalization
    - __call__

[[autodoc]] LogitsProcessor
    - __call__

[[autodoc]] LogitsProcessorList
    - __call__

[[autodoc]] LogitsWarper
    - __call__

[[autodoc]] MinLengthLogitsProcessor
    - __call__

[[autodoc]] MinNewTokensLengthLogitsProcessor
    - __call__

[[autodoc]] NoBadWordsLogitsProcessor
    - __call__

[[autodoc]] NoRepeatNGramLogitsProcessor
    - __call__

[[autodoc]] PrefixConstrainedLogitsProcessor
    - __call__

[[autodoc]] RepetitionPenaltyLogitsProcessor
    - __call__

[[autodoc]] SequenceBiasLogitsProcessor
    - __call__

[[autodoc]] SuppressTokensAtBeginLogitsProcessor
    - __call__

[[autodoc]] SuppressTokensLogitsProcessor
    - __call__

[[autodoc]] TemperatureLogitsWarper
    - __call__

[[autodoc]] TopKLogitsWarper
    - __call__

[[autodoc]] TopPLogitsWarper
    - __call__

[[autodoc]] TypicalLogitsWarper
    - __call__

[[autodoc]] UnbatchedClassifierFreeGuidanceLogitsProcessor
    - __call__

[[autodoc]] WhisperTimeStampLogitsProcessor
    - __call__

### TensorFlow

[[autodoc]] TFForcedBOSTokenLogitsProcessor
    - __call__

[[autodoc]] TFForcedEOSTokenLogitsProcessor
    - __call__

[[autodoc]] TFForceTokensLogitsProcessor
    - __call__

[[autodoc]] TFLogitsProcessor
    - __call__

[[autodoc]] TFLogitsProcessorList
    - __call__

[[autodoc]] TFLogitsWarper
    - __call__

[[autodoc]] TFMinLengthLogitsProcessor
    - __call__

[[autodoc]] TFNoBadWordsLogitsProcessor
    - __call__

[[autodoc]] TFNoRepeatNGramLogitsProcessor
    - __call__

[[autodoc]] TFRepetitionPenaltyLogitsProcessor
    - __call__

[[autodoc]] TFSuppressTokensAtBeginLogitsProcessor
    - __call__

[[autodoc]] TFSuppressTokensLogitsProcessor
    - __call__

[[autodoc]] TFTemperatureLogitsWarper
    - __call__

[[autodoc]] TFTopKLogitsWarper
    - __call__

[[autodoc]] TFTopPLogitsWarper
    - __call__

### FLAX

[[autodoc]] FlaxForcedBOSTokenLogitsProcessor
    - __call__

[[autodoc]] FlaxForcedEOSTokenLogitsProcessor
    - __call__

[[autodoc]] FlaxForceTokensLogitsProcessor
    - __call__

[[autodoc]] FlaxLogitsProcessor
    - __call__

[[autodoc]] FlaxLogitsProcessorList
    - __call__

[[autodoc]] FlaxLogitsWarper
    - __call__

[[autodoc]] FlaxMinLengthLogitsProcessor
    - __call__

[[autodoc]] FlaxSuppressTokensAtBeginLogitsProcessor
    - __call__

[[autodoc]] FlaxSuppressTokensLogitsProcessor
    - __call__

[[autodoc]] FlaxTemperatureLogitsWarper
    - __call__

[[autodoc]] FlaxTopKLogitsWarper
    - __call__

[[autodoc]] FlaxTopPLogitsWarper
    - __call__

[[autodoc]] FlaxWhisperTimeStampLogitsProcessor
    - __call__

## 停止准则

[`StoppingCriteria`]可以用于更改停止生成的条件（除了EOS标记）。请注意，这仅适用于我们的PyTorch实现。

[[autodoc]] StoppingCriteria
    - __call__

[[autodoc]] StoppingCriteriaList
    - __call__

[[autodoc]] MaxLengthCriteria
    - __call__

[[autodoc]] MaxTimeCriteria
    - __call__

## 约束

[`Constraint`]可以用于强制生成结果中包含特定标记或序列。请注意，这仅适用于我们的PyTorch实现。

[[autodoc]] Constraint

[[autodoc]] PhrasalConstraint

[[autodoc]] DisjunctiveConstraint

[[autodoc]] ConstraintListState

## Beam搜索

[[autodoc]] BeamScorer
    - process
    - finalize

[[autodoc]] BeamSearchScorer
    - process
    - finalize

[[autodoc]] ConstrainedBeamSearchScorer
    - process
    - finalize

## 工具

[[autodoc]] top_k_top_p_filtering

[[autodoc]] tf_top_k_top_p_filtering

## 数据流

[[autodoc]] TextStreamer

[[autodoc]] TextIteratorStreamer
