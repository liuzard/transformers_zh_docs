<!--版权2022️ HuggingFace团队。版权所有。

根据Apache许可证，版本2.0（“许可证”），您不得使用此文件，除非符合许可证的要求。
您可以在以下位置获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，本许可证下分发的软件是基于“按原样”发布的，
没有任何形式的明示或暗示担保条件。有关更多详细信息，请参阅许可证。

⚠️请注意，该文件是Markdown文件，但包含特定于我们的doc-builder语法（类似于MDX），可能无法在Markdown查看器中正确显示。

-->

# BLOOM

## 概览

BLOOM模型是通过[BigScience Workshop](https://bigscience.huggingface.co/)提出的，经过不同版本的改进。BigScience受到其他开放科学倡议的启发，研究人员共同投入时间和资源，实现更高的影响力。
BLOOM的架构基本类似于GPT3（下一个token预测的自回归模型），但在46种不同语言和13种编程语言上进行了训练。
同一数据集上训练了多个较小版本的模型。BLOOM有以下版本可用：

- [bloom-560m](https://huggingface.co/bigscience/bloom-560m)
- [bloom-1b1](https://huggingface.co/bigscience/bloom-1b1)
- [bloom-1b7](https://huggingface.co/bigscience/bloom-1b7)
- [bloom-3b](https://huggingface.co/bigscience/bloom-3b)
- [bloom-7b1](https://huggingface.co/bigscience/bloom-7b1)
- [bloom](https://huggingface.co/bigscience/bloom)（176B参数）

## 资源

以下是官方Hugging Face和社区（用🌎表示）提供的资源，可帮助您开始使用BLOOM。如果您有兴趣提交资源以包含在此处，请随时发起拉取请求，我们将进行审查！资源最好能够展示一些新的东西，而不是重复现有的资源。

<PipelineTag pipeline="text-generation"/>

- [`BloomForCausalLM`]的支持可参考[因果语言建模示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#gpt-2gpt-and-causal-language-modeling)和[notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb)。

另请参阅：
- [因果语言建模任务指南](../tasks/language_modeling)
- [文本分类任务指南](../tasks/sequence_classification)
- [令牌分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)


⚡️ 推理
- 有关[优化故事：BLOOM推理](https://huggingface.co/blog/bloom-inference-optimization)的博客。
- 有关[通过DeepSpeed和Accelerate实现极快的BLOOM推理](https://huggingface.co/blog/bloom-inference-pytorch-scripts)的博客。

⚙️ 训练
- 有关[BLOOM训练背后的技术](https://huggingface.co/blog/bloom-megatron-deepspeed)的博客。

## BloomConfig

[[autodoc]] BloomConfig
    - all

## BloomModel

[[autodoc]] BloomModel
    - forward

## BloomTokenizerFast

[[autodoc]] BloomTokenizerFast
    - all

## BloomForCausalLM

[[autodoc]] BloomForCausalLM
    - forward

## BloomForSequenceClassification

[[autodoc]] BloomForSequenceClassification
    - forward

## BloomForTokenClassification

[[autodoc]] BloomForTokenClassification
    - forward

## BloomForQuestionAnswering

[[autodoc]] BloomForQuestionAnswering
    - forward

## FlaxBloomModel

[[autodoc]] FlaxBloomModel
    - __call__

## FlaxBloomForCausalLM

[[autodoc]] FlaxBloomForCausalLM
    - __call__