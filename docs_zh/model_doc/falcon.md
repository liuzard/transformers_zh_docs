<!--
版权所有2023的HuggingFace团队。保留所有权利。

根据Apache许可证第2.0版（“许可证”）许可；除非符合许可证的规定，在符合许可证的规定之外，你不得使用此文件。你可以在以下网址获得许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，依据许可证分发的软件是基于“按原样（AS IS）”的基础分发的，不带任何明示或暗示的担保或条件。有关许可证下的特定语言资格和限制，请参阅许可证。

⚠️注意，此文件是采用Markdown格式的，但包含了我们的文档构建器的特定语法（类似于MDX），在你的Markdown查看器中可能无法正确呈现。

-->

# 猎鹰

## 概述

猎鹰是由[TII](https://www.tii.ae/)构建的仅因果解码器类模型。最大的猎鹰检查点已经在大于1T个文本标记的语料库上进行了训练，特别注重[RefinedWeb](https://arxiv.org/abs/2306.01116)语料库。它们在Apache 2.0许可证下可用。

猎鹰的架构是现代化的，针对推理进行了优化，并支持多查询注意力和`FlashAttention`等高效注意力变体。提供了仅作为因果语言模型进行训练的“base”模型以及经过进一步微调的“instruct”模型。

猎鹰模型（截至2023年）是一些最大且最强大的开源语言模型，并在[OpenLLM排行榜](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)中始终排名靠前。

## 转换自定义检查点

<Tip>

最初，猎鹰模型是作为Hugging Face Hub的自定义代码检查点添加的。然而，猎鹰现在在Transformers库中得到了完全支持。如果你从自定义代码检查点微调了模型，我们建议将你的检查点转换为新的库内格式，这应该显著提高稳定性和性能，特别是对于生成任务，并且无需使用`trust_remote_code=True`！

</Tip>

你可以使用Transformers库中[Falcon模型目录](https://github.com/huggingface/transformers/tree/main/src/transformers/models/falcon)中的`convert_custom_code_checkpoint.py`脚本将自定义代码检查点转换为完整的Transformers检查点。要使用此脚本，只需调用`python convert_custom_code_checkpoint.py --checkpoint_dir my_model`。这将原地转换你的检查点，然后你可以立即从目录中加载它，例如`from_pretrained()`。如果你的模型尚未上传到Hub，请在尝试转换之前进行备份，以防万一！

## FalconConfig

[[autodoc]] FalconConfig
    - all

## FalconModel

[[autodoc]] FalconModel
    - forward

## FalconForCausalLM

[[autodoc]] FalconForCausalLM
    - forward

## FalconForSequenceClassification

[[autodoc]] FalconForSequenceClassification
    - forward

## FalconForTokenClassification

[[autodoc]] FalconForTokenClassification
    - forward

## FalconForQuestionAnswering

[[autodoc]] FalconForQuestionAnswering
    - forward
-->