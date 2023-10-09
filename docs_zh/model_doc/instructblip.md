<!--版权所有2023年The HuggingFace团队。保留所有权利。


根据Apache许可证第2.0版（“许可证”），你不得使用此文件，除非符合许可证的规定。你可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0


除非适用法律要求或书面同意，根据许可证分发的软件均以“原样”分发，不附带任何种类的明示或暗示的担保。详情请参阅许可证，以了解许可证下的特定语言的权限和限制。-->

# InstructBLIP

## 概览

InstructBLIP模型是由Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, Steven Hoi提出的[“InstructBLIP：具备指导调整功能的通用视觉语言模型”](https://arxiv.org/abs/2305.06500)。InstructBLIP利用[BLIP-2](blip2)架构进行视觉指导调优。

论文摘要如下：

*预训练和指导调优测试驱动下，已经出现了可以解决不同语言领域任务的通用语言模型。然而，构建通用视觉语言模型由于额外的视觉输入引入了增加的任务差异而具有挑战性。尽管视觉语言预训练已经广泛研究，但视觉语言指导调优相对较少探讨。本文对基于预训练的BLIP-2模型的视觉语言指导调优进行了系统全面的研究。我们收集了26个公开可用的不同类型数据集，将其转换为指导调优格式，并将其分为两个簇进行保留指导调优和保留零-shot评估。此外，我们引入了指导意识的视觉特征提取，这是一种关键方法，使模型能够提取适应给定指导的信息特征。得到的InstructBLIP模型在所有13个保留数据集上实现了最先进的零-shot性能，大幅超越了BLIP-2和更大的Flamingo。当对各个下游任务进行微调（例如科学问答任务IMG的90.7%准确率）时，我们的模型也达到了最先进的性能。此外，我们在定性上证明了InstructBLIP相对于其他多模态模型的优势。*

提示：

- InstructBLIP使用与[BLIP-2](blip2)相同的架构，但有一个微小但重要的差别：它还将文本提示（指导）提供给Q-Former。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/instructblip_architecture.jpg"
alt="drawing" width="600"/>

<small> InstructBLIP架构。来自<a href="https://arxiv.org/abs/2305.06500">原始论文。</a> </small>

该模型由[nielsr](https://huggingface.co/nielsr)贡献。
原始代码可在[此处](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip)找到。


## InstructBlipConfig

[[autodoc]] InstructBlipConfig
    - from_vision_qformer_text_configs

## InstructBlipVisionConfig

[[autodoc]] InstructBlipVisionConfig

## InstructBlipQFormerConfig

[[autodoc]] InstructBlipQFormerConfig

## InstructBlipProcessor

[[autodoc]] InstructBlipProcessor

## InstructBlipVisionModel

[[autodoc]] InstructBlipVisionModel
    - forward

## InstructBlipQFormerModel

[[autodoc]] InstructBlipQFormerModel
    - forward

## InstructBlipForConditionalGeneration

[[autodoc]] InstructBlipForConditionalGeneration
    - forward
    - generate