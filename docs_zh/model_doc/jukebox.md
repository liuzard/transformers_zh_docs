<!--版权所有 2022年HuggingFace团队。保留所有权利。

根据Apache许可证第2版（“许可证”），你除非符合许可证的规定，否则不得使用此文件。你可以在以下链接获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件以“按原样”分发，不附带任何保证或条件，无论是明示还是暗示。请参阅许可证以获取许可的特定语言和限制。

⚠️ 请注意，这个文件是Markdown格式的，但包含我们的文档构建器（类似于MDX）的特定语法，这可能在你的Markdown查看器中无法正确渲染。

-->
# Jukebox

## 概述

Jukebox模型的提出来源于[Prafulla Dhariwal, Heewoo Jun, Christine Payne, Jong Wook Kim, Alec Radford, Ilya Sutskever撰写的《Jukebox: A generative model for music》](https://arxiv.org/pdf/2005.00341.pdf)。该模型引入了一种生成音乐的生成模型，可以产生基于艺术家、流派和歌词的一分钟长的音乐样本。

论文摘要如下：

*我们介绍了Jukebox模型，它可以在原始音频领域生成带有唱歌的音乐。我们使用多尺度VQ-VAE来压缩原始音频的长上下文，并使用自回归Transformer模型对其进行建模。我们展示了这种规模的综合模型可以生成高保真度且多样化的音乐，连贯性可达多分钟。我们可以根据艺术家和流派进行条件设置，以调节音乐和声音风格，并根据不对齐的歌词使唱歌更加可控。我们发布了数千个非抽选样本，以及模型权重和代码。*

如下图所示，Jukebox由3个仅解码器模型（`prior`）组成。它们遵循[Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509)中描述的架构，但对其进行了修改以支持更长的上下文长度。首先，使用自编码器对文本歌词进行编码。接下来，第一个（也称为`top_prior`）prior关注从歌词编码器中提取的最后隐藏状态。prior与之前的每个prior通过`AudioConditioner`模块链接。`AudioConditioner`将之前prior的输出上采样到每秒的原始音频帧分辨率。元数据（如*艺术家、流派和时序*）以起始token和定位嵌入形式传递给每个prior。隐藏状态被映射到最接近的VQVAE码本向量，以将它们转换为原始音频。

![JukeboxModel](https://gist.githubusercontent.com/ArthurZucker/92c1acaae62ebf1b6a951710bdd8b6af/raw/c9c517bf4eff61393f6c7dec9366ef02bdd059a3/jukebox.svg)

提示：
- 此模型仅支持推断。这是由于训练过程中需要大量内存。请随时发布PR并添加缺失的内容，以实现与Hugging Face traineer的完全集成！
- 此模型非常缓慢，使用5b top prior在V100 GPU上生成一分钟长的音频需要8小时。为了自动处理模型应在其上执行的设备，请使用`accelerate`。
- 与论文不同，prior的顺序从`0`到`1`，因为这样更直观：我们从`0`开始进行采样。
- 基于原始音频的primed采样（将采样与原始音频条件连接）比ancestral采样需要更多内存，应将`fp16`设置为`True`。

此模型由[Arthur Zucker](https://huggingface.co/ArthurZ)贡献。
原始代码可以在[这里](https://github.com/openai/jukebox)找到。

## JukeboxConfig

[[autodoc]] JukeboxConfig

## JukeboxPriorConfig

[[autodoc]] JukeboxPriorConfig

## JukeboxVQVAEConfig

[[autodoc]] JukeboxVQVAEConfig

## JukeboxTokenizer

[[autodoc]] JukeboxTokenizer
    - save_vocabulary

## JukeboxModel

[[autodoc]] JukeboxModel
    - ancestral_sample
    - primed_sample
    - continue_sample
    - upsample
    - _sample


## JukeboxPrior

[[autodoc]] JukeboxPrior
    - sample
    - forward


## JukeboxVQVAE

[[autodoc]] JukeboxVQVAE
    - forward
    - encode
    - decode