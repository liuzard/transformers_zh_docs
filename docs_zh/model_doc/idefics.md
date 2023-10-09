<!--版权2023年The HuggingFace团队。版权所有。

根据Apache许可证第2版（“许可证”）获得许可，您除非符合许可证的约定，否则不得使用此文件。您可以在

http://www.apache.org/licenses/LICENSE-2.0

获取许可证副本。

除非适用法律要求或书面同意，以“现状”分发的软件根据许可证分发，不附带任何形式的保证或条件，无论是明示的还是暗示的。有关许可证的特定语言，请参阅许可证。

⚠️请注意，此文件以Markdown格式，但包含我们的文档生成器（类似于MDX）的特定语法，可能无法在Markdown查看器中正确呈现。-->

# IDEFICS

## 概览

IDEFICS模型由[Hugo Laurençon，Lucile Saulnier，Léo Tronchon，Stas Bekman，Amanpreet Singh，Anton Lozhkov，Thomas Wang，Siddharth Karamcheti，Alexander M. Rush，Douwe Kiela，Matthieu Cord，Victor Sanh](https://huggingface.co/papers/2306.16527)在[OBELICS: An Open Web-Scale Filtered Dataset of Interleaved Image-Text Documents](https://huggingface.co/papers/2306.16527)中提出。

论文摘要如下：

*在自然文件中训练的大型多模态模型（交织图像和文本），在需要对一个或多个图像进行推理以生成文本的各种多模态基准测试中，优于在图像-文本对上训练的模型。然而，用于训练这些模型的数据集尚未发布，并且采集过程尚未完全指定。我们介绍由聚合爬取生成的用于训练GB领域语言模型的关联图像和文本对含有14.1亿网页，3.53亿图像和1150亿文本令牌的大规模过滤数据集OBELICS。我们描述了数据集的创建过程，介绍了全面的过滤规则，并对数据集的内容进行了分析。为了展示OBELISC的可行性，我们在数据集上训练一个具有800亿个参数的视觉和语言模型，并在各种多模态基准测试中获得了竞争性能。我们发布了生成数据集的代码以及数据集本身。*

该模型由[HuggingFaceM4](https://huggingface.co/HuggingFaceM4)提供。原始代码可在[此处](<INSERT LINK TO GITHUB REPO HERE>)找到。 (TODO: 暂时尚无公开链接。)


<Tip warning={true}>

Transformers中的Idefics建模代码用于微调和推断预训练的Idefics模型。

要从头开始训练新的Idefics模型，请使用m4代码库（一旦公开链接将提供）

</Tip>


## IdeficsConfig

[[autodoc]] IdeficsConfig

## IdeficsModel

[[autodoc]] IdeficsModel
    - forward

## IdeficsForVisionText2Text

[[autodoc]] IdeficsForVisionText2Text
    - forward

## IdeficsImageProcessor

[[autodoc]] IdeficsImageProcessor
    - preprocess

## IdeficsProcessor

[[autodoc]] IdeficsProcessor
    - __call__