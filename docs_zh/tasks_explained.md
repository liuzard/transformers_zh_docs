<!--版权2023年HuggingFace团队。版权所有。

根据Apache License Version 2.0（“许可证”）许可；除非符合许可证中的条款，
否则不得使用此文件。您可以在http://www.apache.org/licenses/LICENSE-2.0获取许可证的副本。

除非适用法律要求或书面同意，根据许可证分发的软件均按“按原样”分发，
不附带任何明示或暗示的保证或条件。请参阅许可证以了解特定语言的权限和限制。

⚠️注意，该文件采用Markdown格式，但包含特定语法，用于我们的文档构建器（类似于MDX），
这些语法在您的Markdown查看器中可能无法正确渲染。

-->

# 如何使用🤗 Transformers解决任务

在[🤗 Transformers能做什么](task_summary.md)一节中，您了解了自然语言处理（NLP）、语音和音频、计算机视觉任务及其一些重要应用程序。本页将详细介绍模型如何解决这些任务，并解释底层的工作原理。解决给定任务有许多方法，其中一些模型可能实现了特定的技术，甚至从一个新的角度来处理任务，但对于Transformer模型来说，基本思想是相同的。由于其灵活的架构，大多数模型都是编码器、解码器或编码器-解码器结构的变体。除了Transformer模型之外，我们的库还包含了几个卷积神经网络（CNNs），这些网络在计算机视觉任务中仍然被广泛使用。我们还将解释现代CNN是如何工作的。

为了解释任务是如何解决的，我们将详细介绍模型内部的工作过程，以输出有用的预测结果。

- [Wav2Vec2](model_doc/wav2vec2)：用于音频分类和自动语音识别（ASR）
- [Vision Transformer（ViT）](model_doc/vit)和[ConvNeXT](model_doc/convnext)：用于图像分类
- [DETR](model_doc/detr)：用于目标检测
- [Mask2Former](model_doc/mask2former)：用于图像分割
- [GLPN](model_doc/glpn)：用于深度估计
- [BERT](model_doc/bert)：用于NLP任务，如文本分类、标记分类和问题回答，使用的是编码器
- [GPT2](model_doc/gpt2)：用于如文本生成等NLP任务，使用的是解码器
- [BART](model_doc/bart)：用于NLP任务如摘要和翻译，使用的是编码器-解码器

<Tip>

在继续之前，最好具备对原始Transformer架构的基本了解。了解编码器、解码器和注意力如何工作将有助于您理解不同的Transformer模型如何工作。如果您刚开始或需要复习，请查看我们的[课程](https://huggingface.co/course/chapter1/4?fw=pt)以获取更多信息！

</Tip>

## 语音和音频

[Wav2Vec2](model_doc/wav2vec2)是在未标记的语音数据上进行自我监督预训练并在标记数据上进行音频分类和自动语音识别的微调的模型。

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/wav2vec2_architecture.png"/>
</div>

该模型具有四个主要组件：

1. *特征编码器*接受原始音频波形，将其归一化为零均值和单位方差，并将其转换为每个20ms长度的特征向量序列。

2. 由于波形是连续的，因此无法像将文本序列拆分为单词那样将其分割为单独的单元。因此，将特征向量传递给*量化模块*，该模块旨在学习离散的语音单位。从Codebook（可以将其视为词汇表）中选择最能代表连续音频输入的向量或语音单位，并将其传递给模型。

3. 大约一半的特征向量是随机屏蔽的，并且屏蔽的特征向量被输入到*上下文网络*中，该网络是一个Transformer编码器，还会添加相对位置嵌入。

4. 上下文网络的预训练目标是*对比任务*。模型必须从一组错误的预测中预测屏蔽预测的真实量化语音表示，以鼓励模型找到与上下文向量和量化语音单位（目标标签）最相似的量化语音表示。

现在，可以在音频分类或自动语音识别中对wav2vec2进行预训练，以在您的数据上进行微调！

### 音频分类

为了将预训练模型用于音频分类，可以在基础的Wav2Vec2模型之上添加一个序列分类头。分类头是一个线性层，接受编码器的隐藏状态。隐藏状态表示每个音频帧的学习特征，这些音频帧的长度可能不同。为了创建一个定长的向量，首先对隐藏状态进行池化，然后将其转换为类标签上的logits。通过计算logits和目标之间的交叉熵损失来找到最可能的类别。

准备好尝试音频分类了吗？请查看我们完整的[音频分类指南](tasks/audio_classification)，了解如何微调Wav2Vec2并进行推理！

### 自动语音识别

为了将预训练模型用于自动语音识别，可以在基础的Wav2Vec2模型之上添加一个带有语言建模头部的模型，用于[连接主义时序分类（CTC）](glossary.md#connectionist-temporal-classification-ctc)。语言建模头是一个线性层，接受编码器的隐藏状态并将其转换为logits。每个logit表示一个标记类别（标记数来自任务词汇表）。通过计算logits和目标之间的CTC损失来找到最可能的标记序列，然后将其解码成一个转录结果。

准备好尝试自动语音识别了吗？请查看我们完整的[自动语音识别指南](tasks/asr)了解如何微调Wav2Vec2并进行推理！

## 计算机视觉

处理计算机视觉任务有两种方法：

1. 将图像拆分成一系列补丁，并使用Transformer并行处理它们。
2. 使用现代的卷积神经网络，如[ConvNeXT](model_doc/convnext)，它依赖卷积层但采用了现代网络设计。

<Tip>

第三种方法将Transformer与卷积混合在一起（例如，[卷积视觉Transformer](model_doc/cvt)或[LeViT](model_doc/levit)），不过我们不会讨论这些方法，因为它们只是将我们在这里研究的两种方法组合而成。

</Tip>

虽然ViT和ConvNeXT通常用于图像分类，但对于目标检测、分割和深度估计等其他视觉任务，我们将分别介绍DETR、Mask2Former和GLPN；这些模型更适合这些任务。

### 图像分类

ViT和ConvNeXT都可用于图像分类；主要区别在于ViT使用了注意力机制，而ConvNeXT使用了卷积。

#### Transformer

[ViT](model_doc/vit)完全用纯Transformer架构替换了卷积。如果您熟悉原始Transformer，那么您已经了解ViT的大部分内容。

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/vit_architecture.jpg"/>
</div>

ViT引入的主要改变是图像被输入Transformer的方式：

1. 图像被拆分成非重叠的正方形补丁，每个补丁都被转换为一个向量或*patch embedding*。补丁嵌入由一个卷积2D层生成，该层创建正确的输入维度（对于基础Transformer，每个补丁的768个值）。如果图像是224x224像素，可以将其拆分为196个16x16的图像补丁。就像将文本标记为单词一样，图像被“tokenized”为一系列补丁。

2. 添加一个可学习的嵌入——一个特殊的`[CLS]`标记——其位置在补丁嵌入的开头，就像BERT一样。`[CLS]`标记的最终隐藏状态用作被附加分类头的输入；其他输出被忽略。该标记帮助模型学习如何编码图像的表示。

3. 添加到补丁和可学习嵌入的最后一件事是*位置嵌入*，因为模型不知道如何对图像补丁进行排序。位置嵌入也是可学习的，与补丁嵌入具有相同的大小。最后，所有嵌入都传递给Transformer编码器。

4. 输出，特别是仅包含`[CLS]`标记的输出，传递给多层感知机头（MLP）。ViT的预训练目标仅仅是分类。与其他分类头一样，MLP头将输出转换为类标签上的logits，并计算交叉熵损失以找到最可能的类。

准备好尝试图像分类了吗？请查看我们完整的[图像分类指南](tasks/image_classification)了解如何微调ViT并进行推理！

#### CNN

<Tip>

本节简要介绍了卷积，但最好事先了解它们如何改变图像的形状和大小。如果您对卷积不熟悉，请查看fastai书中的[卷积神经网络章节](https://github.com/fastai/fastbook/blob/master/13_convolutions.ipynb)！

</Tip>

[ConvNeXT](model_doc/convnext)是一种卷积神经网络架构，采用新颖的、现代的网络设计以提高性能。然而，卷积仍然是该模型的核心。从高层次的角度来看，[卷积](glossary.md#convolution)是一种操作，其中较小的矩阵（*卷积核*）与图像像素的一个小窗口相乘。它计算出一些特征，例如特定纹理或线条的曲率。然后它滑动到下一个像素窗口；卷积移动的距离被称为*步幅*。

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/convolution.gif"/>
</div>

<small><a href="https://arxiv.org/abs/1603.07285">深度学习的卷积算术指南</a>中的一个基本卷积示例，不带填充和步幅。</small>

您可以将这个输出传递给另一个卷积层，每个连续的层网络都能学习到更复杂和抽象的东西，比如热狗或火箭。在卷积层之间，通常在卷积层之间添加一个池化层，以降低维度，并使模型对特征位置的变化更加鲁棒。

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/convnext_architecture.png"/>
</div>

ConvNeXT以五种方式现代化了CNN：

1. 在每个阶段中更改块的数量，并使用更大的步幅和相应的核大小来对图像进行“patchify”。非重叠滑动窗口使得这种“patchifying”策略类似于ViT将图像拆分为补丁的方式。

2. *瓶颈*层减小了通道数，然后恢复它，因为做一个1x1的卷积更快，可以增加深度。反向瓶颈则相反，通过扩展通道数然后将其收缩，这样更节省内存。

3. 用*深度卷积替换瓶颈层中典型的3x3卷积层*，深度卷积对每个输入通道分别应用卷积，然后在最后将它们堆叠在一起。这会增加网络宽度，以提高性能。

4. ViT具有全局感受野，这意味着它可以看到更多的图像，这得益于其注意力机制。ConvNeXT通过将核大小增加到7x7来复制该效果。

5. ConvNeXT还对几个层设计进行了一些更改，以模仿Transformer模型。激活和归一化层更少，激活函数从ReLU切换到GELU，使用LayerNorm而不是BatchNorm。

从卷积块的输出结果通过一个分类头，将输出转换为logits，并计算交叉熵损失来找到最可能的标签。

### 目标检测

[DETR](model_doc/detr)（即*DEtection TRansformer*）是一种端到端的目标检测模型，它将一个CNN与一个Transformer编码器-解码器结合起来。

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/detr_architecture.png"/>
</div>

1. 预训练的CNN *后骨干网络*接受由像素值表示的图像，并创建一个低分辨率的特征图。然后，将1x1卷积应用于特征图以降低维度，并创建具有高级图像表示的新特征图。由于Transformer是一个串行模型，将特征图展平为一系列特征向量，并将其与位置嵌入相结合。

2. 特征向量经过编码器，使用其注意力层学习图像表示。然后，编码器的隐藏状态与解码器中的*目标查询*相结合。目标查询是学习的嵌入，用于关注图像的不同区域，并在每个注意力层中按顺序更新。解码器的隐藏状态被传递给一个前馈网络，该网络预测了每个目标查询的边界框坐标和类标签，如果没有目标，则直接预测`no object`。

    DETR并行解码每个目标查询，输出*N*个最终预测结果，其中*N*是查询的数量。与典型的自回归模型一次预测一个元素不同，目标检测是一个集合预测任务（`边界框`，`类标签`），可以在单次传递中进行*N*次预测。

3. DETR在训练过程中使用*双向匹配损失*来将一组固定数量的预测与一组固定的真实标签进行比较。如果在*N*个标签集中有较少的真实标签，则它们将使用`no object`类进行填充。此损失函数鼓励DETR在预测和真实标签之间找到一对一的匹配。如果边界框或类标签不正确，则会产生损失。同样，如果DETR预测了不存在的对象，它也会受到惩罚。这鼓励DETR在图像中找到其他对象，而不是只关注一个非常明显的对象。

在DETR之上添加一个目标检测头部，以找到类标签和边界框的坐标。

目标检测头部由两个组件组成：一个线性层，将解码器的隐藏状态转换为类标签上的logits；一个多层感知机，用于预测边界框。

准备好尝试目标检测了吗？查看我们的完整[目标检测指南](tasks/object_detection)，了解如何微调DETR并将其用于推理！

### 图像分割

[Mask2Former](model_doc/mask2former)是一个通用的图像分割任务架构。传统的分割模型通常针对图像分割的特定子任务进行调整，如实例、语义或全景分割。Mask2Former将每个分割任务都视为一个*掩码分类*问题。掩码分类将像素分组为*N*段，并为给定图像预测*N*个掩码及其相应的类别标签。本节中，我们将解释Mask2Former的工作原理，然后您可以尝试微调SegFormer。

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/mask2former_architecture.png"/>
</div>

Mask2Former有三个主要组成部分：

1. [Swin](model_doc/swin)骨干网络接受图像并通过3个连续的3x3卷积生成低分辨率图像特征图。

2. 特征图传递给逐渐将低分辨率特征上采样为高分辨率每像素嵌入的*像素解码器*。像素解码器实际上会生成多尺度特征（包含低分辨率和高分辨率特征），分辨率为原始图像的1/32、1/16和1/8。

3. 这些不同尺度的特征图依次传递给一层接一层的Transformer解码器，以捕捉高分辨率特征中的小对象。Mask2Former的关键在于解码器中的*掩码注意力*机制。与可以关注整个图像的交叉注意力不同，掩码注意力只关注图像的某个区域。这样做更快，并且由于模型足够从图像的局部特征中学习，因此能够获得更好的性能。

4. 像[DETR](tasks_explained.md#object-detection)一样，Mask2Former还使用了学习的对象查询，并将其与来自像素解码器的图像特征组合起来进行一组预测（`class label`，`mask prediction`）。解码器的隐藏状态传递到线性层中，并转换为类别标签上的logits。计算logits和类别标签之间的交叉熵损失，找到最可能的结果。

    掩码预测通过将像素嵌入与最终的解码器隐藏状态相结合生成。通过计算logits和真实掩码之间的sigmoid交叉熵损失和dice损失，找到最可能的掩码。

准备好尝试目标检测了吗？查看我们的完整[图像分割指南](tasks/semantic_segmentation)，了解如何微调SegFormer并将其用于推理！

### 深度估计

[GLPN](model_doc/glpn)（全局-局部路径网络）是用于深度估计的一种Transformer模型，它将[SegFormer](model_doc/segformer)编码器与轻量级解码器结合在一起。

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/glpn_architecture.jpg"/>
</div>

1. 类似于ViT，图像被分割成一个个补丁序列，只是这些图像补丁较小。对于像分割或深度估计这样的密集预测任务，这样做更好。图像补丁被转换为补丁嵌入（有关如何创建补丁嵌入的更多详细信息，请参见[图像分类](#image-classification)部分），并将其馈送到编码器。

2. 编码器接受补丁嵌入，并将它们传递到多个编码器块中。每个块包括注意力和Mix-FFN层。后者的目的是提供位置信息。在每个编码器块的末尾，有一个*补丁合并*层用于创建分层表示。相邻补丁组的特征被串联在一起，并应用线性层将串联特征的数量减少到1/4的分辨率。这将成为下一个编码器块的输入，在该过程中再次重复此过程，直到获得分辨率为1/8、1/16和1/32的图像特征。

3. 轻量级解码器将编码器的最后特征图（1/32的比例）上采样到1/16的比例。从这里开始，将特征传递到一个*选择性特征融合（SFF）*模块中，该模块从每个特征的注意力图中选择和合并局部和全局特征，然后将其上采样到1/8th。这个过程重复，直到解码的特征大小与原始图像相同。输出通过两个卷积层，然后应用sigmoid激活，以预测每个像素的深度。

## 自然语言处理

Transformer最初是为机器翻译而设计的，此后，它几乎成为解决所有自然语言处理任务的默认架构。某些任务适合Transformer的编码器结构，而其他任务更适合解码器。还有一些任务使用了Transformer的编码器-解码器结构。

### 文本分类

[BERT](model_doc/bert)是一个仅编码器模型，是第一个有效实现深度双向性以通过依赖两侧的单词来学习更丰富的文本表示的模型。

1. BERT使用[WordPiece](tokenizer_summary.md#wordpiece)分词将文本生成一个文本的标记嵌入。为了区分一个单独的句子和一对句子，特殊的`[SEP]`标记添加以区分它们。特殊的`[CLS]`标记添加到每个文本序列的开头。用`[CLS]`标记的最终输出作为分类任务的输入。BERT还添加了一个分段嵌入，用于标识一个标记是否属于一对句子中的第一个或第二个句子。

2. BERT使用两个目标进行预训练：遮盖语言建模和下一句预测。在遮盖语言建模中，输入标记中的某些百分比随机被遮盖，模型需要预测这些标记。这解决了双向性的问题，其中模型可以作弊并看到所有的单词，并且“预测”下一个单词。预测的掩码标记的最终隐藏状态通过一个带有softmax层的前馈网络，以预测被遮盖的单词。

    第二个预训练目标是下一句预测。模型必须预测句子B是否跟随在句子A的后面。一半的时间句子B是下一个句子，另一半时间，句子B是一个随机的句子。预测（是否为下一句）通过一个带有softmax层的前馈网络，输出为两个类（`IsNext`和`NotNext`）。

3. 输入嵌入通过多个编码器层输出一些最终隐藏状态。

要将预训练模型用于文本分类，将一个序列分类头添加到基本的BERT模型上。序列分类头是一个线性层，接受最终的隐藏状态并进行线性变换，将其转换为logits。计算logits和目标之间的交叉熵损失，找到最可能的标签。

准备好尝试您的文本分类了吗？查看我们的完整[文本分类指南](tasks/sequence_classification)，了解如何微调DistilBERT并将其用于推理！

### 标记分类

要在诸如命名实体识别（NER）之类的标记分类任务中使用BERT，可以在基本的BERT模型之上添加一个标记分类头。标记分类头是一个线性层，接受最终的隐藏状态并进行线性变换，将其转换为logits。计算logits和每个标记之间的交叉熵损失，找到最可能的标签。

准备好尝试您的标记分类了吗？查看我们的完整[标记分类指南](tasks/token_classification)，了解如何微调DistilBERT并将其用于推理！

### 问答

要在问题回答中使用BERT，可以在基本的BERT模型上添加一个跨度分类头。这个线性层接受最终隐藏状态并进行线性变换，计算与答案相对应的`span`的`start`和`end`的logits。计算logits和标签位置之间的交叉熵损失，找到与答案对应的最可能文本跨度。

准备好尝试您的问题回答了吗？查看我们的完整[问题回答指南](tasks/question_answering)，了解如何微调DistilBERT并将其用于推理！

<Tip>

💡注意一旦预训练了BERT，使用它解决不同任务非常容易。您只需要在预训练模型之上添加一个特定类型的头，将隐藏状态转换为所需的输出！

</Tip>

### 文本生成

[GPT-2](model_doc/gpt2)是一个仅解码器模型，使用大量文本进行预训练。它可以在给定提示的情况下生成令人信服（虽然不总是正确！）的文本，并且尽管没有明确训练过，也可以完成其他NLP任务，如问题回答。

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/gpt2_architecture.png"/>
</div>

1. GPT-2使用[字节对编码（BPE）](tokenizer_summary.md#bytepair-encoding-bpe)对单词进行分词，并生成一个标记嵌入。位置编码被添加到标记嵌入中，以指示序列中每个标记的位置。输入嵌入通过多个解码器块，生成一些最终的隐藏状态。在每个解码器块中，GPT-2使用*遮盖的自注意力*层，这意味着GPT-2不能关注未来的标记，它只能关注左边的标记。这与BERT的[`mask`]标记不同，因为在遮盖的自注意力中，会使用注意力掩码将分数设置为`0`以阻止未来的标记。

2. 从解码器的输出传递到语言建模头部，该头部将隐藏状态进行线性变换，将其转换为logits。标签是序列中的下一个标记，这些标签是通过将logits向右移动一个位置来创建的。计算位移后的logits和标签之间的交叉熵损失，输出下一个最可能的标记。

GPT-2的预训练目标完全基于[因果语言模型](glossary.md#causal-language-modeling)，即预测序列中的下一个词。这使得GPT-2在涉及生成文本的任务上特别有效。

准备好尝试您的文本生成了吗？查看我们的完整[因果语言模型指南](tasks/language_modeling#causal-language-modeling)，了解如何微调DistilGPT-2并将其用于推理！

<Tip>

有关文本生成的更多信息，请查看[文本生成策略](generation_strategies.md)指南！

</Tip>

### 摘要

像[BART](model_doc/bart)和[T5](model_doc/t5)这样的编码器-解码器模型专为摘要任务的序列到序列模式而设计。我们在本节中将解释BART的工作原理，然后您可以尝试微调T5。

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bart_architecture.png"/>
</div>

1. BART的编码器架构与BERT非常相似，接受文本的标记和位置嵌入。BART通过破坏输入然后使用解码器重构来进行预训练。与具有特定破坏策略的其他编码器不同，BART可以应用任何类型的破坏。不过，*文本插入*破坏策略效果最好。在文本插入中，一些文本片段用一个**单个**[`mask`]标记替换。这一点非常重要，因为模型必须预测缺少的标记数量。输入嵌入和遮盖的片段通过编码器传递以输出一些最终的隐藏状态，但是与BERT不同，BART不会在最后添加一个最终的前馈网络来预测一个词。

2. 编码器的输出传递到解码器，解码器必须同时预测编码器输出中的遮盖标记和任何未破坏的标记。这样可以提供额外的上下文，以帮助解码器恢复原始文本。从解码器输出的结果传递到语言建模头部，该头部对隐藏状态进行线性变换，将其转换为logits。计算logits和标签之间的交叉熵损失，其中标签只是通过将logits向右移动一个位置。

准备好尝试您的摘要了吗？查看我们的完整[摘要指南](tasks/summarization)，了解如何微调T5并将其用于推理！

<Tip>

有关文本生成的更多信息，请查看[文本生成策略](generation_strategies.md)指南！

</Tip>

### 翻译

翻译是另一个序列到序列任务的例子，这意味着您可以使用像[BART](model_doc/bart)或[T5](model_doc/t5)这样的编码器-解码器模型来进行翻译。我们在本节中将解释BART的工作原理，然后您可以尝试微调T5。

BART通过添加单独的随机初始化编码器将一个源语言映射到可以解码为目标语言的输入。这个新的编码器的嵌入被传递到预训练的编码器而不是原始的词嵌入。源编码器通过使用模型输出的交叉熵损失进行源编码器、位置编码和输入嵌入的更新来进行训练。在第一步中，模型参数被冻结，第二步中，所有模型参数一起进行训练。

随后，BART推出了一种用于翻译和在许多不同语言上进行预训练的多语言版本，mBART。

准备好尝试您的翻译了吗？查看我们的完整[翻译指南](tasks/summarization)，了解如何微调T5并将其用于推理！

<Tip>

有关文本生成的更多信息，请查看[文本生成策略](generation_strategies.md)指南！

</Tip>