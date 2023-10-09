<!--版权所有2020的拥抱面团团队。保留所有权利。

根据Apache许可证，版本2.0（“许可证”）的规定，除非符合许可证的规定，否则您不得使用此文件。

您可以在以下位置获取许可证的副本:

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件基础上提供，不提供任何担保或条件。有关许可下特定语言的具体语言，请参见许可协议的相关部分。

⚠️请注意，此文件采用Markdown格式，但包含我们doc-builder的特定语法(类似于MDX)，可能在您的Markdown查看器中无法正确呈现。

-->

# Transformer模型系列

自2017年首次推出[原始Transformer](https://arxiv.org/abs/1706.03762)模型以来，它已经激发了许多新颖的模型，并超越了自然语言处理（NLP）任务。有模型用于[预测蛋白质的折叠结构](https://huggingface.co/blog/deep-learning-with-proteins)，[训练一只猎豹奔跑](https://huggingface.co/blog/train-decision-transformers)和[时间序列预测](https://huggingface.co/blog/time-series-transformers)。由于有这么多Transformer的变种可用，很容易忽视整体情况。所有这些模型的共同之处在于它们都基于原始的Transformer架构。某些模型仅使用编码器或解码器，而其他模型则同时使用两者。这为对Transformer系列模型进行分类和检查其高级差异提供了有用的分类法，并且它将帮助您理解以前未遇到的Transformer模型。

如果您对原始Transformer模型不熟悉或需要复习，请查看Hugging Face课程中的[Transformer是如何工作的](https://huggingface.co/course/chapter1/4?fw=pt)章节。

<div align="center">
    <iframe width="560" height="315" src="https://www.youtube.com/embed/H39Z_720T5s" title="YouTube视频播放器"
    frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope;
    picture-in-picture" allowfullscreen></iframe>
</div>

## 计算机视觉

<iframe style="border: 1px solid rgba(0, 0, 0, 0.1);" width="1000" height="450" src="https://www.figma.com/embed?embed_host=share&url=https%3A%2F%2Fwww.figma.com%2Ffile%2FacQBpeFBVvrDUlzFlkejoz%2FModelscape-timeline%3Fnode-id%3D0%253A1%26t%3Dm0zJ7m2BQ9oe0WtO-1" allowfullscreen></iframe> 

### 卷积网络

长期以来，卷积网络（CNN）一直是计算机视觉任务中的主导范式，直到[视觉Transformer](https://arxiv.org/abs/2010.11929)展示了其可扩展性和效率。即使如此，某些CNN的优点（例如平移不变性）在某些任务中非常强大，因此一些Transformer在其架构中加入了卷积。[ConvNeXt](model_doc/convnext)在这个交换中发生了变化，并从Transformer中选择设计选择以使CNN现代化。例如，ConvNeXt使用非重叠滑动窗口对图像进行分块，并使用更大的内核增加全局感受野。 ConvNeXt还进行了一些层设计选择，以提高内存效率和性能，因此与Transformer相比，它具有竞争优势！

### 编码器[[cv-encoder]]

[视觉Transformer（ViT）](model_doc/vit)打开了无需卷积的计算机视觉任务的大门。ViT使用标准的Transformer编码器，但其主要突破在于其如何处理图像。它将图像分割为固定大小的块，并将它们用于创建嵌入，就像将句子分割为标记一样。通过利用Transformer的高效架构，ViT展示出与当时的CNN相竞争的结果，同时需要更少的资源进行训练。ViT很快受到其他视觉模型的追随，这些模型也可以处理密集的视觉任务，例如分割和检测。

其中一个模型是[Swin](model_doc/swin) Transformer。它从较小的块中构建分层特征图（类似于CNN 👀，不同于ViT），并在更深层中与相邻块合并。只在局部窗口内计算注意力，并在注意力层之间移动窗口以创建连接，以帮助模型学习更好。由于Swin Transformer可以产生分层特征图，因此它是密集预测任务（例如分割和检测）的良好候选模型。 [SegFormer](model_doc/segformer)也使用Transformer编码器构建分层特征图，但它在顶部添加了一个简单的多层感知机（MLP）解码器，以组合所有特征图并进行预测。

其他一些计算机视觉模型，例如BeIT和ViTMAE，从BERT的无监督预训练目标中汲取灵感。 [BeIT](model_doc/beit)通过*模糊图像建模(MIM)*进行预训练；图像块被随机遮挡，且图像同时被标记为视觉标记。 BeIT在预测与遮挡块对应的视觉标记方面进行训练。 [ViTMAE](model_doc/vitmae)具有类似的预训练目标，只是它必须预测像素而不是视觉标记。不寻常的是，图像的75%被遮挡！解码器从被遮挡的标记和编码的块中重新构建像素。经过预训练后，解码器被舍弃，编码器准备用于下游任务。

### 解码器[[cv-decoder]]

仅具有解码器的视觉模型很少见，因为大多数视觉模型依赖于编码器来学习图像表示。但对于图像生成等用例，解码器是一个自然选择，就像我们从文本生成模型（如GPT-2）中看到的那样。 [ImageGPT](model_doc/imagegpt)使用与GPT-2相同的架构，但是它不是预测序列中的下一个标记，而是预测图像中的下一个像素。除了图像生成，ImageGPT还可以微调进行图像分类。

### 编码器-解码器[[cv-encoder-decoder]]

视觉模型通常使用编码器（也称为主干）在将重要的图像特征传递给Transformer解码器之前提取这些特征。[DETR](model_doc/detr)具有预训练的主干，但它还使用完整的Transformer编码器-解码器架构进行对象检测。编码器学习图像表示并将其与对象查询（每个对象查询是一个集中在图像的某个区域或对象上的学习嵌入）结合在解码器中。DETR预测每个对象查询的边界框坐标和类别标签。

## 自然语言处理

<iframe style="border: 1px solid rgba(0, 0, 0, 0.1);" width="1000" height="450" src="https://www.figma.com/embed?embed_host=share&url=https%3A%2F%2Fwww.figma.com%2Ffile%2FUhbQAZDlpYW5XEpdFy6GoG%2Fnlp-model-timeline%3Fnode-id%3D0%253A1%26t%3D4mZMr4r1vDEYGJ50-1" allowfullscreen></iframe>

### 编码器[[nlp-encoder]]

[BERT](model_doc/bert)是一种仅编码器的Transformer，它随机遮挡输入中的某些标记，以避免看到其他标记，这将使其“作弊”。预训练目标是根据上下文预测遮挡的标记。这使得BERT能够充分利用左右上下文来帮助其学习更深入和更丰富的输入表示。然而，BERT的预训练策略仍有改进的空间。[RoBERTa](model_doc/roberta)在此基础上进行了改进，引入了一种新的预训练模式，包括更长时间的训练和更大的批次，每个时期随机遮挡令牌，而不仅仅是在预处理期间遮挡一次，并且删除了下一个句子预测目标。

提高性能的主要策略是增加模型大小。但是训练大型模型计算成本很高。减少计算成本的一种方法是使用较小的模型，如[DistilBERT](model_doc/distilbert)。DistilBERT使用[知识蒸馏](https://arxiv.org/abs/1503.02531)-一种压缩技术，创建BERT的一个较小版本，同时保留几乎全部的语言理解能力。

然而，大多数Transformer模型仍然趋向于更多的参数，导致了专注于提高训练效率的新模型。[ALBERT](model_doc/albert)通过两种方式降低内存消耗：将更大的词汇嵌入分为两个较小的矩阵，并允许层共享参数。[DeBERTa](model_doc/deberta)增加了一个分离的注意力机制，其中单词和其位置分别编码为两个向量。注意力计算是从这些单独的向量而不是包含单词和位置嵌入的单个向量中进行的。[Longformer](model_doc/longformer)还专注于使注意机制更高效，特别是用于处理具有较长序列长度的文档。它使用局部窗口注意力（仅从每个令牌周围的固定窗口大小计算注意力）和全局注意力（仅适用于特定任务令牌，例如分类的`[CLS]`）的组合，以创建稀疏的注意力矩阵而不是完整的注意力矩阵。

### 解码器[[nlp-decoder]]

[GPT-2](model_doc/gpt2)是一个仅具有解码器的Transformer，它预测序列中的下一个单词。它遮挡右侧的标记，以防止模型通过提前查看来“作弊”。通过在大量文本训练，即使文本有时是不精确或不真实的，GPT-2在生成文本方面表现得非常好。但是GPT-2缺乏BERT的双向上下文，这使其不适用于某些任务。[XLNET](model_doc/xlnet)通过使用置换语言建模目标（PLM）结合了BERT和GPT-2的预训练目标，从而实现了双向学习。

在GPT-2之后，语言模型变得更大，并且现在被称为*大型语言模型（LLMs）*。如果在足够大的数据集上进行预训练，LLMs可以进行零射击学习甚至是零射击学习。[GPT-J](model_doc/gptj)是一个带有60亿参数并在400亿令牌训练的LLM。其之后是[OPT](model_doc/opt)，是一个仅具有解码器的模型系列，其中最大的模型有1750亿个参数并在1800亿个令牌上进行训练。[BLOOM](model_doc/bloom)在同一时间发布，系列中最大的模型具有1760亿个参数，并在46种语言和13种编程语言中的366亿个令牌上进行训练。

### 编码器-解码器[[nlp-encoder-decoder]]

[BART](model_doc/bart)保留了原始的Transformer架构，但是使用*文本填充*的错误方式修改了预训练目标，其中一些文本段被替换为单个`mask`标记。解码器预测未损坏的标记（未来的标记被遮挡），并使用编码器的隐藏状态来帮助它。[Pegasus](model_doc/pegasus)与BART类似，但是Pegasus遮挡整个句子而不是文本段。除了遮挡语言建模，Pegasus还通过缺口句子生成（GSG）进行预训练。GSG目标遮挡了一篇文档中重要的完整句子，并用`mask`标记替换它们。解码器必须从剩余的句子生成输出。[T5](model_doc/t5)是一种更独特的模型，它使用特定的前缀将所有NLP任务转化为文本到文本问题。例如，前缀`Summarize:`表示概括任务。T5通过监督训练（GLUE和SuperGLUE）和自监督训练（随机抽样和丢弃15％的标记）进行预训练。

## 音频

<iframe style="border: 1px solid rgba(0, 0, 0, 0.1);" width="1000" height="450" src="https://www.figma.com/embed?embed_host=share&url=https%3A%2F%2Fwww.figma.com%2Ffile%2Fvrchl8jDV9YwNVPWu2W0kK%2Fspeech-and-audio-model-timeline%3Fnode-id%3D0%253A1%26t%3DmM4H8pPMuK23rClL-1" allowfullscreen></iframe>

### 编码器[[audio-encoder]]

[Wav2Vec2](model_doc/wav2vec2)使用Transformer编码器直接从原始音频波形中学习语音表示。它使用对比任务进行预训练，以确定真实语音表示与一组错误表示之间的区别。[HuBERT](model_doc/hubert)与Wav2Vec2类似，但具有不同的训练过程。目标标签是通过聚类步骤创建的，其中将相似音频的片段分配给一个成为隐藏单元的聚簇。隐藏单元被映射到一个嵌入以进行预测。

### 编码器-解码器[[audio-encoder-decoder]]

[Speech2Text](model_doc/speech_to_text)是专为自动语音识别（ASR）和语音翻译而设计的语音模型。该模型接受从音频波形中提取的对数梅尔频率倒谱系数作为输入特征，并进行预训练来生成转录或翻译。[Whisper](model_doc/whisper)也是一个ASR模型，但与许多其他语音模型不同，它在大量的✨标记的✨音频转录数据上预训练以进行零射击性能。数据集中的大部分还包含非英语语言，这意味着Whisper也可以用于低资源语言。在结构上，Whisper类似于Speech2Text。音频信号被转换为编码器生成的对数梅尔频谱图。解码器根据编码器的隐藏状态和先前的标记从编码器的隐藏状态自回归地生成转录。

## 多模态

<iframe style="border: 1px solid rgba(0, 0, 0, 0.1);" width="1000" height="450" src="https://www.figma.com/embed?embed_host=share&url=https%3A%2F%2Fwww.figma.com%2Ffile%2FcX125FQHXJS2gxeICiY93p%2Fmultimodal%3Fnode-id%3D0%253A1%26t%3DhPQwdx3HFPWJWnVf-1" allowfullscreen></iframe>

### 编码器[[mm-encoder]]

[VisualBERT](model_doc/visual_bert)是一个用于视觉-语言任务的多模态模型，它在BERT和经过预训练的物体检测系统之间组合了嵌入，将其传递给BERT。VisualBERT根据未遮挡的文本和视觉嵌入预测遮挡的文本，并且还必须预测文本是否与图像对齐。当ViT发布时，[ViLT](model_doc/vilt)采用了ViT的架构，因为这样更容易获得图像嵌入。图像嵌入与文本嵌入一起进行联合处理。从那时起，ViLT通过图像文本匹配、遮挡语言建模和整词遮挡进行预训练。

[CLIP](model_doc/clip)采用了一种不同的方法，并对(`image`,`text`)对进行一对预测。图像编码器（ViT）和文本编码器（Transformer）共同在一个4亿个(`image`,`text`)对的数据集上进行训练，以最大限度地提高(`image`,`text`)对之间的图像和文本嵌入的相似性。预训练后，您可以使用自然语言指示CLIP来预测给定图像或反之亦然的文本。

[OWL-ViT](model_doc/owlvit)在CLIP的基础上构建，将其用作进行零射击对象检测的骨干网络。预训练后，会添加一个物体检测头，以对(`class`,`bounding box`)对进行集合预测。

### 编码器-解码器[[mm-编码器-解码器]]

光学字符识别（OCR）是一项长期存在的文本识别任务，通常涉及多个组件来理解图像并生成文本。[TrOCR](model_doc/trocr)使用端到端的Transformer简化了这个过程。编码器是一个类似ViT的图像理解模型，将图像处理为固定大小的补丁。解码器接受编码器的隐藏状态，并自回归地生成文本。[Donut](model_doc/donut)是一个更通用的视觉文档理解模型，不依赖于基于OCR的方法。它使用Swin Transformer作为编码器和多语言BART作为解码器。Donut通过预测基于图像和文本注释的下一个单词来预训练以读取文本。解码器根据提示生成令牌序列。提示由每个下游任务的特殊令牌表示。例如，文档解析具有一个特殊的`parsing`令牌，它与编码器的隐藏状态组合，将文档解析为结构化的输出格式（JSON）。

## 强化学习

<iframe style="border: 1px solid rgba(0, 0, 0, 0.1);" width="1000" height="450" src="https://www.figma.com/embed?embed_host=share&url=https%3A%2F%2Fwww.figma.com%2Ffile%2FiB3Y6RvWYki7ZuKO6tNgZq%2Freinforcement-learning%3Fnode-id%3D0%253A1%26t%3DhPQwdx3HFPWJWnVf-1" allowfullscreen></iframe>

### 解码器[[rl-解码器]]

决策与轨迹Transformer将状态、动作和奖励视为序列建模问题。[决策Transformer](model_doc/decision_transformer)根据返回至结束的奖励、过去的状态和动作生成一系列导致未来期望回报的动作。在最后的K个时间步，这三种模态都被转换为令牌嵌入，并由类似GPT的模型处理以预测未来的动作令牌。[轨迹Transformer](model_doc/trajectory_transformer)也对状态、动作和奖励进行分词，并使用GPT架构进行处理。不同于注重奖励调节的决策Transformer，轨迹Transformer使用束搜索生成未来的动作。