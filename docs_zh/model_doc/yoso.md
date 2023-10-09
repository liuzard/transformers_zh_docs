<!--版权2022 HuggingFace团队。保留所有权利。

根据Apache许可证第2.0版（“许可证”）获得许可;除非符合许可证，否则不得使用此文件
许可证的复制可在获得许可证时获得

Http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则依据许可证分发的软件是分发的
“基础上，没有任何形式的明示或暗示的保证或条件。有关许可的详细信息，参见
特定语言中的特定说明。

⚠️请注意，此文件是Markdown格式，但包含我们的文档生成器（类似于MDX）的特定语法，可能无法在
您的Markdown查看器中正确呈现。

-->

# YOSO

## 概述

YOSO模型是由Zhanpeng Zeng, Yunyang Xiong, Sathya N. Ravi, Shailesh Acharya, Glenn Fung和Vikas Singh在[You Only Sample (Almost) Once: Linear Cost Self-Attention Via Bernoulli Sampling](https://arxiv.org/abs/2111.09714)一文中提出的。
YOSO通过基于局部敏感哈希（LSH）的伯努利采样方案近似标准的softmax自注意力。原理上，所有伯努利随机变量可以使用单个哈希进行采样。

以下是论文的摘要：

*基于Transformer的模型在自然语言处理（NLP）中被广泛使用。Transformer模型的核心是自注意机制，它捕捉输入序列中令牌对的相互作用，并且对序列长度呈二次依赖。在较长的序列上训练这样的模型是昂贵的。在本文中，我们展示了一种基于局部敏感哈希（LSH）的伯努利采样注意机制，将这样的模型的二次复杂性降低为线性。我们通过将自注意视为与伯努利随机变量相关联的个别令牌的总和来绕过二次成本，这些伯努利随机变量原则上可以通过单个哈希同时采样（尽管在实践中，这个数字可能是一个小常数）。这导致了一种有效的采样方案，用于估计依赖于特定修改的LSH（以便在GPU架构上进行部署）的自注意。我们在具有标准512序列长度的GLUE基准上评估了我们的算法，在性能上与标准预训练Transformer相比具有有利。在长范围竞技场（LRA）基准上，用于评估长序列上的性能，我们的方法取得了与softmax自注意力一致的结果，但具有可观的速度提升和内存节省，并且往往优于其他高效的自注意力方法。我们的代码可以在这个https URL上找到*

提示：

- YOSO注意算法通过自定义CUDA核实现，这是在GPU上并行执行多次的CUDA C++编写的函数。
- 核心提供了`fast_hash`函数，该函数使用快速哈达玛变换来近似查询和键的随机投影。使用这些
哈希码，`lsh_cumulation`函数通过基于LSH的伯努利采样来近似自注意力。
- 要使用自定义核心，用户应设置`config.use_expectation = False`。为了确保核心成功编译，
用户必须安装正确版本的PyTorch和cudatoolkit。默认情况下，`config.use_expectation = True`，使用YOSO-E
不需要编译CUDA核心。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/yoso_architecture.jpg"
alt="drawing" width="600"/> 

<small> YOSO注意算法。摘自<a href="https://arxiv.org/abs/2111.09714">原始论文</a>。</small>

该模型由[novice03](https://huggingface.co/novice03)贡献。原始代码可以在[这里](https://github.com/mlpen/YOSO)找到。

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)
- [掩码语言建模任务指南](../tasks/masked_language_modeling)
- [多项选择任务指南](../tasks/multiple_choice)

## YosoConfig

[[autodoc]] YosoConfig


## YosoModel

[[autodoc]] YosoModel
    - 前向


## YosoForMaskedLM

[[autodoc]] YosoForMaskedLM
    - 前向


## YosoForSequenceClassification

[[autodoc]] YosoForSequenceClassification
    - 前向

## YosoForMultipleChoice

[[autodoc]] YosoForMultipleChoice
    - 前向


## YosoForTokenClassification

[[autodoc]] YosoForTokenClassification
    - 前向


## YosoForQuestionAnswering

[[autodoc]] YosoForQuestionAnswering
    - 前向