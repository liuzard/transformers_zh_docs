<!--版权所有2021年NVIDIA Corporation和HuggingFace团队。保留所有权利。

根据Apache许可证第2版（“许可证”），您不得使用此文件，除非符合许可证的规定。您可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非法律要求或书面同意，根据许可证分发的软件是按“原样”分发的，不附带任何明示或暗示的担保或条件。请参阅许可证以获取特定语言下许可证的权限和限制。

⚠️请注意，此文件以Markdown格式编写，但包含我们文档构建器（类似于MDX）的特定语法，可能在您的Markdown查看器中无法正确呈现。-->

# MegatronGPT2

## 概述

MegatronGPT2模型是由Mohammad Shoeybi、Mostofa Patwary、Raul Puri、Patrick LeGresley、Jared Casper和Bryan Catanzaro在[《Megatron-LM：使用模型并行性训练数十亿参数的语言模型》](https://arxiv.org/abs/1909.08053)中提出的。

来自论文的摘要如下：

*语言模型的最新研究表明，训练大型Transformer模型可以推动自然语言处理应用的技术发展。但是，由于内存限制，训练非常庞大的模型可能非常困难。在本文中，我们介绍了训练非常大的Transformer模型的技术，并实现了一种简单高效的层间模型并行方法，使得可以训练拥有数十亿参数的Transformer模型。我们的方法不需要新的编译器或库更改，是与流水线模型并行性正交并互补的，并且可以在本机PyTorch中通过插入少量的通信操作来完全实现。我们通过使用512台GPU，将基于Transformer的模型收敛到83亿参数，并在整个应用程序中维持15.1千兆浮点操作每秒的计算性能，与能够维持39兆浮点操作每秒峰值性能的单一GPU基线相比具有76％的可扩放性效率。为了证明大型语言模型可以进一步推动技术发展，我们训练了一个与GPT-2相似的83亿参数Transformer语言模型以及一个与BERT相似的39亿参数模型。我们表明，在BERT-like模型中，对于层归一化的放置需要引起注意，因为这对于在模型规模增长时实现性能提升至关重要。使用GPT-2模型，我们在WikiText103（相对于15.8的SOTA困惑度达到10.8）和LAMBADA（相对于63.2%的SOTA准确率达到66.5%）数据集上实现了SOTA结果。我们的BERT模型在RACE数据集上实现了SOTA结果（相对于89.4％的SOTA准确率达到90.9％）。*

提示：

我们提供了预训练的[GPT2-345M](https://ngc.nvidia.com/catalog/models/nvidia:megatron_lm_345m)检查点，供评估或微调下游任务使用。

要访问这些检查点，请首先[注册](https://ngc.nvidia.com/signup)，并设置NVIDIA GPU云（NGC）注册表CLI。有关下载模型的更多文档，请参阅[NGC文档](https://docs.nvidia.com/dgx/ngc-registry-cli-user-guide/index.html#topic_6_4_1)。

或者，您可以直接使用以下命令下载检查点：

```bash
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_lm_345m/versions/v0.0/zip -O
megatron_gpt2_345m_v0_0.zip
```

一旦您从NVIDIA GPU云（NGC）获得了检查点，您需要将其转换为Hugging Face Transformers GPT2实现可轻松加载的格式。

以下命令允许您进行转换。我们假设文件夹`models/megatron_gpt2`包含`megatron_gpt2_345m_v0_0.zip`，并且该命令在该文件夹中运行：

```bash
python3 $PATH_TO_TRANSFORMERS/models/megatron_gpt2/convert_megatron_gpt2_checkpoint.py megatron_gpt2_345m_v0_0.zip
```

此模型由[jdemouth](https://huggingface.co/jdemouth)提供。原始代码可以在[此处](https://github.com/NVIDIA/Megatron-LM)找到。该存储库包含Megatron语言模型的多GPU和多节点实现。特别是，它包含使用“张量并行”和“流水线并行”技术的混合模型并行方法。