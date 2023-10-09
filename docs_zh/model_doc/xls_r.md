<!--版权所有2021年HuggingFace团队。保留所有权利。

根据Apache许可证第2版许可，除非遵守许可证，否则不得使用此文件。

您可以在下面获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

请注意，此文件为Markdown格式，但包含我们的文档构建器的特定语法（类似于MDX），在您的Markdown查看器中可能无法正确显示。

-->

# XLS-R

## 概述

[XLS-R：基于wav2vec 2.0的自监督跨语言语音表示学习]（https://arxiv.org/abs/2111.09296）由Arun Babu，Changhan Wang，Andros Tjandra，Kushal Lakhotia，Qiantong Xu，Naman提出。Goyal，Kritika Singh，Patrick von Platen，Yatharth Saraf，Juan Pino，Alexei Baevski，Alexis Conneau，Michael Auli。

论文摘要如下：

*本文介绍了XLS-R，一种基于wav2vec 2.0的大规模跨语言语音表示学习模型。我们使用多达20亿个参数在近50万小时的公开可用语音音频中训练模型，这比已知的最大规模先前工作的公共数据量增加了一个数量级。我们的评估涵盖了各种任务、领域、数据范围和语言，包括高和低资源语言。在CoVoST-2语音翻译基准测试中，我们将先前的最先进水平提高了平均7.4 BLEU，涉及21个翻译方向到英语。对于语音识别，XLS-R相对于已知的最佳先前工作在BABEL、MLS、CommonVoice和VoxPopuli上降低了14-34%的错误率。XLS-R还在VoxLingua107语言识别上树立了新的最先进水平。此外，我们还表明，具有足够大小的模型时，跨语言预训练可以胜过仅英语预训练，当将英语语音翻译成其他语言时，这种设置有利于单语预训练。我们希望XLS-R能够帮助提高世界上更多语言的语音处理任务。*

提示:

- XLS-R是一个接受对应于语音信号原始波形的浮点数组的语音模型。
- XLS-R模型是使用连接主义时间分类（CTC）进行训练的，因此模型输出必须使用[`Wav2Vec2CTCTokenizer`]进行解码。

相关检查点可以在https://huggingface.co/models?other=xls_r下找到。

XLS-R的架构基于Wav2Vec2模型，因此可以参考[Wav2Vec2的文档页面]（wav2vec2）。

原始代码可以在[这里找到]（https://github.com/pytorch/fairseq/tree/master/fairseq/models/wav2vec）。