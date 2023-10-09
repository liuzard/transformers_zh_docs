<!--版权所有2020年抱抱小方团队。保留所有权利。

根据Apache许可证第2.0版（“许可证”）的规定，您除非符合许可证的规定，否则不得使用此文件。
您可以在以下地址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”提供的，不附带任何明示或暗示的任何形式的保证或条件。有关许可证的具体内容，请参阅许可证。

⚠️请注意，此文件采用Markdown格式，但包含我们文档生成器（类似于MDX）的特定语法，可能无法在您的Markdown查看器中正确呈现。-->

# RetriBERT

<Tip warning={true}>

此模型仅处于维护模式，因此我们不接受任何更改其代码的新PR。

如果您在运行此模型时遇到任何问题，请重新安装支持此模型的最后版本：v4.30.0。
您可以通过运行以下命令来执行此操作：`pip install -U transformers==4.30.0`。

</Tip>

## 概览

RetriBERT模型是在博文[Explain Anything Like I'm Five: A Model for Open Domain Long Form
Question Answering](https://yjernite.github.io/lfqa.html)中提出的。RetriBERT是一个使用单个或一对BERT编码器进行文本的稠密语义索引的小型模型。

此模型由[yjernite](https://huggingface.co/yjernite)贡献。可以在[此处](https://github.com/huggingface/transformers/tree/main/examples/research-projects/distillation)找到用于训练和使用该模型的代码。

## RetriBertConfig

[[autodoc]] RetriBertConfig

## RetriBertTokenizer

[[autodoc]] RetriBertTokenizer

## RetriBertTokenizerFast

[[autodoc]] RetriBertTokenizerFast

## RetriBertModel

[[autodoc]] RetriBertModel
    - forward