版权所有2020年The HuggingFace团队。保留所有权利。

根据Apache 2.0许可证（“许可证”），除非符合许可证，否则您不得使用此文件。您可以在以下位置获得许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何明示或暗示的保证或条件。请参阅许可证以了解许可证下的特定语言和限制。

⚠️ 请注意，此文件以Markdown格式编写，但包含我们文档生成器的特定语法（类似于MDX），可能无法在您的Markdown查看器中正确呈现。

# CPM

## 概述

CPM模型在《CPM：一种大规模生成式的中文预训练语言模型》一文中提出，作者为郑彦张，韩旭，周浩，柯沛，谷雨仙，叶德明，秦宇佳，
苏宇升，季豪哲，关建，齐凡超，王晓智，郑娜娜，曾国阳，曹焕琪，陈胜琦，李岱轩，孙祯博，刘智源，黄民烈，韩文涛，唐杰，李卷子，
朱小燕，孙茂松。

论文中的摘要如下：

“预训练语言模型（Pre-trained Language Models，PLMs）已经被证明对各种下游自然语言处理（NLP）任务有益。最近，拥有1750亿参数和570GB训练数据的
GPT-3因其少量训练样本（甚至零样本）学习的能力而引起了很多关注。然而，将GPT-3应用于解决中文NLP任务仍然具有挑战性，因为GPT-3的训练语料库主要是英文，
而且参数是不公开的。在这份技术报告中，我们发布了一种中国预训练语言模型（Chinese Pre-trained Language Model，CPM），它在大规模中文训练数据上进行了生成式
预训练。据我们所知，CPM是具有26亿参数和100GB中文训练数据的最大中文预训练语言模型，可以促进多个下游的中文NLP任务，例如会话，文章生成，填空测试和
语言理解。大量实验证明，CPM在少量训练样本（甚至零样本）学习的情况下，在许多NLP任务上取得了很好的性能。”

该模型由[canwenxu](https://huggingface.co/canwenxu)贡献。原始实现可在此处找到：https://github.com/TsinghuaAI/CPM-Generate

注意：我们这里只有一个分词器，因为模型架构与GPT-2相同。

## CpmTokenizer

[[autodoc]] CpmTokenizer

## CpmTokenizerFast

[[autodoc]] CpmTokenizerFast