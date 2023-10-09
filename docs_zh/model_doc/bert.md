<!--版权2020 HuggingFace团队。保留所有权利。

根据Apache License，Version 2.0 (许可证)许可; 除非符合许可证的要求，
否则不得使用此文件。您可以在下面的链接处获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用的法律要求或书面同意，根据许可协议分发的软件以"AS IS"的方式分发，
不提供任何明示或暗示的担保，包括但不限于对特定目的的适用性和不侵权的担保。
查看许可证以获取特殊语法的Markdown文件，使用我们的文档构建器(类似MDX)可能无法正确
在您的Markdown查看器中渲染。

-->

# BERT

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=bert">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-bert-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/bert-base-uncased">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## 概述

BERT模型在[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)一文中由Jacob Devlin, Ming-Wei Chang, Kenton Lee和Kristina Toutanova提出。它是使用变压器进行预训练的双向变压器，通过对由多伦多图书语料库和维基百科组成的大型语料库进行掩码语言建模和下一句预测的组合进行预训练。

在论文的摘要中，描述了BERT：

*我们介绍了一种名为BERT的新的语言表示模型，它代表双向编码器变换的表示。与最近的语言表示模型不同，BERT旨在通过在所有层中联合调节左右上下文来预训练深度双向表示，从而从无标签文本中进行深度预训练。结果是，预训练的BERT模型可以仅通过添加一个额外的输出层进行微调，从而创建用于各种任务的最先进的模型，例如问题回答和语言推理，而无需进行重大的任务特定的体系结构修改。*

*BERT的概念简单而实用。它在包括推动GLUE得分达到80.5% （7.7%的绝对改进）、将MultiNLI的准确率提高到86.7%（4.6%的绝对改进）、将SQuAD v1.1问题回答测试F1提高到93.2（1.5%的绝对改进）和将SQuAD v2.0测试F1提高到83.1%（5.1%的绝对改进）在内的十一个自然语言处理任务上取得了最新的成果。*

提示：

- BERT是一个带有绝对位置嵌入的模型，所以通常建议在右边而不是左边填充输入。
- BERT是通过掩码语言建模（MLM）和下一句预测（NSP）目标进行训练的。它在预测掩码标记和自然语言理解方面效果很好，但对于文本生成来说并不是最优选择。
- 使用随机掩码来破坏输入，更准确地说，在预训练过程中，给定的一定比例的标记（通常为15%）通过以下方式进行掩码：

    * 使用特殊掩码标记的概率为0.8
    * 使用与被掩码标记不同的随机标记的概率为0.1
    * 使用相同标记的概率为0.1
    
- 该模型必须预测原始句子，但还有一个第二个目标：输入是两个句子A和B（中间有一个分隔标记）。这两个句子在语料库中有50%的概率是连续的，在剩下的50%中，它们不相关。模型必须预测句子是否连续。

此模型由[thomwolf](https://huggingface.co/thomwolf)贡献。原始代码可以在[此处](https://github.com/google-research/bert)找到。

## 资源

以下是Hugging Face官方资源和社区（由🌎表示）资源的列表，可帮助您开始使用BERT。如果您有兴趣提交资源以包含在这里，请随时提交请求，我们将进行审核！该资源应该理想地展示了一些新东西，而不是重复现有的资源。

<PipelineTag pipeline="text-classification"/>

- 一篇关于[以不同语言进行BERT文本分类](https://www.philschmid.de/bert-text-classification-in-a-different-language)的博客文章。
- 一个用于[对多标签文本分类进行BERT微调](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb)的笔记本。
- 一个关于如何[使用PyTorch对多标签分类进行BERT微调](https://colab.research.google.com/github/abhimishra91/transformers-tutorials/blob/master/transformers_multi_label_classification.ipynb)的笔记本。🌎
- 一个关于如何[使用BERT进行Encoder-Decoder模型的温启动（用于摘要）](https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/BERT2BERT_for_CNN_Dailymail.ipynb)的笔记本。
- [`BertForSequenceClassification`]是由这个[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb)支持的。
- [`TFBertForSequenceClassification`]是由这个[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/text-classification)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification-tf.ipynb)支持的。
- [`FlaxBertForSequenceClassification`]是由这个[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/flax/text-classification)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification_flax.ipynb)支持的。
- [文本分类任务指南](../tasks/sequence_classification)

<PipelineTag pipeline="token-classification"/>

- 一篇关于如何使用[Keras的Hugging Face Transformers进行BERT的命名实体识别](https://www.philschmid.de/huggingface-transformers-keras-tf)的博客文章。
- 一个用于[对命名实体识别进行BERT微调](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/Custom_Named_Entity_Recognition_with_BERT_only_first_wordpiece.ipynb)的笔记本，该笔记本仅使用单词标签的第一个字片段进行标记器。要将单词的标签传播到所有字片段，请参阅该[版本](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT.ipynb)的笔记本。
- [`BertForTokenClassification`]是由这个[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/token-classification)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb)支持的。
- [`TFBertForTokenClassification`]是由这个[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/token-classification)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification-tf.ipynb)支持的。
- [`FlaxBertForTokenClassification`]是由这个[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/flax/token-classification)支持的。
- [Token分类](https://huggingface.co/course/chapter7/2?fw=pt)：🤗 Hugging Face课程的章节。
- [Token分类任务指南](../tasks/token_classification)

<PipelineTag pipeline="fill-mask"/>

- [`BertForMaskedLM`]是由这个[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#robertabertdistilbert-and-masked-language-modeling)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb)支持的。
- [`TFBertForMaskedLM`]是由这个[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_mlmpy)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb)支持的。
- [`FlaxBertForMaskedLM`]是由这个[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#masked-language-modeling)支持的。
- [掩码语言建模](https://huggingface.co/course/chapter7/3?fw=pt)：🤗 Hugging Face课程的章节。
- [掩码语言建模任务指南](../tasks/masked_language_modeling)

<PipelineTag pipeline="question-answering"/>

- [`BertForQuestionAnswering`]是由这个[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb)支持的。
- [`TFBertForQuestionAnswering`]是由这个[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/question-answering)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering-tf.ipynb)支持的。
- [`FlaxBertForQuestionAnswering`]是由这个[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/flax/question-answering)支持的。
- [问答](https://huggingface.co/course/chapter7/7?fw=pt)：🤗 Hugging Face课程的章节。
- [问答任务指南](../tasks/question_answering)

**多项选择**
- [`BertForMultipleChoice`]是由这个[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/multiple-choice)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice.ipynb)支持的。
- [`TFBertForMultipleChoice`]是由这个[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/multiple-choice)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice-tf.ipynb)支持的。
- [多选任务指南](../tasks/multiple_choice)

⚡️ **推理**
- 一篇关于如何[使用Hugging Face Transformers和AWS Inferentia加速BERT推理](https://huggingface.co/blog/bert-inferentia-sagemaker)的博客文章。
- 一篇关于如何[使用DeepSpeed-Inference在GPU上加速BERT推理](https://www.philschmid.de/bert-deepspeed-inference)的博客文章。

⚙️ **预训练**
- 一篇关于[使用Hugging Face Transformers和Habana Gaudi进行BERT预训练](https://www.philschmid.de/pre-training-bert-habana)的博客文章。

🚀 **部署**
- 一篇关于如何[使用Hugging Face Optimum将Transformers转换为ONNX](https://www.philschmid.de/convert-transformers-to-onnx)的博客文章。
- 一篇关于如何[使用Habana Gaudi在AWS上设置Hugging Face Transformers的深度学习环境](https://www.philschmid.de/getting-started-habana-gaudi#conclusion)的博客文章。
- 一篇关于[使用Terraform模块将BERT与HuggingFace，Amazon SageMaker和自动伸缩相结合](https://www.philschmid.de/terraform-huggingface-amazon-sagemaker-advanced)的博客文章。
- 一篇关于[使用HuggingFace、AWS Lambda和Docker实现无服务器BERT](https://www.philschmid.de/serverless-bert-with-huggingface-aws-lambda-docker)的博客文章。
- 一篇关于[Hugging Face Transformers BERT在Amazon SageMaker和Training Compiler上进行微调](https://www.philschmid.de/huggingface-amazon-sagemaker-training-compiler)的博客文章。
- 一篇关于使用Transformers和Amazon SageMaker进行[面向任务的BERT知识蒸馏](https://www.philschmid.de/knowledge-distillation-bert-transformers)的博客文章。

## BertConfig

[[autodoc]] BertConfig
    - all

## BertTokenizer

[[autodoc]] BertTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## BertTokenizerFast

[[autodoc]] BertTokenizerFast

## TFBertTokenizer

[[autodoc]] TFBertTokenizer

## Bert特定的输出

[[autodoc]] models.bert.modeling_bert.BertForPreTrainingOutput

[[autodoc]] models.bert.modeling_tf_bert.TFBertForPreTrainingOutput

[[autodoc]] models.bert.modeling_flax_bert.FlaxBertForPreTrainingOutput

## BertModel

[[autodoc]] BertModel
    - forward

## BertForPreTraining

[[autodoc]] BertForPreTraining
    - forward

## BertLMHeadModel

[[autodoc]] BertLMHeadModel
    - forward

## BertForMaskedLM

[[autodoc]] BertForMaskedLM
    - forward

## BertForNextSentencePrediction

[[autodoc]] BertForNextSentencePrediction
    - forward

## BertForSequenceClassification

[[autodoc]] BertForSequenceClassification
    - forward

## BertForMultipleChoice

[[autodoc]] BertForMultipleChoice
    - forward

## BertForTokenClassification

[[autodoc]] BertForTokenClassification
    - forward

## BertForQuestionAnswering

[[autodoc]] BertForQuestionAnswering
    - forward

## TFBertModel

[[autodoc]] TFBertModel
    - call

## TFBertForPreTraining

[[autodoc]] TFBertForPreTraining
    - call

## TFBertModelLMHeadModel

[[autodoc]] TFBertLMHeadModel
    - call

## TFBertForMaskedLM

[[autodoc]] TFBertForMaskedLM
    - call

## TFBertForNextSentencePrediction

[[autodoc]] TFBertForNextSentencePrediction
    - call

## TFBertForSequenceClassification

[[autodoc]] TFBertForSequenceClassification
    - call

## TFBertForMultipleChoice

[[autodoc]] TFBertForMultipleChoice
    - call

## TFBertForTokenClassification

[[autodoc]] TFBertForTokenClassification
    - call

## TFBertForQuestionAnswering

[[autodoc]] TFBertForQuestionAnswering
    - call

## FlaxBertModel

[[autodoc]] FlaxBertModel
    - __call__

## FlaxBertForPreTraining

[[autodoc]] FlaxBertForPreTraining
    - __call__

## FlaxBertForCausalLM

[[autodoc]] FlaxBertForCausalLM
    - __call__

## FlaxBertForMaskedLM

[[autodoc]] FlaxBertForMaskedLM
    - __call__

## FlaxBertForNextSentencePrediction

[[autodoc]] FlaxBertForNextSentencePrediction
    - __call__

## FlaxBertForSequenceClassification

[[autodoc]] FlaxBertForSequenceClassification
    - __call__

## FlaxBertForMultipleChoice

[[autodoc]] FlaxBertForMultipleChoice
    - __call__

## FlaxBertForTokenClassification

[[autodoc]] FlaxBertForTokenClassification
    - __call__

## FlaxBertForQuestionAnswering

[[autodoc]] FlaxBertForQuestionAnswering
    - __call__