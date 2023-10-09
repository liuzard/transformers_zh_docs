<!---
版权所有2023年HuggingFace团队。
根据Apache许可证2.0版（“许可证”）的规定进行许可；
你不得在未遵守许可证的情况下使用此文件。
你可以在以下位置获得许可证的副本

 http://www.apache.org/licenses/LICENSE-2.0

除非适用的法律要求或书面同意，软件
根据“原样”的基础分发，
没有明示或暗示的担保和
适销性和特定用途的适用性的任何种类。
此许可证下的限制。

-->

# 🤗 转化器笔记本

这里可以找到Hugging Face提供的官方笔记本的列表。

此外，我们还希望在此列出社区创建的有趣内容。
如果你编写了一些利用 🤗 转换器的笔记本，并希望在此列出，请打开
拉动请求，以便将其包含在社区笔记本中。


## Hugging Face的笔记本 🤗

### 文档笔记本

你可以在Colab中打开文档的任何页面作为笔记本（页面上有一个直接的按钮），但如果需要，你也可以在此处列出它们：

| 笔记本 | 描述 |   |   |
|:----------|:-------------|:-------------|------:|
| [库的快速浏览](https://github.com/huggingface/notebooks/blob/main/transformers_doc/en/quicktour.ipynb)  | 演示了转换器中的各种API |[![在Colab中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/transformers_doc/en/quicktour.ipynb)| [![在AWS Studio中打开](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/zh/transformers_doc/quicktour.ipynb)|
| [任务摘要](https://github.com/huggingface/notebooks/blob/main/transformers_doc/en/task_summary.ipynb)  | 如何按任务运行转换器库的模型 |[![在Colab中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/transformers_doc/en/task_summary.ipynb)| [![在AWS Studio中打开](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/zh/transformers_doc/task_summary.ipynb)|
| [预处理数据](https://github.com/huggingface/notebooks/blob/main/transformers_doc/en/preprocessing.ipynb)  | 如何使用分词器预处理数据 |[![在Colab中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/transformers_doc/en/preprocessing.ipynb)| [![在AWS Studio中打开](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/zh/transformers_doc/preprocessing.ipynb)|
| [在预训练模型上进行微调](https://github.com/huggingface/notebooks/blob/main/transformers_doc/en/training.ipynb)  | 如何使用Trainer在预训练模型上进行微调 |[![在Colab中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/transformers_doc/en/training.ipynb)| [![在AWS Studio中打开](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/zh/transformers_doc/training.ipynb)|
| [分词器摘要](https://github.com/huggingface/notebooks/blob/main/transformers_doc/en/tokenizer_summary.ipynb)  | 分词器算法的差异 |[![在Colab中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/transformers_doc/en/tokenizer_summary.ipynb)| [![在AWS Studio中打开](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/zh/transformers_doc/tokenizer_summary.ipynb)|
| [多语言模型](https://github.com/huggingface/notebooks/blob/main/transformers_doc/en/multilingual.ipynb)  | 如何使用库中的多语言模型 |[![在Colab中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/transformers_doc/en/multilingual.ipynb)| [![在AWS Studio中打开](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/zh/transformers_doc/multilingual.ipynb)|


### PyTorch示例

#### 自然语言处理[[pytorch-nlp]]

| 笔记本 | 描述 |   |   |
|:----------|:-------------|:-------------|------:|
| [训练你的分词器](https://github.com/huggingface/notebooks/blob/main/examples/tokenizer_training.ipynb)  | 如何训练和使用你自己的分词器  |[![在Colab中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/tokenizer_training.ipynb)| [![在AWS Studio中打开](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/zh/examples/tokenizer_training.ipynb)|
| [训练你的语言模型](https://github.com/huggingface/notebooks/blob/main/examples/language_modeling_from_scratch.ipynb)   | 如何轻松开始使用转换器  |[![在Colab中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling_from_scratch.ipynb)| [![在AWS Studio中打开](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/zh/examples/language_modeling_from_scratch.ipynb)|
| [如何在文本分类上微调模型](https://github.com/huggingface/notebooks/blob/main/examples/text_classification.ipynb)| 显示如何预处理数据并在任何GLUE任务上微调预训练模型。 | [![在Colab中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb)| [![在AWS Studio中打开](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/zh/examples/text_classification.ipynb)|
| [如何在语言建模上微调模型](https://github.com/huggingface/notebooks/blob/main/examples/language_modeling.ipynb)| 显示如何预处理数据并在因果或遮掩的LM任务上微调预训练模型。 | [![在Colab中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb)| [![在AWS Studio中打开](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/zh/examples/language_modeling.ipynb)|
| [如何对标记分类任务微调模型](https://github.com/huggingface/notebooks/blob/main/examples/token_classification.ipynb)| 显示如何预处理数据并在标记分类任务（NER，POS）上微调预训练模型。 | [![在Colab中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb)| [![在AWS Studio中打开](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/zh/examples/token_classification.ipynb)|
| [如何对问答任务微调模型](https://github.com/huggingface/notebooks/blob/main/examples/question_answering.ipynb)| 显示如何预处理数据并在SQUAD上微调预训练模型。 | [![在Colab中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb)| [![在AWS Studio中打开](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/zh/examples/question_answering.ipynb)|
| [如何对多项选择任务微调模型](https://github.com/huggingface/notebooks/blob/main/examples/multiple_choice.ipynb)| 显示如何预处理数据并在SWAG上微调预训练模型。 | [![在Colab中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice.ipynb)| [![在AWS Studio中打开](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/zh/examples/multiple_choice.ipynb)|
| [如何对翻译任务微调模型](https://github.com/huggingface/notebooks/blob/main/examples/translation.ipynb)| 显示如何预处理数据并在WMT上微调预训练模型。 | [![在Colab中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/translation.ipynb)| [![在AWS Studio中打开](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/zh/examples/translation.ipynb)|
| [如何对摘要任务微调模型](https://github.com/huggingface/notebooks/blob/main/examples/summarization.ipynb)| 显示如何预处理数据并在XSUM上微调预训练模型。 | [![在Colab中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/summarization.ipynb)| [![在AWS Studio中打开](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/zh/examples/summarization.ipynb)|
| [如何从头开始训练语言模型](https://github.com/huggingface/blog/blob/main/notebooks/01_how_to_train.ipynb)| 强调了在自定义数据上有效训练Transformer模型的所有步骤 | [![在Colab中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/01_how_to_train.ipynb)| [![在AWS Studio中打开](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/blog/blob/main/zh/notebooks/01_how_to_train.ipynb)|
| [如何生成文本](https://github.com/huggingface/blog/blob/main/notebooks/02_how_to_generate.ipynb)| 如何使用不同的解码方法进行语言生成 | [![在Colab中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/02_how_to_generate.ipynb)| [![在AWS Studio中打开](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/blog/blob/main/zh/notebooks/02_how_to_generate.ipynb)|
| [如何生成文本（带约束条件）](https://github.com/huggingface/blog/blob/main/notebooks/53_constrained_beam_search.ipynb)| 如何使用用户提供的约束条件指导语言生成 | [![在Colab中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/53_constrained_beam_search.ipynb)| [![在AWS Studio中打开](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/blog/blob/main/zh/notebooks/53_constrained_beam_search.ipynb)|
| [改革者](https://github.com/huggingface/blog/blob/main/notebooks/03_reformer.ipynb)| 改革者如何推动语言建模的极限 | [![在Colab中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/patrickvonplaten/blog/blob/main/notebooks/03_reformer.ipynb)| [![在AWS Studio中打开](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/patrickvonplaten/blog/blob/main/zh/notebooks/03_reformer.ipynb)|

#### 计算机视觉[[pytorch-cv]]

#### 自然语言处理[[tensorflow-nlp]]

| Notebook     |      Description      |   |   |
|:----------|:-------------|:-------------|------:|
| [How to fine-tune a model on text classification (Text Classification)](https://github.com/huggingface/notebooks/blob/main/examples/text_classification_tf.ipynb)| Show how to preprocess the data and fine-tune any transformer model on Text Classification    | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification_tf.ipynb)| [![Open in AWS Studio](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/examples/text_classification_tf.ipynb)|
| [How to train a language model (Text Generation)](https://github.com/huggingface/notebooks/blob/main/examples/text_generation_tf.ipynb)| Show how to train a transformer model for text generation tasks using the `tf.data.Dataset` API   | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_generation_tf.ipynb)| [![Open in AWS Studio](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/examples/text_generation_tf.ipynb)|
| [How to use pipelines in TensorFlow](https://github.com/huggingface/notebooks/blob/main/examples/pipelines_tf.ipynb)| Highlight the easy-to-use `tf.data.Dataset` input pipelines for token classification and text generation tasks | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/pipelines_tf.ipynb)| [![Open in AWS Studio](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/examples/pipelines_tf.ipynb)|
| [How to write a training loop from scratch](https://github.com/huggingface/notebooks/blob/main/examples/training_loop_tf.ipynb)| Show how to write a training loop in TensorFlow for fine-tuning transformers | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/training_loop_tf.ipynb)| [![Open in AWS Studio](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/examples/training_loop_tf.ipynb)|
| [How to use TensorFlow with ktrain](https://github.com/huggingface/notebooks/blob/main/examples/text_classification_with_ktrain.ipynb)| Example showing how to use `ktrain` to train a classifier based on a pretrained transformer | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification_with_ktrain.ipynb)| [![Open in AWS Studio](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/examples/text_classification_with_ktrain.ipynb)|

#### Computer Vision[[tensorflow-cv]]

| Notebook     |      Description      |   |   |
|:----------|:-------------|:-------------|------:|
| [How to fine-tune a model on image classification (TorchVision)](https://github.com/huggingface/notebooks/blob/main/examples/image_classification_tf.ipynb)| Show how to preprocess the data and fine-tune any pretrained Vision model on Image Classification with TensorFlow    | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification_tf.ipynb)| [![Open in AWS Studio](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/examples/image_classification_tf.ipynb)|
| [How to fine-tune a model on image classification (Albumentations)](https://github.com/huggingface/notebooks/blob/main/examples/image_classification_albumentations_tf.ipynb)| Show how to preprocess the data using Albumentations and fine-tune any pretrained Vision model on Image Classification with TensorFlow   | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification_albumentations_tf.ipynb)| [![Open in AWS Studio](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/examples/image_classification_albumentations_tf.ipynb)|
| [How to fine-tune a model on image classification (Keras)](https://github.com/huggingface/notebooks/blob/main/examples/image_classification_keras_tf.ipynb)| Show how to preprocess the data using Keras and fine-tune any pretrained Vision model on Image Classification with TensorFlow   | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification_keras_tf.ipynb)| [![Open in AWS Studio](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/examples/image_classification_keras_tf.ipynb)|
| [How to build an image similarity system with TensorFlow](https://github.com/huggingface/notebooks/blob/main/examples/image_similarity_tf.ipynb)| Show how to build an image similarity system using TensorFlow   | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_similarity_tf.ipynb)| [![Open in AWS Studio](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/examples/image_similarity_tf.ipynb)|

#### Audio[[tensorflow-audio]]

| Notebook     |      Description      |   |   |
|:----------|:-------------|:-------------|------:|
| [How to fine-tune a speech recognition model in English](https://github.com/huggingface/notebooks/blob/main/examples/speech_recognition_tf.ipynb)| Show how to preprocess the data and fine-tune a pretrained Speech model on TIMIT with TensorFlow  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/speech_recognition_tf.ipynb)| [![Open in AWS Studio](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/examples/speech_recognition_tf.ipynb)|
| [How to fine-tune a speech recognition model in any language](https://github.com/huggingface/notebooks/blob/main/examples/multi_lingual_speech_recognition_tf.ipynb)| Show how to preprocess the data and fine-tune a multi-lingually pretrained Speech model on Common Voice with TensorFlow  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multi_lingual_speech_recognition_tf.ipynb)| [![Open in AWS Studio](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/examples/multi_lingual_speech_recognition_tf.ipynb)|
| [How to fine-tune a model on audio classification](https://github.com/huggingface/notebooks/blob/main/examples/audio_classification_tf.ipynb)| Show how to preprocess the data and fine-tune a pretrained Speech model on Keyword Spotting with TensorFlow  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/audio_classification_tf.ipynb)| [![Open in AWS Studio](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/examples/audio_classification_tf.ipynb)|

#### Other modalities[[tensorflow-other]]

| Notebook     | Description                                                                             |   |   |
|:----------|:----------------------------------------------------------------------------------------|:-------------|------:|
| [Probabilistic Time Series Forecasting with TensorFlow](https://github.com/huggingface/notebooks/blob/main/examples/time_series_transformers_tf.ipynb) | See how to train Time Series Transformer on a custom dataset using TensorFlow                              | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/time_series_transformers_tf.ipynb) | [![Open in AWS Studio](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/examples/time_series_transformers_tf.ipynb) |

#### Utility notebooks[[tensorflow-utility]]

| Notebook     |      Description      |   |   |
|:----------|:-------------|:-------------|------:|
| [How to export model to TensorFlow SavedModel](https://github.com/huggingface/notebooks/blob/main/examples/export_tensorflow_savedmodel.ipynb)| Highlight how to export and serve a TensorFlow SavedModel |
| [How to use benchmarks](https://github.com/huggingface/notebooks/blob/main/examples/benchmark_tf.ipynb)| How to benchmark models with transformers | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/benchmark_tf.ipynb)| [![Open in AWS Studio](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/examples/benchmark_tf.ipynb)|

| Notebook     |      Description      |   |   |
|:----------|:-------------|:-------------|------:|
| [训练你的分词器（Train your tokenizer）](https://github.com/huggingface/notebooks/blob/main/examples/tokenizer_training.ipynb)  | 如何训练和使用你自己的分词器  |[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/tokenizer_training.ipynb)| [![在 AWS Studio 中打开](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/examples/tokenizer_training.ipynb)|
| [训练你的语言模型（Train your language model）](https://github.com/huggingface/notebooks/blob/main/examples/language_modeling_from_scratch-tf.ipynb)   | 如何轻松开始使用 Transformers  |[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling_from_scratch-tf.ipynb)| [![在 AWS Studio 中打开](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/examples/language_modeling_from_scratch-tf.ipynb)|
| [如何在文本分类上对模型进行微调（How to fine-tune a model on text classification）](https://github.com/huggingface/notebooks/blob/main/examples/text_classification-tf.ipynb)| 展示如何预处理数据并在任何 GLUE 任务上对预训练模型进行微调。 | [![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification-tf.ipynb)| [![在 AWS Studio 中打开](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/examples/text_classification-tf.ipynb)|
| [如何在语言建模上对模型进行微调（How to fine-tune a model on language modeling）](https://github.com/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb)| 展示如何预处理数据并在因果或掩码语言模型任务上对预训练模型进行微调。 | [![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb)| [![在 AWS Studio 中打开](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb)|
| [如何在标记分类上对模型进行微调（How to fine-tune a model on token classification）](https://github.com/huggingface/notebooks/blob/main/examples/token_classification-tf.ipynb)| 展示如何预处理数据并在标记分类任务（NER、PoS）上对预训练模型进行微调。 | [![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification-tf.ipynb)| [![在 AWS Studio 中打开](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/examples/token_classification-tf.ipynb)|
| [如何在问答中对模型进行微调（How to fine-tune a model on question answering）](https://github.com/huggingface/notebooks/blob/main/examples/question_answering-tf.ipynb)| 展示如何预处理数据并在 SQUAD 上对预训练模型进行微调。 | [![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering-tf.ipynb)| [![在 AWS Studio 中打开](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/examples/question_answering-tf.ipynb)|
| [如何在多项选择上对模型进行微调（How to fine-tune a model on multiple choice）](https://github.com/huggingface/notebooks/blob/main/examples/multiple_choice-tf.ipynb)| 展示如何预处理数据并对预训练模型进行微调，针对 SWAG 进行预测模型。 | [![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice-tf.ipynb)| [![在 AWS Studio 中打开](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/examples/multiple_choice-tf.ipynb)|
| [如何在翻译上对模型进行微调（How to fine-tune a model on translation）](https://github.com/huggingface/notebooks/blob/main/examples/translation-tf.ipynb)| 展示如何预处理数据并在 WMT 上对预训练模型进行微调。 | [![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/translation-tf.ipynb)| [![在 AWS Studio 中打开](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/examples/translation-tf.ipynb)|
| [如何在摘要上对模型进行微调（How to fine-tune a model on summarization）](https://github.com/huggingface/notebooks/blob/main/examples/summarization-tf.ipynb)| 展示如何预处理数据并在 XSUM 上对预训练模型进行微调。 | [![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/summarization-tf.ipynb)| [![在 AWS Studio 中打开](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/examples/summarization-tf.ipynb)|

#### 计算机视觉[[tensorflow-cv]]

| Notebook                                                                                                                                                 | Description                                                                                         |   |   |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------|:-------------|------:|
| [如何在图像分类上对模型进行微调（How to fine-tune a model on image classification）](https://github.com/huggingface/notebooks/blob/main/examples/image_classification-tf.ipynb)            | 展示如何预处理数据并在任何预训练的计算机视觉模型上对图像分类进行微调   | [![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification-tf.ipynb)| [![在 AWS Studio 中打开](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/examples/image_classification-tf.ipynb)|
| [如何在语义分割上对 SegFormer 模型进行微调（How to fine-tune a SegFormer model on semantic segmentation）](https://github.com/huggingface/notebooks/blob/main/examples/semantic_segmentation-tf.ipynb) | 展示如何预处理数据并在预训练的 SegFormer 模型上对语义分割进行微调 | [![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/semantic_segmentation-tf.ipynb)| [![在 AWS Studio 中打开](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/examples/semantic_segmentation-tf.ipynb)|

#### 生物序列[[tensorflow-bio]]

| Notebook     |      Description      |   |   |
|:----------|:-------------|:-------------|------:|
| [如何对预训练蛋白质模型进行微调（How to fine-tune a pre-trained protein model）](https://github.com/huggingface/notebooks/blob/main/examples/protein_language_modeling-tf.ipynb) | 展示如何对蛋白质进行分词并在大型预训练蛋白质“语言”模型上进行微调 | [![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/protein_language_modeling-tf.ipynb) | [![在 AWS Studio 中打开](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/examples/protein_language_modeling-tf.ipynb) |

#### 实用笔记本[[tensorflow-utility]]

| Notebook     |      Description      |   |                                                                                                                                                                                      |
|:----------|:-------------|:-------------|------:|
| [如何在 TPU 上训练 TF/Keras 模型（How to train TF/Keras models on TPU）](https://github.com/huggingface/notebooks/blob/main/examples/tpu_training-tf.ipynb) | 展示如何在 Google 的 TPU 硬件上以高速训练模型 | [![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/tpu_training-tf.ipynb) | [![在 AWS Studio 中打开](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/examples/tpu_training-tf.ipynb) |

### Optimum notebooks

🤗  [Optimum](https://github.com/huggingface/optimum) 是 🤗 Transformers 的扩展，提供了一组性能优化工具，实现在特定硬件上训练和运行模型的最大效率。

| Notebook     |      Description      |   |   |
|:----------|:-------------|:-------------|------:|
| [如何使用 ONNX Runtime 对文本分类模型进行量化（How to quantize a model with ONNX Runtime for text classification）](https://github.com/huggingface/notebooks/blob/main/examples/text_classification_quantization_ort.ipynb)| 展示如何使用 ONNX Runtime 对任何 GLUE 任务的模型应用静态和动态量化。 | [![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification_quantization_ort.ipynb)| [![在 AWS Studio 中打开](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/examples/text_classification_quantization_ort.ipynb)|
| [如何使用 Intel Neural Compressor 对文本分类模型进行量化（How to quantize a model with Intel Neural Compressor for text classification）](https://github.com/huggingface/notebooks/blob/main/examples/text_classification_quantization_inc.ipynb)| 展示如何使用 Intel Neural Compressor (INC) 对任何 GLUE 任务的模型应用静态、动态和训练感知量化。 | [![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification_quantization_inc.ipynb)| [![在 AWS Studio 中打开](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/examples/text_classification_quantization_inc.ipynb)|
| [如何在文本分类上使用 ONNX Runtime 对模型进行微调（How to fine-tune a model on text classification with ONNX Runtime）](https://github.com/huggingface/notebooks/blob/main/examples/text_classification_ort.ipynb)| 展示如何预处理数据并使用 ONNX Runtime 在任何 GLUE 任务上微调模型。 | [![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification_ort.ipynb)| [![在 AWS Studio 中打开](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/examples/text_classification_ort.ipynb)|
| [如何在摘要上使用 ONNX Runtime 对模型进行微调（How to fine-tune a model on summarization with ONNX Runtime）](https://github.com/huggingface/notebooks/blob/main/examples/summarization_ort.ipynb)| 展示如何预处理数据并使用 ONNX Runtime 在 XSUM 上微调模型。 | [![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/summarization_ort.ipynb)| [![在 AWS Studio 中打开](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/examples/summarization_ort.ipynb)|

## 社区笔记本:

社区开发的更多笔记本可在[此处](https://hf.co/docs/transformers/community#community-notebooks)找到。