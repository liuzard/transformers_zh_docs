<!--
版权 2022 年 HuggingFace 团队。保留所有权利。

根据 Apache 许可证 2.0 版（"许可证"）获得许可；您可能不会使用此文件，除非符合许可证的规定。
您可以通过点击以下链接获得许可证的副本。

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”基础分发的，不附带任何明示或暗示的担保或条件。
有关具体语言的版本和许可证下的限制的详细信息，请参阅许可证。

⚠️ 请注意，此文件采用 Markdown 格式，但包含特定于我们的文档生成器（类似于 MDX）的语法，您的 Markdown 查看器可能无法正确呈现。
-->

# GPT-Sw3

## 概述

GPT-Sw3 模型最早由 Ariel Ekgren、Amaru Cuba Gyllensten、Evangelia Gogoulou、Alice Heiman、Severine Verlinden、Joey Öhman、Fredrik Carlsson、Magnus Sahlgren 在 [Lessons Learned from GPT-SW3: Building the First Large-Scale Generative Language Model for Swedish](http://www.lrec-conf.org/proceedings/lrec2022/pdf/2022.lrec-1.376.pdf) 中提出。

自那篇论文以来，作者已经扩展了他们的工作，并在他们的新 1.2TB 语料库“北欧堆”中对新模型进行了训练。

GPT-Sw3 是由 AI Sweden 与 RISE 和 WASP WARA for Media and Language 协作开发的一系列大型仅解码器预训练的 transformer 语言模型。GPT-Sw3 在包含 320B 个标记的瑞典语、挪威语、丹麦语、冰岛语、英语和编程代码的数据集上进行了预训练。该模型使用基于因果语言模型（CLM）目标和 NeMo Megatron GPT 实现进行了预训练。

本模型由 [AI Sweden](https://huggingface.co/AI-Sweden) 贡献。

该实现使用 [GPT2Model](https://huggingface.co/docs/transformers/model_doc/gpt2) 与我们的 `GPTSw3Tokenizer` 结合使用。这意味着 `AutoTokenizer` 和 `AutoModelForCausalLM` 分别映射到我们的分词器实现和相应的 GPT2 模型实现。
*请注意，使用我们的分词器需要安装 sentencepiece，可以使用以下命令安装：* `pip install transformers[sentencepiece]` 或 `pip install sentencepiece`

使用示例：
```python
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("AI-Sweden/gpt-sw3-356m")
>>> model = AutoModelForCausalLM.from_pretrained("AI-Sweden/gpt-sw3-356m")

>>> input_ids = tokenizer("Träd är fina för att", return_tensors="pt")["input_ids"]

>>> generated_token_ids = model.generate(inputs=input_ids, max_new_tokens=10, do_sample=True)[0]

>>> print(tokenizer.decode(generated_token_ids))
Träd är fina för att de är färgstarka. Men ibland är det fint
```

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [因果语言模型任务指南](../tasks/language_modeling)

## GPTSw3Tokenizer

[[自动文档]] GPTSw3Tokenizer
    - save_vocabulary
