<!--版权2023年The HuggingFace团队。保留所有权利。

根据Apache许可证第2.0版（“许可证”）许可；除非符合许可证的条款，否则不得使用此文件。
你可以在以下位置获取许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件按“原样”分发，不附带任何形式的保证或条件。有关详细信息，请参阅许可证。

⚠️请注意，此文件是Markdown格式的，但包含我们的文档生成器（类似于MDX）的特定语法，可能在你的Markdown查看器中无法正确渲染。

-->

# CodeLlama

## 概述

Code Llama模型是由Baptiste Rozière，Jonas Gehring，Fabian Gloeckle，Sten Sootla，Itai Gat，Xiaoqing Ellen Tan，Yossi Adi，Jingyu Liu，Tal Remez，Jérémy Rapin，Artyom Kozhevnikov，Ivan Evtimov，Joanna Bitton，Manish Bhatt，Cristian Canton Ferrer，Aaron Grattafiori，Wenhan Xiong，Alexandre Défossez，Jade Copet，Faisal Azhar，Hugo Touvron，Louis Martin，Nicolas Usunier，Thomas Scialom和Gabriel Synnaeve在[Code Llama：面向代码的开放基础模型](https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/)中提出的。

论文中的摘要如下：

*我们发布了Code Llama，这是一个基于Llama 2的大规模编码语言模型系列，提供了在开放模型中最先进的性能，填充能力，支持大型输入上下文以及零-shot编程任务中的指令跟随能力。我们提供多个版本以覆盖广泛的应用场景：基础模型（Code Llama），Python专业化（Code Llama - Python）以及指令跟随模型（Code Llama - Instruct），每个模型都具有7B，13B和34B的参数。所有模型都以16ktoken的序列进行训练，并在具有高达100ktoken的输入上显示出改进。7B和13B Code Llama和Code Llama - Instruct变体支持基于周围内容的填充。 Code Llama在多个代码基准测试中达到了开放模型的最先进性能，HumanEval和MBPP分别达到了53%和55%的分数。值得注意的是，Code Llama - Python 7B在HumanEval和MBPP上的表现优于Llama 2 70B，并且我们的所有模型都优于其他所有公开可用的模型。我们根据一种宽松的许可证发布Code Llama，该许可证允许进行研究和商业使用。*

在这里查看所有Code Llama模型[here](https://huggingface.co/models?search=code_llama)，并在[codellama org](https://huggingface.co/codellama)中的官方发布。

<Tip warning={true}>

基于Code Llama的`Llama2`系列模型在训练时使用了`bfloat16`，但原始推断使用了`float16`。让我们来看看不同的精度：

* `float32`：PyTorch约定在模型初始化时以`float32`加载模型，无论模型权重存储时使用的是哪种`dtype`。`transformers`为了与PyTorch保持一致也遵循了这个约定。这是默认选项。如果你希望`AutoModel` API使用存储权重类型来加载检查点，你必须指定`torch_dtype="auto"`，例如`model = AutoModelForCausalLM.from_pretrained("path", torch_dtype = "auto")`。
* `bfloat16`：Code Llama是使用这种精度进行训练的，因此我们建议在后续训练或微调时使用它。
* `float16`：我们建议使用这种精度进行推断，因为它通常比`bfloat16`更快，并且评估指标显示与`bfloat16`相比没有明显的性能下降。你也可以使用`bfloat16`进行推断，并建议你在微调之后使用`float16`和`bfloat16`检查推断结果。

如上所述，存储权重的`dtype`大多无关紧要，除非你在初始化模型时使用`torch_dtype="auto"`。原因是模型首先会被下载（使用在线检查点的`dtype`），然后会被转换为`torch`的默认`dtype`（变为`torch.float32`）。如果指定了`torch_dtype`，则将使用指定的`dtype`。

</Tip>

提示：

- 这些模型和`Llama2`模型具有相同的架构
- 支持填充任务。你应该使用`tokenizer.fill_token`，将其放在要进行填充的位置。
- 模型转换脚本与`Llama2`系列相同：

以下是示例用法
```bash
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
```
注意，执行该脚本需要足够的CPU RAM来托管以float16精度保存的整个模型（即使最大版本分为几个检查点，每个检查点都包含模型的一部分权重，因此我们需要将所有检查点加载到RAM中）。

- 转换后，可以通过以下方式加载模型和tokenizer：

```python
>>> from transformers import LlamaForCausalLM, CodeLlamaTokenizer

>>> tokenizer = CodeLlamaTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
>>> model = LlamaForCausalLM.from_pretrained("codellama/CodeLlama-7b-hf")
>>> PROMPT = '''def remove_non_ascii(s: str) -> str:
    """ <FILL_ME>
    return result
'''
>>> input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
>>> generated_ids = model.generate(input_ids, max_new_tokens=128)

>>> filling = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens = True)[0]
>>> print(PROMPT.replace("<FILL_ME>", filling))
def remove_non_ascii(s: str) -> str:
    """ Remove non-ASCII characters from a string.

    Args:
        s: The string to remove non-ASCII characters from.

    Returns:
        The string with non-ASCII characters removed.
    """
    result = ""
    for c in s:
        if ord(c) < 128:
            result += c
    return result
```

如果只想要填充的部分：
```python
>>> from transformers import pipeline
>>> import torch

>>> generator = pipeline("text-generation",model="codellama/CodeLlama-7b-hf",torch_dtype=torch.float16, device_map="auto")
>>> generator('def remove_non_ascii(s: str) -> str:\n    """ <FILL_ME>\n    return result', max_new_tokens = 128, return_type = 1)
```

在内部，tokenizer [自动基于"<FILL_ME>"进行拆分](https://huggingface.co/docs/transformers/main/model_doc/code_llama#transformers.CodeLlamaTokenizer.fill_token)，创建符合[原始训练模式](https://github.com/facebookresearch/codellama/blob/cb51c14ec761370ba2e2bc351374a79265d0465e/llama/generation.py#L402)的格式化输入字符串。这比自己准备模式更稳健：它避免了非常难以调试的token glueing等问题。要查看此模型或其他模型所需的CPU和GPU内存量，请尝试[此计算器](https://huggingface.co/spaces/hf-accelerate/model-memory-usage)，它可以帮助确定该值。

- LLaMA分词器是基于[sentencepiece](https://github.com/google/sentencepiece)的BPE模型。sentencepiece的一个怪异之处是，当解码序列时，如果第一个token是单词的开头（例如"Banana"），则分词器不会将前缀空格添加到字符串中。

此模型由[ArthurZucker](https://huggingface.co/ArthurZ)贡献。原作者的原始代码可以在[这里](https://github.com/facebookresearch/llama)找到。


## CodeLlamaTokenizer

[[autodoc]] CodeLlamaTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## CodeLlamaTokenizerFast

[[autodoc]] CodeLlamaTokenizerFast
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - update_post_processor
    - save_vocabulary