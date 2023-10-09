<!--版权 2020 HuggingFace团队 保留所有权利。

根据Apache许可证，第2.0版（“许可证”）许可；除非符合许可证的规定，否则不得使用此文件。
您可以在以下位置获取许可证的副本 http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面协议，否则根据许可证分发的软件基于“按原样”分发的基础上，
没有任何保证或条件，无论是明示的还是暗示的。有关许可下的特定语言的权限和限制，请参阅许可证。

⚠️ 请注意，此文件是Markdown格式，但包含特定的语法，用于我们的文档构建器（类似于MDX），可能在您的Markdown查看器中无法正常显示。

-->

# MarianMT

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=marian">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-marian-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/opus-mt-zh-en">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

**Bugs:** 如果您发现任何异常，请[提交 Github 问题](https://github.com/huggingface/transformers/issues/new?assignees=sshleifer&labels=&template=bug-report.md&title)
并指派 @patrickvonplaten。

翻译应该与模型卡中的测试集输出类似，但不完全相同。

提示：

- 翻译模型的框架使用与 BART 相同的模型。

## 实施注意事项

- 每个模型在磁盘上约为 298 MB，共有 1000 多个模型。
- 支持的语言对列表可以在[此处](https://huggingface.co/Helsinki-NLP)找到。
- 这些模型最初是由[Jörg Tiedemann](https://researchportal.helsinki.fi/en/persons/j%C3%B6rg-tiedemann)使用[Marian](https://marian-nmt.github.io/) C++ 库进行训练和翻译的，该库支持快速训练和翻译。
- 所有模型都是具有 6 层的变压器编码器-解码器。每个模型的性能在模型卡中有记录。
- 不支持需要 BPE 预处理的 80 个 opus 模型。
- 模型代码与[`BartForConditionalGeneration`]相同，只是做了一些微小的修改：
  - 静态（正弦）位置嵌入（`MarianConfig.static_position_embeddings=True`）
  - 无 layernorm_embedding（`MarianConfig.normalize_embedding=False`）
  - 模型以`pad_token_id`（具有 0 作为 token_embedding 的标记）作为前缀开始生成（Bart 使用 `<s/>`），
- 批量转换模型的代码可以在`convert_marian_to_pytorch.py`中找到。
- 此模型由[sshleifer](https://huggingface.co/sshleifer)贡献。

## 命名

- 所有模型名称都使用以下格式：`Helsinki-NLP/opus-mt-{src}-{tgt}`
- 用于命名模型的语言代码是不一致的。两位数字代码通常可以在[此处](https://developers.google.com/admin-sdk/directory/v1/languages)找到，三位数字代码需要搜索“language
  code {code}”。
- 以`es_AR`格式化的代码通常是`code_{region}`。该代码是指来自阿根廷的西班牙语。
- 模型分两个阶段进行转换。前 1000 个模型使用 ISO-639-2 代码标识语言，第二组使用 ISO-639-5 代码和 ISO-639-2 代码相结合。

## 示例

- 由于Marian模型比库中其他许多翻译模型要小，因此它们可以用于微调实验和集成测试。
- [在 GPU 上微调](https://github.com/huggingface/transformers/blob/master/examples/legacy/seq2seq/train_distil_marian_enro.sh)

## 多语言模型

- 所有模型名称都使用以下格式：`Helsinki-NLP/opus-mt-{src}-{tgt}`：
- 如果模型可以输出多种语言，则应通过在`src_text`前面添加所需的输出语言来指定语言代码。
- 您可以在模型卡中查看模型支持的语言代码，如[opus-mt-en-roa](https://huggingface.co/Helsinki-NLP/opus-mt-en-roa)中的目标成分中所示。
- 请注意，如果模型只在源端是多语言的，例如`Helsinki-NLP/opus-mt-roa-en`，则不需要语言代码。

来自[Tatoeba-Challenge 仓库](https://github.com/Helsinki-NLP/Tatoeba-Challenge)的新多语言模型需要 3 个字符的语言代码：

```python
>>> from transformers import MarianMTModel, MarianTokenizer

>>> src_text = [
...     ">>fra<< this is a sentence in english that we want to translate to french",
...     ">>por<< This should go to portuguese",
...     ">>esp<< And this to Spanish",
... ]

>>> model_name = "Helsinki-NLP/opus-mt-en-roa"
>>> tokenizer = MarianTokenizer.from_pretrained(model_name)
>>> print(tokenizer.supported_language_codes)
['>>zlm_Latn<<', '>>mfe<<', '>>hat<<', '>>pap<<', '>>ast<<', '>>cat<<', '>>ind<<', '>>glg<<', '>>wln<<', '>>spa<<', '>>fra<<', '>>ron<<', '>>por<<', '>>ita<<', '>>oci<<', '>>arg<<', '>>min<<']

>>> model = MarianMTModel.from_pretrained(model_name)
>>> translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
>>> [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
["c'est une phrase en anglais que nous voulons traduire en français",
 'Isto deve ir para o português.',
 'Y esto al español']
```

以下是查看枢纽上所有可用预训练模型的代码：

```python
from huggingface_hub import list_models

model_list = list_models()
org = "Helsinki-NLP"
model_ids = [x.modelId for x in model_list if x.modelId.startswith(org)]
suffix = [x.split("/")[1] for x in model_ids]
old_style_multi_models = [f"{org}/{s}" for s in suffix if s != s.lower()]
```

## 旧风格多语言模型

这些是从 OPUS-MT-Train 仓库移植的旧风格多语言模型：以及每个语言组的成员：

```python no-style
['Helsinki-NLP/opus-mt-NORTH_EU-NORTH_EU',
 'Helsinki-NLP/opus-mt-ROMANCE-en',
 'Helsinki-NLP/opus-mt-SCANDINAVIA-SCANDINAVIA',
 'Helsinki-NLP/opus-mt-de-ZH',
 'Helsinki-NLP/opus-mt-en-CELTIC',
 'Helsinki-NLP/opus-mt-en-ROMANCE',
 'Helsinki-NLP/opus-mt-es-NORWAY',
 'Helsinki-NLP/opus-mt-fi-NORWAY',
 'Helsinki-NLP/opus-mt-fi-ZH',
 'Helsinki-NLP/opus-mt-fi_nb_no_nn_ru_sv_en-SAMI',
 'Helsinki-NLP/opus-mt-sv-NORWAY',
 'Helsinki-NLP/opus-mt-sv-ZH']
GROUP_MEMBERS = {
 'ZH': ['cmn', 'cn', 'yue', 'ze_zh', 'zh_cn', 'zh_CN', 'zh_HK', 'zh_tw', 'zh_TW', 'zh_yue', 'zhs', 'zht', 'zh'],
 'ROMANCE': ['fr', 'fr_BE', 'fr_CA', 'fr_FR', 'wa', 'frp', 'oc', 'ca', 'rm', 'lld', 'fur', 'lij', 'lmo', 'es', 'es_AR', 'es_CL', 'es_CO', 'es_CR', 'es_DO', 'es_EC', 'es_ES', 'es_GT', 'es_HN', 'es_MX', 'es_NI', 'es_PA', 'es_PE', 'es_PR', 'es_SV', 'es_UY', 'es_VE', 'pt', 'pt_br', 'pt_BR', 'pt_PT', 'gl', 'lad', 'an', 'mwl', 'it', 'it_IT', 'co', 'nap', 'scn', 'vec', 'sc', 'ro', 'la'],
 'NORTH_EU': ['de', 'nl', 'fy', 'af', 'da', 'fo', 'is', 'no', 'nb', 'nn', 'sv'],
 'SCANDINAVIA': ['da', 'fo', 'is', 'no', 'nb', 'nn', 'sv'],
 'SAMI': ['se', 'sma', 'smj', 'smn', 'sms'],
 'NORWAY': ['nb_NO', 'nb', 'nn_NO', 'nn', 'nog', 'no_nb', 'no'],
 'CELTIC': ['ga', 'cy', 'br', 'gd', 'kw', 'gv']
}
```

将英语翻译成多种罗曼语言的示例，使用的是旧风格的 2 个字符语言代码：

```python
>>> from transformers import MarianMTModel, MarianTokenizer

>>> src_text = [
...     ">>fr<< this is a sentence in english that we want to translate to french",
...     ">>pt<< This should go to portuguese",
...     ">>es<< And this to Spanish",
... ]

>>> model_name = "Helsinki-NLP/opus-mt-en-ROMANCE"
>>> tokenizer = MarianTokenizer.from_pretrained(model_name)

>>> model = MarianMTModel.from_pretrained(model_name)
>>> translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
>>> tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
["c'est une phrase en anglais que nous voulons traduire en français", 
 'Isto deve ir para o português.',
 'Y esto al español']
```

## 文档资源

- [翻译任务指南](../tasks/translation)
- [摘要任务指南](../tasks/summarization)
- [因果语言建模任务指南](../tasks/language_modeling)

## MarianConfig

[[autodoc]] MarianConfig

## MarianTokenizer

[[autodoc]] MarianTokenizer
    - build_inputs_with_special_tokens

## MarianModel

[[autodoc]] MarianModel
    - forward

## MarianMTModel

[[autodoc]] MarianMTModel
    - forward

## MarianForCausalLM

[[autodoc]] MarianForCausalLM
    - forward

## TFMarianModel

[[autodoc]] TFMarianModel
    - call

## TFMarianMTModel

[[autodoc]] TFMarianMTModel
    - call

## FlaxMarianModel

[[autodoc]] FlaxMarianModel
    - __call__

## FlaxMarianMTModel

[[autodoc]] FlaxMarianMTModel
    - __call__