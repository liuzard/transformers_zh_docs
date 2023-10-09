import tensorflow as tf

tokenizer = AutoTokenizer.from_pretrained("my_awesome_opus_books_model")
inputs = tokenizer(text, return_tensors="tf").input_ids
```

Use the [`~transformers.generation_utils.GenerationMixin.generate`] method to create the translation. For more details about the different text generation strategies and

```py
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("my_awesome_opus_books_model")
outputs = model.generate(inputs, max_length=40, do_sample=True, top_k=30, top_p=0.95)
```

Decode the generated token ids back into text:

```py
tokenizer.decode(outputs[0], skip_special_tokens=True)
```
'Legumes partagent des ressources avec les bactéries fixatrices d'azote.'

将下面这句话翻译成中文，格式是markdown，<>里面的保留原文，也不要添加额外的内容：
```plaintext
>>> tokenizer = AutoTokenizer.from_pretrained("my_awesome_opus_books_model")
>>> inputs = tokenizer(text, return_tensors="tf").input_ids
```
使用[`~transformers.generation_tf_utils.TFGenerationMixin.generate`]方法创建翻译。有关不同文本生成策略和控制生成的参数的更多详细信息，请查看[Text Generation](../main_classes/text_generation) API。

```py
>>> from transformers import TFAutoModelForSeq2SeqLM

>>> model = TFAutoModelForSeq2SeqLM.from_pretrained("my_awesome_opus_books_model")
>>> outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
```

将生成的令牌id解码回文本：

```py
>>> tokenizer.decode(outputs[0], skip_special_tokens=True)
'Les lugumes partagent les ressources avec des bactéries fixatrices d'azote.'
```
</tf>
</frameworkcontent>