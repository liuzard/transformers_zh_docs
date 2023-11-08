
<!--版权2022 HuggingFace团队。保留所有权利。

根据Apache许可证第2版（“许可证”）许可您只能在符合许可证的情况下使用此文件。您可以在以下网址获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件基于“原样” BASIS ，WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️需要注意的是，此文件以Markdown的格式编写，但包含了特定的标记语法以供doc-builder（类似于MDX）使用，这些语法可能在您的Markdown查看器中无法正确渲染。-->

# 摘要

[[在Colab中查看代码]]


<Youtube id="yHnr5Dk2zCI"/>

摘要生成一个较短的文档或文章，该摘要捕捉到所有重要信息。与翻译一样，摘要是可以被定义成序列-序列任务的另一个例子。摘要可以是：

- 抽取式的：从文档中抽取最相关的信息。
- 创造性的：生成新的文本，抓住最相关的信息。

本指南将告诉您如何：

1. 对加利福尼亚州法案子集上的[T5](https://huggingface.co/t5-small)进行微调，以进行创造性摘要。
2. 使用您微调的模型进行推断。


>此教程中显示的任务由以下模型架构支持：
>[BART](../model_doc/bart), [BigBird-Pegasus](../model_doc/bigbird_pegasus), [Blenderbot](../model_doc/blenderbot), [BlenderbotSmall](../model_doc/blenderbot-small), [Encoder decoder](../model_doc/encoder-decoder), [FairSeq Machine-Translation](../model_doc/fsmt), [GPTSAN-japanese](../model_doc/gptsan-japanese), [LED](../model_doc/led), [LongT5](../model_doc/longt5), [M2M100](../model_doc/m2m_100), [Marian](../model_doc/marian), [mBART](../model_doc/mbart), [MT5](../model_doc/mt5), [MVP](../model_doc/mvp), [NLLB](../model_doc/nllb), [NLLB-MOE](../model_doc/nllb-moe), [Pegasus](../model_doc/pegasus), [PEGASUS-X](../model_doc/pegasus_x), [PLBart](../model_doc/plbart), [ProphetNet](../model_doc/prophetnet), [SwitchTransformers](../model_doc/switch_transformers), [T5](../model_doc/t5), [UMT5](../model_doc/umt5), [XLM-ProphetNet](../model_doc/xlm-prophetnet)


在开始之前，请确保已经安装了所有必要的库文件：

```bash
pip install transformers datasets evaluate rouge_score
```

我们鼓励您登录您的Hugging Face账户，这样您就可以与社区上传和共享模型。当提示时，请输入您的令牌进行登录：

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## 加载BillSum数据集

首先从🤗数据集库中加载较小的加利福尼亚州法案子集的BillSum数据集：

```py
>>> from datasets import load_dataset

>>> billsum = load_dataset("billsum", split="ca_test")
```

使用[`~datasets.Dataset.train_test_split`]方法将数据集分割成训练集和测试集：

```py
>>> billsum = billsum.train_test_split(test_size=0.2)
```

然后查看一个示例：

```py
>>> billsum["train"][0]
{'summary': 'Existing law authorizes state agencies to enter into contracts for the acquisition of goods or services upon approval by the Department of General Services. Existing law sets forth various requirements and prohibitions for those contracts, including, but not limited to, a prohibition on entering into contracts for the acquisition of goods or services of $100,000 or more with a contractor that discriminates between spouses and domestic partners or same-sex and different-sex couples in the provision of benefits. Existing law provides that a contract entered into in violation of those requirements and prohibitions is void and authorizes the state or any person acting on behalf of the state to bring a civil action seeking a determination that a contract is in violation and therefore void. Under existing law, a willful violation of those requirements and prohibitions is a misdemeanor.\nThis bill would also prohibit a state agency from entering into contracts for the acquisition of goods or services of $100,000 or more with a contractor that discriminates between employees on the basis of gender identity in the provision of benefits, as specified. By expanding the scope of a crime, this bill would impose a state-mandated local program.\nThe California Constitution requires the state to reimburse local agencies and school districts for certain costs mandated by the state. Statutory provisions establish procedures for making that reimbursement.\nThis bill would provide that no reimbursement is required by this act for a specified reason.',

## 翻译 private_upload/default/2023-11-07-19-45-48/summarization.md.part-1.md

'text': '加利福尼亚州的人民以如下方式行事：\n\n\n第1节\n第10295.35节被添加到《公共合同法典》中，如下所读：\n10295.35。\n（a）（1）不顾任何其他法律的规定，国家机构不得以一百万元（$100000）或更多的金额与承包商订立任何购买货物或服务的合同，该承包商在提供福利方面对雇员根据雇员的或依赖的实际或被视为的性别身份进行歧视，包括但不限于将雇员或依赖人员的身份定为跨性别。\n（2）为了本节的目的，“合同”包括每年每个承包商累计一百万元（$100000）或更多的金额的合同。\n（3）为了本节的目的，如果计划不符合卫生和安全法典第1365.5节和保险法典第10140节的规定，则雇员的健康计划是歧视性的。\n（4）本节的要求仅适用于发生在承包商经营的以下情况：\n（A）国内。\n（B）在国内以外的房地产上，如果该房地产归国有，或者如果国家有权占用该房地产，并且如果承包商在该位置的存在与国家的合同有关。\n（C）在美国其他地方，正在进行与国家合同有关的工作。\n（b）承包商应按现行法律或承包商的保险提供者的要求，最大程度地保密雇员或求职人员对就业福利的请求或对资格证明提交的任何文件。\n（c）在国家机构确定的所有合理措施都已采取的情况下，本节的要求可以在以下任何情况下豁免：\n（1）只有一个准备与国家机构订立特定合同的潜在承包商。\n（2）合同是为了应对国家机构认定的危机，该危机危及公共健康、福利或安全，或者该合同是为了提供基本服务而必要的，且没有符合本节要求的实体能够立即提供危机响应。\n（3）本节的要求违反或与授权适用于本节的任何授权、津贴或协议的条款或条件不一致，如果代理机构已经尽力改变任何授权、津贴或协议的条款或条件以授权本节的适用。\n（4）该承包商提供批发或大宗水、电力或天然气，以及与之有关的输送或传输，或附属服务，作为根据良好的公用实践确保可靠服务所需的，如果无法通过标准竞争招标程序实现对同样需购买的，则该承包商不提供直接零售服务给最终用户。\n（d）（1）如果承包商在提供福利时支付获得福利所发生的实际费用，则该承包商在提供福利方面不被视为歧视。\n（2）如果承包商无法提供某项福利，尽管采取合理措施使其能够提供，该承包商在提供福利方面不被视为歧视。\n（e）（1）本章适用的每份合同应包含一个声明，承包商在该声明中证明承包商符合本节的规定。\n（2）该部门或其他承包机构应根据其现有的执法权力来执行本节。\n（3）（A）如果承包商虚假证明其符合本节的规定，那么与该承包商的合同应受到《现行法典》第10420节开始的第9篇的约束，除非在部门或其他承包机构指定的时间内，该承包商向该部门或机构提供符合或正在符合本节的证明。\n（B）将《现行法典》第10420节开始的第9篇的补救措施或处罚适用于本章适用的合同不会排斥部门或其他承包机构在其现有的执法权力下的任何现有补救措施。\n(f)本节的任何规定都不意味着调整任何地方行政区的承包惯例。\n（g）本节应解释为不与适用的联邦法律、规则或法规冲突。如果有据可信的法院或具有管辖权的机构认为联邦法律、规则或法规使本代码的任何条款、句子、段落或节无效，或者使其适用于任何人或情况无效，那么州意图是法院或机构撤销该条款、句子、段落或节，以使本节的其余部分继续有效。\n第2节\n《公共合同法典》第10295.35节不得解释为在第44REC#的97号中创建任何新的执法权限或责任的部门总务部或任何其他承包机构。\n\n\n第三节\n根据加利福尼亚州宪法第13 XXB条第6节，该法律不需要根据第93CHAR#7Q3.9号第17556条对任何地方机构或学区进行补偿，因为该法律可能由于创建新的犯罪或副相之外的犯罪、副相的涉及，第93CHAR#7Q3.9号第17556条的罚金变化，或者根据加利福尼亚州宪法第13X字条第6节对犯罪的定义进行更改而产生的费用。',
 'title': '添加第10295.35节至公共合同法典，有关公共合同。'}

## 翻译 private_upload/default/2023-11-07-19-45-48/summarization.md.part-2.md

要应用预处理函数到整个数据集，可以使用🤗 Datasets [`~datasets.Dataset.map`] 方法。通过设置 `batched=True` 来加速 `map` 函数，以一次处理数据集中的多个元素：

```py
>>> tokenized_billsum = billsum.map(preprocess_function, batched=True)
```

现在使用 [`DataCollatorForSeq2Seq`] 创建一个批次的示例。在汇编期间，通过动态填充句子到批次中的最大长度，而不是填充整个数据集到最大长度，可以提高效率。

**1、pytorch代码**

```py
>>> from transformers import DataCollatorForSeq2Seq

>>> data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)
```

**2、tensorflow 代码**

```py
>>> from transformers import DataCollatorForSeq2Seq

>>> data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint, return_tensors="tf")
```


## 评估

在训练过程中，添加度量方法通常有助于评估模型的性能。您可以通过🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) 库快速加载一个评估方法。对于此任务，加载 [ROUGE](https://huggingface.co/spaces/evaluate-metric/rouge) 度量（参阅🤗 评估库的[快速导览](https://huggingface.co/docs/evaluate/a_quick_tour)以了解如何加载和计算度量）：

```py
>>> import evaluate

>>> rouge = evaluate.load("rouge")
```

然后创建一个函数，将您的预测和标签传递给 [`~evaluate.EvaluationModule.compute`] 来计算 ROUGE 度量：

```py
>>> import numpy as np


>>> def compute_metrics(eval_pred):
...     predictions, labels = eval_pred
...     decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
...     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
...     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

...     result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

...     prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
...     result["gen_len"] = np.mean(prediction_lens)

...     return {k: round(v, 4) for k, v in result.items()}
```

现在您的 `compute_metrics` 函数已准备就绪，当您设置训练时将返回到它。

## 训练

**1、pytorch 代码**


>如果您对使用 [`Trainer`] 对模型进行微调不熟悉，请查看基本教程[这里](../training#train-with-pytorch-trainer)！


现在可以开始训练您的模型了！使用 [`AutoModelForSeq2SeqLM`] 加载 T5：

```py
>>> from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

>>> model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
```

目前只剩下三个步骤：

1. 在 [`Seq2SeqTrainingArguments`] 中定义您的训练超参数。唯一所需的参数是 `output_dir`，指定保存模型的位置。通过设置 `push_to_hub=True` 将此模型推送到 Hub（您需要登录 Hugging Face 才能上传模型）。在每个 epoch 结束时，[`Trainer`] 将评估 ROUGE 度量并保存训练检查点。
2. 将训练参数与模型、数据集、分词器、数据汇集器和 `compute_metrics` 函数一起传递给 [`Seq2SeqTrainer`]。
3. 调用 [`~Trainer.train`] 对模型进行微调。

```py
>>> training_args = Seq2SeqTrainingArguments(
...     output_dir="my_awesome_billsum_model",
...     evaluation_strategy="epoch",
...     learning_rate=2e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_size=16,
...     weight_decay=0.01,
...     save_total_limit=3,
...     num_train_epochs=4,
...     predict_with_generate=True,
...     fp16=True,
...     push_to_hub=True,
... )

>>> trainer = Seq2SeqTrainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_billsum["train"],
...     eval_dataset=tokenized_billsum["test"],
...     tokenizer=tokenizer,
...     data_collator=data_collator,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

训练完成后，使用 [`~transformers.Trainer.push_to_hub`] 方法将模型共享到 Hub，以便每个人都可以使用您的模型：

```py
>>> trainer.push_to_hub()
```

**2、tensorflow代码**



>如果您对使用 Keras 对模型进行微调不熟悉，请查看基本教程[这里](../training#train-a-tensorflow-model-with-keras)！


要在 TensorFlow 中微调模型，请首先设置优化器函数、学习率计划和一些训练超参数：

```py
>>> from transformers import create_optimizer, AdamWeightDecay

>>> optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
```

然后，使用 [`TFAutoModelForSeq2SeqLM`] 加载 T5：

```py
>>> from transformers import TFAutoModelForSeq2SeqLM

>>> model = TFAutoModelForSeq2SeqLM.from_pretrained(checkpoint)
```

使用 [`~transformers.TFPreTrainedModel.prepare_tf_dataset`] 将数据集转换为 `tf.data.Dataset` 格式：

```py
>>> tf_train_set = model.prepare_tf_dataset(
...     tokenized_billsum["train"],
...     shuffle=True,
...     batch_size=16,
...     collate_fn=data_collator,
... )

>>> tf_test_set = model.prepare_tf_dataset(
...     tokenized_billsum["test"],
...     shuffle=False,
...     batch_size=16,
...     collate_fn=data_collator,
... )
```

使用 [`compile`](https://keras.io/api/models/model_training_apis/#compile-method) 配置用于训练的模型。请注意，Transformers 模型都有一个默认的任务相关损失函数，所以您不需要指定一个，除非您想要使用自定义的：

```py
>>> import tensorflow as tf

## 翻译 private_upload/default/2023-11-07-19-45-48/summarization.md.part-3.md

```markdown
>>> model.compile(optimizer=optimizer)  # 没有损失参数！
```

开始训练之前，最后要做的两件事情是从预测中计算 ROUGE 分数，并提供一种将模型推送到 Hub 的方法。这两个可以通过使用 [Keras 回调](../main_classes/keras_callbacks) 来完成。

将你的 `compute_metrics` 函数传递给 [`~transformers.KerasMetricCallback`]：

```py
>>> from transformers.keras_callbacks import KerasMetricCallback

>>> metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)
```

指定将你的模型和 tokenizer 推送到的位置，用 [`~transformers.PushToHubCallback`]：

```py
>>> from transformers.keras_callbacks import PushToHubCallback

>>> push_to_hub_callback = PushToHubCallback(
...     output_dir="my_awesome_billsum_model",
...     tokenizer=tokenizer,
... )
```

然后将你的回调函数捆绑在一起：

```py
>>> callbacks = [metric_callback, push_to_hub_callback]
```

最后，你可以开始训练模型了！通过调用 [`fit`](https://keras.io/api/models/model_training_apis/#fit-method)，传递你的训练集、验证集、训练轮数和回调函数来微调模型：

```py
>>> model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=3, callbacks=callbacks)
```

训练完成后，你的模型将自动上传到 Hub，大家都可以使用！



>更详细的摘要微调模型示例，请查看相应的
[PyTorch notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/summarization.ipynb)
或 [TensorFlow notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/summarization-tf.ipynb)。

## 推理

很好，现在你已经微调了一个模型，可以用它做推理了！

准备一些你想要进行摘要的文本。对于 T5，你需要根据你正在处理的任务为输入加上前缀。对于摘要，你应该如下所示为输入加上前缀：

```py
>>> text = "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."
```

尝试使用微调后的模型进行推理的最简单方法是在 [`pipeline`] 中使用它。使用你的模型实例化一个摘要的 `pipeline`，并将文本传递给它：

```py
>>> from transformers import pipeline

>>> summarizer = pipeline("summarization", model="stevhliu/my_awesome_billsum_model")
>>> summarizer(text)
[{"summary_text": "The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country."}]
```

如果你愿意，你也可以手动复制 `pipeline` 的结果：

**1、pytorch代码**

将文本进行分词，将 `input_ids` 返回为 PyTorch 张量：

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_billsum_model")
>>> inputs = tokenizer(text, return_tensors="pt").input_ids
```

使用 [`~transformers.generation_utils.GenerationMixin.generate`] 方法进行摘要。有关不同的文本生成策略和用于控制生成的参数的更多详细信息，请查看 [文本生成](../main_classes/text_generation) API。

```py
>>> from transformers import AutoModelForSeq2SeqLM

>>> model = AutoModelForSeq2SeqLM.from_pretrained("stevhliu/my_awesome_billsum_model")
>>> outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)
```

将生成的标记 ID 解码回文本：

```py
>>> tokenizer.decode(outputs[0], skip_special_tokens=True)
'the inflation reduction act lowers prescription drug costs, health care costs, and energy costs. it's the most aggressive action on tackling the climate crisis in american history. it will ask the ultra-wealthy and corporations to pay their fair share.'
```

**2、tensorflow代码**

将文本进行分词，将 `input_ids` 返回为 TensorFlow 张量：

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_billsum_model")
>>> inputs = tokenizer(text, return_tensors="tf").input_ids
```

使用 [`~transformers.generation_tf_utils.TFGenerationMixin.generate`] 方法进行摘要。有关不同的文本生成策略和用于控制生成的参数的更多详细信息，请查看 [文本生成](../main_classes/text_generation) API。

```py
>>> from transformers import TFAutoModelForSeq2SeqLM

>>> model = TFAutoModelForSeq2SeqLM.from_pretrained("stevhliu/my_awesome_billsum_model")
>>> outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)
```

将生成的标记 ID 解码回文本：

```py
>>> tokenizer.decode(outputs[0], skip_special_tokens=True)
'the inflation reduction act lowers prescription drug costs, health care costs, and energy costs. it's the most aggressive action on tackling the climate crisis in american history. it will ask the ultra-wealthy and corporations to pay their fair share.'
```


