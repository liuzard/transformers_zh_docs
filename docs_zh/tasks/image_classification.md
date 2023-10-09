```python
>>> model.compile(optimizer=optimizer, loss=SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
```

Setup callbacks and use the `fit()` method to run the training:

```py
>>> callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)]

>>> model.fit(
...     tf_train_dataset,
...     validation_data=tf_eval_dataset,
...     epochs=num_epochs,
...     callbacks=callbacks,
... )
```

Once training is completed, share your model to the Hub using [`hub` module] so that everyone can use your model:

```py
>>> model.save_pretrained("my_awesome_food_model")
```

</tf>
</frameworkcontent>

Congratulations! You have successfully fine-tuned an image classification model using ViT and evaluated its performance.

```md
>>> loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
>>> model.compile(optimizer=optimizer, loss=loss)
```

要计算从预测中获得的准确率并将您的模型推送到🤗 Hub，请使用[Keras回调](../main_classes/keras_callbacks)。
将您的`compute_metrics`函数传递给[KerasMetricCallback](../main_classes/keras_callbacks#transformers.KerasMetricCallback)，
并使用[PushToHubCallback](../main_classes/keras_callbacks#transformers.PushToHubCallback)上传模型：

```py
>>> from transformers.keras_callbacks import KerasMetricCallback, PushToHubCallback

>>> metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_eval_dataset)
>>> push_to_hub_callback = PushToHubCallback(
...     output_dir="food_classifier",
...     tokenizer=image_processor,
...     save_strategy="no",
... )
>>> callbacks = [metric_callback, push_to_hub_callback]
```

最后，您可以开始训练模型了！使用您的训练和验证数据集、epochs的数量和回调函数调用`fit()`以微调模型：

```py
>>> model.fit(tf_train_dataset, validation_data=tf_eval_dataset, epochs=num_epochs, callbacks=callbacks)
Epoch 1/5
250/250 [==============================] - 313s 1s/step - loss: 2.5623 - val_loss: 1.4161 - accuracy: 0.9290
Epoch 2/5
250/250 [==============================] - 265s 1s/step - loss: 0.9181 - val_loss: 0.6808 - accuracy: 0.9690
Epoch 3/5
250/250 [==============================] - 252s 1s/step - loss: 0.3910 - val_loss: 0.4303 - accuracy: 0.9820
Epoch 4/5
250/250 [==============================] - 251s 1s/step - loss: 0.2028 - val_loss: 0.3191 - accuracy: 0.9900
Epoch 5/5
250/250 [==============================] - 238s 949ms/step - loss: 0.1232 - val_loss: 0.3259 - accuracy: 0.9890
```

恭喜！您已经对模型进行了微调，并在🤗 Hub上分享了它。现在您可以用它进行推理了！
</tf>
</frameworkcontent>


<Tip>

有关如何对图像分类模型进行微调的更详细示例，请参阅相应的[PyTorch笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)。

</Tip>

## 推理

很好，现在您已经对模型进行了微调，可以使用它进行推理了！

加载要进行推理的图像：

```py
>>> ds = load_dataset("food101", split="validation[:10]")
>>> image = ds["image"][0]
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png" alt="image of beignets"/>
</div>

尝试使用[`pipeline`]对微调后的模型进行推理是最简单的方法。使用您的模型实例化一个图像分类的`pipeline`，并将图像传递给它：

```py
>>> from transformers import pipeline

>>> classifier = pipeline("image-classification", model="my_awesome_food_model")
>>> classifier(image)
[{'score': 0.31856709718704224, 'label': 'beignets'},
 {'score': 0.015232225880026817, 'label': 'bruschetta'},
 {'score': 0.01519392803311348, 'label': 'chicken_wings'},
 {'score': 0.013022331520915031, 'label': 'pork_chop'},
 {'score': 0.012728818692266941, 'label': 'prime_rib'}]
```

如果希望，您也可以手动复制`pipeline`的结果：

<frameworkcontent>
<pt>
加载图像处理器对图像进行预处理，并将`input`返回为PyTorch张量：

```py
>>> from transformers import AutoImageProcessor
>>> import torch

>>> image_processor = AutoImageProcessor.from_pretrained("my_awesome_food_model")
>>> inputs = image_processor(image, return_tensors="pt")
```

将输入传递给模型并返回logits：

```py
>>> from transformers import AutoModelForImageClassification

>>> model = AutoModelForImageClassification.from_pretrained("my_awesome_food_model")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
```

使用最高概率的预测标签，并使用模型的`id2label`映射将其转换为标签：

```py
>>> predicted_label = logits.argmax(-1).item()
>>> model.config.id2label[predicted_label]
'beignets'
```
</pt>
</frameworkcontent>

<frameworkcontent>
<tf>
加载图像处理器对图像进行预处理并将`input`返回为TensorFlow张量：

```py
>>> from transformers import AutoImageProcessor

>>> image_processor = AutoImageProcessor.from_pretrained("MariaK/food_classifier")
>>> inputs = image_processor(image, return_tensors="tf")
```

将输入传递给模型并返回logits：

```py
>>> from transformers import TFAutoModelForImageClassification

>>> model = TFAutoModelForImageClassification.from_pretrained("MariaK/food_classifier")
>>> logits = model(**inputs).logits
```

使用最高概率的预测标签，并使用模型的`id2label`映射将其转换为标签：

```py
>>> predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
>>> model.config.id2label[predicted_class_id]
'beignets'
```

</tf>
</frameworkcontent>