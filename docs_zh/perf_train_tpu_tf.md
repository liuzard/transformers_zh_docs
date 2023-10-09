<!--版权2023年 HuggingFace团队。保留所有权利。

特此授权，根据Apache许可证第2.0版（“许可证”），除非符合
许可证。您可以在以下位置获取许可证的副本。

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则依据许可证分发的软件是以
“按原样”分发的，不附带任何明示或暗示的担保或条件。有关许可证的详细信息，请参阅

⚠️请注意，此文件是Markdown格式的，但包含我们doc-builder的特定语法（类似于MDX），这些语法可能无法
正确在您的Markdown查看器中渲染。

-->

# 使用TensorFlow在TPU上进行训练

<Tip>

如果您不需要长篇的解释，只想通过TPU代码示例来入门，请查看[我们的TPU示例笔记本！](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/tpu_training-tf.ipynb)

</Tip>

### 什么是TPU?

TPU是**Tensor处理单元**。它们是由Google设计的硬件，用于在神经网络中大大加速张量计算，类似于GPU。它们可用于网络训练和推理。通常通过Google的云服务访问它们，但也可以直接通过Google Colab和Kaggle Kernels免费访问小型TPU。

因为[🤗Transformers中的所有TensorFlow模型都是Keras模型](https://huggingface.co/blog/tensorflow-philosophy)，所以本文档中的大多数方法通常适用于任何Keras模型的TPU训练！然而，在Transformers和Datasets的HuggingFace生态系统（Hug-o-System?）中，还有一些特定的点，我们在接触它们时将确保提醒您。

### 有哪些可用的TPU类型？

新用户通常对各种TPU和访问方式感到非常困惑。首先要理解的关键区别是**TPU Node**和**TPU VM**之间的区别。

使用**TPU Node**实际上是间接访问远程TPU。您需要一个单独的虚拟机，该虚拟机将初始化您的网络和数据流水线，然后将其转发到远程节点。当您在Google Colab中使用TPU时，您正在以**TPU Node**样式访问它。

对于不熟悉TPU节点的人，使用TPU节点可能会有一些意想不到的行为！特别是，因为TPU位于与运行Python代码的机器物理上不同的系统上，因此您的数据不能在本地机器上找到 - 任何从本机内部存储加载的数据流水线都将完全失败！相反，数据必须存储在Google Cloud Storage中，以便您的数据流水线仍然可以访问它，即使在远程TPU节点上运行时也是如此。

<Tip>

如果您可以将所有数据存储为`np.ndarray`或`tf.Tensor`，则可以在使用Colab或TPU节点时对该数据进行`fit()`，而无需将其上传到Google Cloud Storage。

</Tip>

<Tip>

**🤗具体的Hugging Face提示🤗：**我们已经花了很多精力将我们的TensorFlow模型和损失函数重新编写为XLA兼容。我们的模型和损失函数通常默认遵守规则1和2，因此如果您使用`transformers`模型，可以跳过它们。但是，当编写自己的模型和损失函数时，请不要忘记这些规则！

</Tip>

第二种访问TPU的方式是通过**TPU VM**。使用TPU VM时，您直接连接到TPU所连接的机器，就像在GPU VM上进行训练一样。从数据流水线的角度来看，TPU VM通常更易于使用。以上所有警告都不适用于TPU VM！

这是一个有偏见的文档，所以这是我们的观点：**如果可能，请避免使用TPU Node。**它比TPU VM更令人困惑，更难调试。它在未来也可能不受支持 - Google的最新TPU，TPUv4，只能作为TPU VM访问，这表明TPU节点越来越成为一种“传统”的访问方法。但是，我们知道目前唯一免费的TPU访问是在Colab和Kaggle Kernels上，它们使用的是TPU节点 - 因此，如果必须使用，我们将尝试解释如何处理它！请查看[TPU示例笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/tpu_training-tf.ipynb)，以获取更详细的代码示例。

### 有哪些可用的TPU尺寸？

单个TPU（v2-8/v3-8/v4-8）可以运行8个副本。TPU存在可以同时运行数百或数千个副本的**pod**。当您使用多个TPU但小于整个pod（例如，v3-32）时，您的TPU群称为**pod切片**。

当您通过Colab访问免费的TPU时，通常会获得一个单独的v2-8 TPU。

### 我一直听说这个XLA是什么。什么是XLA，它与TPU有什么关系？

XLA是一个优化编译器，被TensorFlow和JAX两者都使用。在JAX中，它是唯一的编译器，而在TensorFlow中，它是可选的（但在TPU上是强制的！）。在训练Keras模型时，启用XLA的最简单方法是将`jit_compile=True`传递给`model.compile()`的参数。如果您没有遇到任何错误，性能也很好，那就是您准备好使用TPU的最好迹象！

在TPU上进行调试通常比在CPU / GPU上更困难，因此我们建议在尝试在TPU上运行代码之前，先在CPU / GPU上运行代码并使用XLA。当然，您不必进行长时间训练 - 只需进行几个步骤，以确保您的模型和数据流水线按照您的期望工作即可。

<Tip>

XLA编译后的代码通常更快 - 因此，即使您不打算在TPU上运行，添加`jit_compile=True`也可以提高性能。但请注意下面关于XLA兼容性的注意事项！

</Tip>

<Tip warning={true}>

**通过痛苦的经验产生的技巧：**尽管使用`jit_compile=True`是提速的好方法，并且可以测试您的CPU / GPU代码是否与XLA兼容，但如果您在实际在TPU上训练代码时将其保留下来，它实际上可能会导致很多问题。 XLA编译将在TPU上隐式发生，所以在实际在TPU上运行代码之前，请记住删除该行！

</Tip>

### 如何使我的模型与XLA兼容？

在许多情况下，您的代码可能已经兼容XLA！但是，有一些在普通TensorFlow中有效的东西在XLA中无法工作。我们将它们总结为以下三条核心规则：

<Tip>

**🤗具体的HuggingFace提示🤗：**我们已经在将我们的TensorFlow模型和损失函数重新编写为XLA兼容方面投入了很多工作。我们的模型和损失函数通常默认遵守规则#1和#2，因此如果您使用`transformers`模型，可以跳过它们。但是，在编写自己的模型和损失函数时，请不要忘记这些规则！

</Tip>

#### XLA规则#1：您的代码不能具有“数据相关条件”

这意味着任何`if`语句不能依赖于`tf.Tensor`内的值。例如，此代码块无法使用XLA编译！

```python
if tf.reduce_sum(tensor) > 10:
    tensor = tensor / 2.0
```

这一初始时可能看起来非常受限，但大多数神经网络代码不需要执行此操作。您通常可以使用`tf.cond`避免此限制（请参阅此处的文档[here](https://www.tensorflow.org/api_docs/python/tf/cond)），或者删除条件并找到使用指示变量的巧妙数学技巧，例如：

```python
sum_over_10 = tf.cast(tf.reduce_sum(tensor) > 10, tf.float32)
tensor = tensor / (1.0 + sum_over_10)
```

此代码与上面的代码完全相同，但通过避免条件语句，我们确保它可以在没有问题的情况下进行XLA编译！

#### XLA规则#2：您的代码不能具有“数据相关形状”

这意味着您的代码中所有`tf.Tensor`对象的形状不能取决于它们的值。例如，函数`tf.unique`不能使用XLA编译，因为它返回一个包含输入中每个唯一值的张量。该输出的形状显然取决于输入张量的重复性，因此XLA拒绝处理它！

通常情况下，大多数神经网络代码默认遵守规则#2。但是，有一些常见情况会导致问题。一个非常常见的案例是使用**标签遮罩**，将标签设置为负值以指示在计算损失时应忽略这些位置。如果查看支持标签遮罩的NumPy或PyTorch损失函数，通常会看到使用[布尔索引](https://numpy.org/doc/stable/user/basics.indexing.html#boolean-array-indexing)的代码，例如：

```python
label_mask = labels >= 0
masked_outputs = outputs[label_mask]
masked_labels = labels[label_mask]
loss = compute_loss(masked_outputs, masked_labels)
mean_loss = torch.mean(loss)
```

这段代码在NumPy或PyTorch中完全正常，但在XLA中会出错！为什么？因为`masked_outputs`和`masked_labels`的形状取决于遮罩了多少位置 - 这使其成为**数据相关形状**。但是，正如对于规则#1一样，我们通常可以通过计算每个位置的损失，但在计算平均值时将它们中的遮罩位置归零，从而避免数据相关形状。

```python
label_mask = tf.cast(labels >= 0, tf.float32)
loss = compute_loss(outputs, labels)
loss = loss * label_mask  # 将负标签位置设置为0
mean_loss = tf.reduce_sum(loss) / tf.reduce_sum(label_mask)
```

在这里，我们通过计算每个位置的损失，但在计算平均值时，通过将分子和分母中的遮罩位置归零，避免了数据相关形状，从而得到与第一块代码完全相同的结果，并同时保持了XLA兼容性。请注意，我们使用了与规则#1中相同的技巧 - 将`tf.bool`转换为`tf.float32`并将其用作指示变量。这是一个非常有用的技巧，所以如果您需要将自己的代码转换为XLA，请记住它！

#### XLA规则#3：XLA将需要为每个不同的输入形状重新编译您的模型

这是最重要的规则。这意味着如果您的输入形状非常变化，XLA将不得不一遍又一遍地重新编译您的模型，这将导致严重的性能问题。这在NLP模型中很常见，因为输入文本在标记化后具有可变长度。在其他模态中，静态形状更常见，这个规则就不是那么严重的问题。

如何解决规则#3？关键是使用**填充** - 如果将所有输入都填充到相同的长度，然后使用`attention_mask`，就可以获得与可变形状相同的结果，但没有任何XLA问题。然而，过多的填充也会导致严重的减速 - 如果将所有样本都填充到整个数据集的最大长度，可能会遇到批次由无尽的填充令牌组成的问题，这将浪费大量计算和内存！

解决此问题并没有完美的解决方案。但是，您可以尝试一些技巧。一个非常有用的技巧是**将样本的批次填充到32或64个标记的倍数。**这通常只会稍微增加标记的数量，但会大大减少独特输入形状的数量，因为现在每个输入形状都必须是32或64的倍数。较少的独特输入形状意味着较少的XLA编译！

<Tip>

**🤗具体的HuggingFace提示🤗:**我们的令牌化器和数据整理器在这方面可以帮助您。在调用分词器时，您可以使用`padding="max_length"`或`padding="longest"`来获取填充数据。我们的令牌化器和数据整理器还具有`pad_to_multiple_of`参数，您可以使用它来减少您看到的独特输入形状的数量！

</Tip>

### 如何在TPU上训练我的模型？

一旦您的训练在XLA兼容且（如果您正在使用TPU Node / Colab）适当准备好您的数据集后，运行在TPU上实际上非常简单！您需要改变代码中的一些行，只需添加几行代码来初始化您的TPU，并确保您的模型和数据集是在`TPUStrategy`范围内创建的。请查看[我们的TPU示例笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/tpu_training-tf.ipynb)，了解更详细的示例！

### 总结

本文中有很多内容，因此让我们通过一个快速检查表来总结，以便您在准备将模型训练适配到TPU时可以使用：

- 确保您的代码遵循XLA的三条规则
- 使用`jit_compile=True`在CPU/GPU上编译您的模型，并确认您可以使用XLA训练它
- 将您的数据集加载到内存中或使用与TPU兼容的数据集加载方法（请参见[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/tpu_training-tf.ipynb)）
- 将您的代码迁移到Colab（使用加速器设置为“TPU”）或Google Cloud上的TPU VM
- 添加TPU初始化代码（请参见[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/tpu_training-tf.ipynb)）
- 创建`TPUStrategy`并确保数据集加载和模型创建在`strategy.scope()`内（请参见[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/tpu_training-tf.ipynb)）
- 不要忘记在切换到TPU时将`jit_compile=True`删除！
- 🙏🙏🙏🥺🥺🥺
- 调用model.fit()
- 你做到了！