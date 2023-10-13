<!--版权所有2023 HuggingFace团队。保留所有权利。

根据Apache许可证第2.0版（“许可证”）许可；除非符合条件，否则你不得使用此文件。
你可以在以下位置获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是按“原样”分发的，
没有任何明示或暗示的担保或条件。有关特定语言下的权限和限制，详见许可证。

⚠️请注意，此文件采用Markdown格式，但包含特定于我们doc-builder（类似于MDX）的语法，你的Markdown查看器可能无法正确呈现。

-->

# TensorFlow模型的XLA集成

[[open-in-colab]]

加速线性代数（Accelerated Linear Algebra，简称XLA）是一种用于加速TensorFlow模型运行时的编译器。根据[官方文档](https://www.tensorflow.org/xla)：

XLA（加速线性代数）是一种专为线性代数而设计的领域特定编译器，可加速TensorFlow模型，而无需进行源代码更改。

在TensorFlow中使用XLA很简单-它已打包在`tensorflow`库中，并且可以通过任何图创建函数的`jit_compile`参数触发，例如[`tf.function`](https://www.tensorflow.org/guide/intro_to_graphs)。当使用`fit()`和`predict()`等Keras方法时，你只需将`jit_compile`参数传递给`model.compile()`即可启用XLA。但是，XLA并不限于这些方法-它也可用于加速任何任意的`tf.function`。

🤗Transformers中的几种TensorFlow方法已被重写为与XLA兼容，包括[GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2)、[T5](https://huggingface.co/docs/transformers/model_doc/t5)和[OPT](https://huggingface.co/docs/transformers/model_doc/opt)等模型的文本生成，以及[Whisper](https://huggingface.co/docs/transformers/model_doc/whisper)等模型的语音处理。

虽然确切的加速度因模型而异，但对于🤗Transformers中的TensorFlow文本生成模型，我们注意到了大约100倍的加速度。本文档将解释如何使用XLA来实现这些模型的最大性能。如果你对基准测试和我们的XLA集成设计哲学感兴趣，我们还将提供其他资源的链接。

## 使用XLA运行TF函数

让我们考虑以下TensorFlow中的模型：

```py
import tensorflow as tf

model = tf.keras.Sequential(
    [tf.keras.layers.Dense(10, input_shape=(10,), activation="relu"), tf.keras.layers.Dense(5, activation="softmax")]
)
```

上述模型接受维度为`(10,)`的输入。我们可以使用以下方式运行模型的前向传递：

```py
# 为模型生成随机输入。
batch_size = 16
input_vector_dim = 10
random_inputs = tf.random.normal((batch_size, input_vector_dim))

# 执行前向传递。
_ = model(random_inputs)
```

要使用XLA编译函数运行前向传递，需要执行以下操作：

```py
xla_fn = tf.function(model, jit_compile=True)
_ = xla_fn(random_inputs)
```

`model`的默认`call()`函数用于编译XLA图。但是，如果还有其他要编译为XLA的模型函数，可以使用以下代码：

```py
my_xla_fn = tf.function(model.my_xla_fn, jit_compile=True)
```

## 使用🤗Transformers中的XLA运行TF文本生成模型

要在🤗Transformers中启用XLA加速的生成功能，你需要安装最新版本的`transformers`。可通过运行以下命令来安装：

```bash
pip install transformers --upgrade
```

然后，你可以运行以下代码：

```py
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM

# 如果未安装Transformers的最低版本，则会出错。
from transformers.utils import check_min_version

check_min_version("4.21.0")


tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left", pad_token="</s>")
model = TFAutoModelForCausalLM.from_pretrained("gpt2")
input_string = ["TensorFlow is"]

# 一行代码创建了一个XLA生成函数
xla_generate = tf.function(model.generate, jit_compile=True)

tokenized_input = tokenizer(input_string, return_tensors="tf")
generated_tokens = xla_generate(**tokenized_input, num_beams=2)

decoded_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
print(f"Generated -- {decoded_text}")
# Generated -- TensorFlow is an open-source, open-source, distributed-source application # framework for the
```

正如你注意到的那样，在`generate()`上启用XLA只需一行代码。其余的代码保持不变。但是，上述代码片段中也有一些特定于XLA的要注意的问题。你需要了解这些问题以实现XLA带来的加速。我们在下一节中讨论这些问题。

## 注意事项

当你首次执行启用了XLA的函数（例如上面的`xla_generate()`）时，它将在内部尝试推断计算图，这是一个耗时的过程。这个过程称为“追踪”。

你可能会注意到生成时间不快。给`xla_generate()`（或任何其他启用XLA的函数）连续调用不需要推断计算图，前提是函数的输入与最初构建计算图时的形状相同。虽然对于具有固定输入形状（例如图像）的模态性来说这不是问题，但如果你正在处理变量输入形状（例如文本），则需要注意。

为了确保`xla_generate()`始终使用相同的输入形状进行操作，可以在调用tokenizer时指定`padding`参数。

```py
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left", pad_token="</s>")
model = TFAutoModelForCausalLM.from_pretrained("gpt2")
input_string = ["TensorFlow is"]

xla_generate = tf.function(model.generate, jit_compile=True)

# 这里，我们使用填充选项调用tokenizer。
tokenized_input = tokenizer(input_string, pad_to_multiple_of=8, padding=True, return_tensors="tf")

generated_tokens = xla_generate(**tokenized_input, num_beams=2)
decoded_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
print(f"Generated -- {decoded_text}")
```

这样，你可以确保`xla_generate()`始终接收与其追踪时相同形状的输入，并导致生成时间的加速。你可以使用以下代码验证这一点：

```py
import time
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left", pad_token="</s>")
model = TFAutoModelForCausalLM.from_pretrained("gpt2")

xla_generate = tf.function(model.generate, jit_compile=True)

for input_string in ["TensorFlow is", "TensorFlow is a", "TFLite is a"]:
    tokenized_input = tokenizer(input_string, pad_to_multiple_of=8, padding=True, return_tensors="tf")
    start = time.time_ns()
    generated_tokens = xla_generate(**tokenized_input, num_beams=2)
    end = time.time_ns()
    print(f"Execution time -- {(end - start) / 1e6:.1f} ms\n")
```

在Tesla T4 GPU上，你可以期望得到如下输出：

```bash
Execution time -- 30819.6 ms

Execution time -- 79.0 ms

Execution time -- 78.9 ms
```

`xla_generate()`的第一次调用由于追踪而耗时，但是后续调用的速度快了几个数量级。请记住，任何时候更改生成选项都将触发重新追踪，从而导致生成时间变慢。

我们在本文档中并未涵盖🤗Transformers在文本生成方面提供的所有选项。我们鼓励你阅读文档以获取更多高级用例。

## 其他资源

在这里，我们为你提供一些其他资源，如果你想深入了解🤗Transformers中的XLA以及其他方面：

* [此Colab笔记本](https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/91_tf_xla_generate.ipynb)提供了一个交互式演示，供你尝试使用与XLA兼容的编码器-解码器（例如[T5](https://huggingface.co/docs/transformers/model_doc/t5)）和仅解码器（例如[GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2)）文本生成模型。
* [此博客文章](https://huggingface.co/blog/tf-xla-generate)提供了与XLA兼容模型的比较基准的概述，以及对TensorFlow中XLA的友好介绍。
* [此博客文章](https://blog.tensorflow.org/2022/11/how-hugging-face-improved-text-generation-performance-with-xla.html)讨论了我们在🤗Transformers中添加对TensorFlow模型的XLA支持的设计哲学。
* 了解有关XLA和TensorFlow图的更多信息的推荐帖子：
    * [XLA：用于机器学习的优化编译器](https://www.tensorflow.org/xla)
    * [图表和tf.function简介](https://www.tensorflow.org/guide/intro_to_graphs)
    * [使用tf.function获得更好的性能](https://www.tensorflow.org/guide/function) 