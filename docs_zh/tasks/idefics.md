<!--
版权所有2023年HuggingFace团队。 版权所有。

根据Apache许可证第2.0版（“许可证”）进行许可； 除非符合许可证的要求，否则你无法使用此文件。
你可以在以下网址获得许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，按原样分发的软件根据许可证分发
“AS IS” BASIS，无论是明示的还是暗示的，不对任何形式的担保或条件负责。
有关许可的特定语言和限制的限制，请参阅许可。

注意，此文件是Markdown格式，但包含我们doc-builder（类似于MDX）的特定语法，这些语法可能无法在你的Markdown查看器中正确显示。

-->

# 使用IDEFICS进行图像任务

[[open-in-colab]]

虽然可以通过优化专门的模型来解决单个任务，但最近出现并日益受到欢迎的另一种方法是使用大型模型处理多样化的任务而无需优化。 例如，大型语言模型可以处理诸如摘要、翻译、分类等NLP任务。这种方法不再局限于文本等单一模态，在本指南中，我们将演示如何使用名为IDEFICS的大型多模态模型来解决图像-文本任务。

[IDEFICS](../model_doc/idefics)是一种用于视觉和文本的开放访问的模型，基于[Flamingo](https://huggingface.co/papers/2204.14198)，这是首先由DeepMind开发的最先进的视觉语言模型。该模型接受任意的图像和文本输入序列，并生成完整的文本作为输出。它可以回答有关图像的问题，描述视觉内容，创建基于多个图像的故事等。IDEFICS有两个变体-[80亿个参数](https://huggingface.co/HuggingFaceM4/idefics-80b)和[9亿个参数](https://huggingface.co/HuggingFaceM4/idefics-9b)，这两个变体都可以在🤗 Hub上找到。对于每个变体，你还可以找到针对会话使用案例进行了调整的模型的精细调整的版本。

该模型非常灵活，可用于各种图像和多模态任务。然而，作为一个大型模型意味着它需要大量的计算资源和基础设施。你需要根据你的使用案例来决定这种方法是否比优化每个单独任务的专门模型更适合你的情况。

在本指南中，你将学习如何：
- [加载IDEFICS](#loading-the-model)，[加载模型的量化版本](#loading-the-quantized-version-of-the-model)
- 使用IDEFICS实现以下功能：
  - [图像字幕](#image-captioning)
  - [提示型图像字幕](#prompted-image-captioning)
  - [少量提示](#few-shot-prompting)
  - [视觉问答](#visual-question-answering)
  - [图像分类](#image-classification)
  - [基于图像的文本生成](#image-guided-text-generation)
- [以批处理模式运行推断](#running-inference-in-batch-mode)
- [IDEFICS训练以用于会话式使用](#idefics-instruct-for-conversational-use)

在开始之前，请确保已安装所有必要的库。

```bash
pip install -q bitsandbytes sentencepiece accelerate transformers
```

<Tip>
要使用模型检查点的非量化版本来运行以下示例，你将需要至少20GB的GPU内存。
</Tip>

## 加载模型

让我们从加载模型的9亿个参数检查点开始：

```py
>>> checkpoint = "HuggingFaceM4/idefics-9b"
```

与其他Transformer模型一样，你需要从检查点中加载处理器和模型本身。
IDEFICS处理器将[`LlamaTokenizer`]和IDEFICS图像处理器包装在一个单一处理器中，负责准备模型的文本和图像输入。

```py
>>> import torch

>>> from transformers import IdeficsForVisionText2Text, AutoProcessor

>>> processor = AutoProcessor.from_pretrained(checkpoint)

>>> model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="auto")
```

将`device_map`设置为`"auto"`将自动确定如何在现有设备上加载和存储模型权重，以最优化的方式。

### 量化模型

如果内存较小的GPU可用性是一个问题，你可以加载模型的量化版本。要加载模型和处理器的4位精度，请向`from_pretrained`方法传递`BitsAndBytesConfig`，并在加载时将模型压缩。

```py
>>> import torch
>>> from transformers import IdeficsForVisionText2Text, AutoProcessor, BitsAndBytesConfig

>>> quantization_config = BitsAndBytesConfig(
...     load_in_4bit=True,
...     bnb_4bit_compute_dtype=torch.float16,
... )

>>> processor = AutoProcessor.from_pretrained(checkpoint)

>>> model = IdeficsForVisionText2Text.from_pretrained(
...     checkpoint,
...     quantization_config=quantization_config,
...     device_map="auto"
... )
```

现在，你已经以建议的方式之一加载了模型，请继续探索可以使用IDEFICS的任务。

## 图像字幕

图像字幕是预测给定图像的标题的任务。一个常见的应用是帮助视觉障碍的人浏览不同的情况，例如在线浏览图像内容。

为了说明该任务，获取一个要添加字幕的图像，例如：

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-im-captioning.jpg" alt="一个花园床的小狗的图像" />
</div>

[Photo by Hendo Wang](https://unsplash.com/@hendoo)。

IDEFICS接受文本和图像提示。但是，要给图像添加字幕，你无需向模型提供文本提示，只需提供经过预处理的输入图像即可。模型将从开始序列令牌（BOS）开始生成文本，从而创建标题。

你可以使用图像对象（`PIL.Image`）或从中检索图像的URL作为模型的图像输入。

```py
>>> prompt = [
...     "https://images.unsplash.com/photo-1583160247711-2191776b4b91?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3542&q=80",
... ]

>>> inputs = processor(prompt, return_tensors="pt").to("cuda")
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, max_new_tokens=10, bad_words_ids=bad_words_ids)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> print(generated_text[0])
一个花园床的小狗
```

<Tip>

在调用`generate`时包括`bad_words_ids`是一个好主意，以避免在增加`max_new_tokens`时引发错误：模型将要生成一个新的`<image>`或`<fake_token_around_image>` token，而模型没有生成图像时。
你可以像本指南中一样即时设置它，也可以根据[文本生成策略](../generation_strategies.md)指南中的描述存储在`GenerationConfig`中。
</Tip>

## 提示型图像字幕

你可以通过提供一个文本提示来扩展图像字幕，模型将根据该文本继续生成图像。

让我们选取另一张图像来说明：

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-prompted-im-captioning.jpg" alt="巴黎夜晚的埃菲尔铁塔的图像" />
</div>

[Photo by Denys Nevozhai](https://unsplash.com/@dnevozhai)。

文本和图像提示可以作为单个列表传递给模型的处理器，以创建适当的输入。

```py
>>> prompt = [
...     "https://images.unsplash.com/photo-1543349689-9a4d426bee8e?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3501&q=80",
...     "这是一张图片，展示了",
... ]

>>> inputs = processor(prompt, return_tensors="pt").to("cuda")
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, max_new_tokens=10, bad_words_ids=bad_words_ids)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> print(generated_text[0])
这是一张图片，展示了巴黎的埃菲尔铁塔。
```

## 少量提示

尽管IDEFICS展示了很好的零提示结果，但你的任务可能需要一定格式的字幕，或者可能带有增加任务复杂性的其他限制或要求。少量提示可用于实现上下文学习。通过在提示中提供示例，你可以引导模型生成与给定示例格式相似的结果。

让我们以埃菲尔铁塔的先前图像为例，为模型构建一个提示，向模型展示除了对象在图像中是什么之外，我们还希望了解一些有趣的信息。然后，让我们看看是否可以为自由女神像的图像获得相同的响应格式：

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg" alt="自由女神像的图像" />
</div>

[Photo by Juan Mayobre](https://unsplash.com/@jmayobres)。

```py
>>> prompt = ["用户：",
...            "https://images.unsplash.com/photo-1543349689-9a4d426bee8e?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3501&q=80",
...            "描述这张图片。\n助手：一张巴黎夜晚的埃菲尔铁塔的图片。有趣的事实：埃菲尔铁塔和一座81层楼高的建筑一样高。\n",
...            "用户：",
...            "https://images.unsplash.com/photo-1524099163253-32b7f0256868?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3387&q=80",
...            "描述这张图片。\n助手:"
...            ]

>>> inputs = processor(prompt, return_tensors="pt").to("cuda")
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, max_new_tokens=30, bad_words_ids=bad_words_ids)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> print(generated_text[0])
用户：描述这张图片。
助手：一张巴黎夜晚的埃菲尔铁塔的图片。有趣的事实：埃菲尔铁塔和一座81层楼高的建筑一样高。
用户：描述这张图片。
助手：一张自由女神像的图片。有趣的事实：自由女神像高151英尺。
```

请注意，仅从单个示例（即1-shot）中，模型已学习到如何执行任务。对于更复杂的任务，可以尝试使用更多示例（例如3-shot、5-shot等）进行实验。

## 视觉问答

视觉问答（VQA）是一种基于图像回答开放式问题的任务。与图像字幕类似，它可以在可访问性应用中使用，也可以在教育（针对视觉资料的推理）、客户服务（基于图像的产品问题）和图像检索中使用。

让我们为这个任务得到一张新的图片：

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-vqa.jpg" alt="一对正在野餐的夫妇的图片" />
</div>

[Photo by Jarritos Mexican Soda](https://unsplash.com/@jarritos)。

你可以通过适当的提示将模型从图像字幕转向视觉问答：

```py
>>> prompt = [
...     "指令：对问题提供答案。使用图像回答。\n",
...     "https://images.unsplash.com/photo-1623944889288-cd147dbb517c?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3540&q=80",
...     "问题：这些人在哪里，天气如何？答案："
... ]

>>> inputs = processor(prompt, return_tensors="pt").to("cuda")
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, max_new_tokens=20, bad_words_ids=bad_words_ids)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> print(generated_text[0])
指令：对问题提供答案。使用图像回答。
问题：这些人在哪里，天气如何？答案：他们在纽约市的一个公园，天气很好。
```

## 图像分类

IDEFICS能够将图像分类到不同的类别，而无需显式地在包含了那些特定类别标签的数据上进行训练。给定一个类别列表，并使用其图像和文本理解能力，模型可以推断出图像可能属于哪个类别。

比如，我们有这张蔬菜摊的图片：

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-classification.jpg" alt="一个蔬菜摊的图片" />
</div>

[Photo by Peter Wendt](https://unsplash.com/@peterwendt)。

我们可以指示模型将图像分类为我们拥有的类别之一：

```py
>>> categories = ['动物', '蔬菜', '城市景观', '汽车', '办公室']
>>> prompt = [f"指令：将以下图像分类为以下列表中的单个类别：{categories}。\n",
...     "https://images.unsplash.com/photo-1471193945509-9ad0617afabf?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3540&q=80",    
...     "类别： "
... ]

>>> inputs = processor(prompt, return_tensors="pt").to("cuda")
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, max_new_tokens=4, bad_words_ids=bad_words_ids)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> print(generated_text[0])
指令：将以下图像分类为以下列表中的单个类别：['动物', '蔬菜', '城市景观', '汽车', '办公室']。
类别： 蔬菜
```  

在上面的示例中，我们要求模型将图像分类为单个类别，但是你也可以要求模型进行排名分类。

## 基于图像的文本生成

对于更具创造性的应用，可以使用基于图像的文本生成根据图像生成文本。这对于创建产品描述、广告、场景描述等非常有用。

例如，以下是一张产品的图片：

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-guided-text.jpg" alt="一个产品的图片" />
</div>

根据图像，你可以使用基于图像的文本生成生成文本。

```py
>>> prompt = [
...     "Instruction: Generate a description of the product based on the image.\n",
...     "https://images.unsplash.com/photo-1491239356657-7fb8e1f38f9e?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1281&q=80"    
... ]

>>> inputs = processor(prompt, return_tensors="pt").to("cuda")
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, max_new_tokens=30, bad_words_ids=bad_words_ids)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> print(generated_text[0])
Instruction: Generate a description of the product based on the image.
This is a beautiful wristwatch with a leather strap and a simple but elegant design.
```

以上示例中，我们要求模型基于图像生成产品描述，但是你可以根据不同的应用场景更改提示。

```


让我们提示IDEFICS根据一幅简单的红色门图片来写一个故事：

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-story-generation.jpg" alt="红色门上有一个南瓜的图片"/>
</div>

由[Craig Tidball](https://unsplash.com/@devonshiremedia)拍摄。

```py
>>> prompt = ["指令：利用这张图片写一个故事。\n",
...     "https://images.unsplash.com/photo-1517086822157-2b0358e7684a?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2203&q=80",
...     "故事：\n"]

>>> inputs = processor(prompt, return_tensors="pt").to("cuda")
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, num_beams=2, max_new_tokens=200, bad_words_ids=bad_words_ids)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> print(generated_text[0]) 
指令：利用这张图片写一个故事。
 故事：
从前，有一个小女孩住在一个有着红门的房子里。她喜欢她的红门，它是全世界最漂亮的门。

一天，小女孩在院子里玩耍时，她注意到有一个男人站在她的门口。他穿着一件长长的黑色外套和一顶高顶帽。

小女孩跑回屋里告诉妈妈这个男人的事。

她妈妈说：“别担心，亲爱的。他只是一个友好的幽灵。”

小女孩不确定自己是否相信她妈妈，但她还是出去了。

当她走到门口时，那个男人已经不见了。

第二天，小女孩再次在院子里玩耍时，她注意到那个男人又站在她的门口。

他穿着一件长黑外套和一顶高顶帽。

小女孩跑了过去。
```

看起来IDEFICS注意到门口的南瓜，并以一个有关幽灵的可怕万圣节故事作为创作主题。

<Tip>

对于这样较长的输出，调整文本生成策略将大大有助于提高生成输出的质量。查看[文本生成策略](../generation_strategies.md)了解更多信息。
</Tip>

## 批处理模式下的推断运行

之前的所有部分都是针对单个示例展示IDEFICS。类似地，你可以通过传递一系列提示来批处理运行推断：

```py
>>> prompts = [
...     [   "https://images.unsplash.com/photo-1543349689-9a4d426bee8e?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3501&q=80",
...         "这是一张图片，它展示了",
...     ],
...     [   "https://images.unsplash.com/photo-1623944889288-cd147dbb517c?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3540&q=80",
...         "这是一张图片，它展示了",
...     ],
...     [   "https://images.unsplash.com/photo-1471193945509-9ad0617afabf?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3540&q=80",
...         "这是一张图片，它展示了",
...     ],
... ]

>>> inputs = processor(prompts, return_tensors="pt")
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, max_new_tokens=10, bad_words_ids=bad_words_ids)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> for i,t in enumerate(generated_text):
...     print(f"{i}:\n{t}\n") 
0:
这是一张图片，它展示了位于巴黎法国的埃菲尔铁塔。

1:
这是一张图片，它展示了一对坐在野餐毯上的情侣。

2:
这是一张图片，它展示了一个蔬菜摊。
```

## IDEFICS指导对于对话使用

对于对话使用情况，你可以在🤗 Hub上找到经过微调的被指导版本的模型：`HuggingFaceM4/idefics-80b-instruct`和`HuggingFaceM4/idefics-9b-instruct`。

这些检查点是在监督学习和指令微调数据集的混合上微调的基础模型的结果，这提升了下游性能，同时使模型在对话环境中更易用。

使用和提示对于对话使用情况非常类似于使用基础模型：

```py
>>> import torch
>>> from transformers import IdeficsForVisionText2Text, AutoProcessor

>>> device = "cuda" if torch.cuda.is_available() else "cpu"

>>> checkpoint = "HuggingFaceM4/idefics-9b-instruct"
>>> model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
>>> processor = AutoProcessor.from_pretrained(checkpoint)

>>> prompts = [
...     [
...         "用户：这张图片里有什么？",
...         "https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG",
...         "<end_of_utterance>",

...         "\n助理：这幅图展示的是《阿斯特兰与奥柏利斯克》中的奥柏利斯克的狗Idefix。Idefix正在地上奔跑。<end_of_utterance>",

...         "\n用户：",
...         "https://static.wikia.nocookie.net/asterix/images/2/25/R22b.gif/revision/latest?cb=20110815073052",
...         "那是谁？<end_of_utterance>",

...         "\n助理：",
...     ],
... ]

>>> # --批处理模式
>>> inputs = processor(prompts, add_end_of_utterance_token=False, return_tensors="pt").to(device)
>>> # --单个样本模式
>>> # inputs = processor(prompts[0], return_tensors="pt").to(device)

>>> # 生成参数
>>> exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, max_length=100)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> for i, t in enumerate(generated_text):
...     print(f"{i}:\n{t}\n")
```
