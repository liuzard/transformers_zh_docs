# 固定长度模型的困惑度

[[在 Colab 中打开]](链接)

困惑度 (Perplexity, PPL) 是评估语言模型最常见的指标之一。在深入讨论之前，我们应该注意到，此指标仅适用于传统语言模型（有时也称为自回归或因果语言模型），而对于像 BERT 这样的遮蔽语言模型，其定义不明确（详情请参阅[模型摘要](model_summary.md)）。

困惑度定义为序列的负对数似然的指数平均值。如果我们有一个标记化的序列 \\(X = (x_0, x_1, \dots, x_t)\\)，那么 \\(X\\) 的困惑度为，

$$\text{PPL}(X) = \exp \left\{ {-\frac{1}{t}\sum_i^t \log p_\theta (x_i|x_{<i}) } \right\}$$

其中 \\(\log p_\theta (x_i|x_{<i})\\) 是模型根据前面的标记 \\(x_{<i}\\) 对第 i 个标记进行建模的对数似然。从直观上讲，困惑度可以被视为模型在给定语料库中的一组指定标记中均匀进行预测的能力的评估。重要的是，这意味着标记化过程对模型的困惑度产生了直接影响，在比较不同模型时，应始终考虑这一点。

这也等价于数据和模型预测之间的交叉熵的指数。有关困惑度及其与每字符比特数 (Bits Per Character，BPC) 和数据压缩的关系的更多直观理解，请参阅[The Gradient](https://thegradient.pub/understanding-evaluation-metrics-for-language-models/)上的[很棒的博客文章](https://thegradient.pub/understanding-evaluation-metrics-for-language-models/)。

## 使用固定长度模型计算困惑度

如果我们不受模型上下文大小的限制，可以通过自回归地对序列进行分解，并在每个步骤中基于整个前序子序列进行建模来评估模型的困惑度，如下所示。


<img width="600" alt="完全分解具有无限上下文长度的序列" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/ppl_full.gif"/>

然而，在处理近似模型时，我们通常对模型可以处理的标记数量有限制。例如，最大的 [GPT-2](model_doc/gpt2) 版本的固定长度为 1024 个标记，因此当 \\(t\\) 大于 1024 时，我们无法直接计算 \\(p_\theta(x_t|x_{<t})\\)。

代替方法通常是将序列分成与模型的最大输入尺寸相同的子序列。如果模型的最大输入尺寸为 \\(k\\)，那么我们只通过前面的 \\(k-1\\) 个标记对标记 \\(x_t\\) 的似然进行近似，而不是使用整个上下文。在评估序列的模型困惑度时，一个诱人但次优的方法是将序列分成不重叠的块，并独立地累积每个分段的分解对数似然。

<img width="600" alt="不利用完整上下文信息的次优困惑度分解" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/ppl_chunked.gif"/>

这样计算速度很快，因为每个分段的困惑度可以在一次前向传递中计算得出，但是这种方法对完全分解的困惑度表示不足，并且通常会导致较高（较差）的困惑度，因为模型在大多数预测步骤中的上下文更少。

相反，应该使用滑动窗口策略来评估固定长度模型的困惑度。这涉及反复滑动上下文窗口，以便模型在进行每个预测时具有更多上下文。

<img width="600" alt="利用所有可用上下文信息的滑动窗口困惑度分解" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/ppl_sliding.gif"/>

这更接近于序列概率的真正分解，并且通常会得到一个更有利的分数。缺点是它需要对语料库中的每个标记进行单独的前向传递。一个好的实际折中方法是使用跨幅滑动窗口，通过较大的跨度移动上下文，而不是每次移动一个标记。这样可以加快计算速度，同时仍然为模型在每一步中提供大量的上下文来进行预测。

## 示例：使用 🤗Transformers 中的 GPT-2 计算困惑度

让我们以 GPT-2 为例演示此过程。

```python
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

device = "cuda"
model_id = "gpt2-large"
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
```

我们将加载 WikiText-2 数据集并使用几种不同的滑动窗口策略来评估困惑度。由于此数据集很小，并且只对数据集进行一次前向传递，所以我们可以只将整个数据集加载和编码到内存中。

```python
from datasets import load_dataset

test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
```

使用 🤗Transformers，我们可以将 `input_ids` 直接传递给我们的模型作为 `labels`，每个标记的平均负对数似然将作为损失返回。然而，对于我们的滑动窗口方法，我们传递给模型的标记存在重叠。我们不希望在损失中包括我们只将其视为上下文的标记的对数似然，因此我们可以将这些目标设置为 `-100`，以将其忽略。以下是我们如何使用步幅为 `512` 的示例。这意味着在计算任何一个标记的条件似然时，模型将至少有 512 个标记的上下文（前提是有 512 个先行标记可用于条件建模）。

```python
import torch
from tqdm import tqdm

max_length = model.config.n_positions
stride = 512
seq_len = encodings.input_ids.size(1)

nlls = []
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # 可能与最后一个步骤上的步幅不同
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

        # 损失使用交叉熵损失函数计算，该函数对有效标签进行平均
        # 注意，模型仅在 trg_len - 1 个标签上计算损失，因为它在内部将标签左移 1 位。
        neg_log_likelihood = outputs.loss

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())
```

将步幅长度设置为最大输入长度时运行此代码等效于我们上面讨论的次优、非滑动窗口策略。步幅越小，模型在进行每个预测时具有的上下文越多，报告的困惑度通常越好。

当我们使用 `stride = 1024`（即无重叠）运行上述代码时，得到的困惑度是 `19.44`，与 GPT-2 论文中报告的 `19.93` 差不多。通过使用 `stride = 512` 或其他步幅值来采用我们的滑动窗口策略，困惑度下降至 `16.45`。这不仅是一个更有利的分数，而且是通过更接近序列概率的真正分解进行计算的。