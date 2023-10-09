版权所有2021年The HuggingFace团队。保留所有权利。

根据Apache许可证，第2.0版（“许可证”），您不得使用此文件除非符合许可证的规定。您可以在以下位置获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于“原样”分发的，不附带任何形式的担保或条件。有关许可证下的特定语言的详细信息，请参见许可证。

⚠️注意，此文件是Markdown格式，但包含我们的文档构建器(类似于MDX)的特定的语法，可能无法在Markdown查看器中正常显示。

调试

多GPU网络问题调试

使用DistributedDataParallel和多个GPU进行训练或推理时，如果遇到进程和/或节点之间的互通问题，可以使用以下脚本来诊断网络问题。

```bash
wget https://raw.githubusercontent.com/huggingface/transformers/main/scripts/distributed/torch-distributed-gpu-test.py
```

例如，要测试2个GPU之间的交互情况，请运行以下命令：

```bash
python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 torch-distributed-gpu-test.py
```
如果两个进程可以互相通信并分配GPU内存，则会打印出OK状态。

对于更多的GPU或节点，请调整脚本中的参数。

您将在诊断脚本中找到更多详细信息，甚至可以找到在SLURM环境中如何运行它的示例。

另一个级别的调试是添加`NCCL_DEBUG=INFO`环境变量，如下所示：

```bash
NCCL_DEBUG=INFO python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 torch-distributed-gpu-test.py
```

这将输出大量与NCCL相关的调试信息，如果发现出现报告的某些问题，您可以在网上搜索其含义。或者，如果不确定如何解释输出，可以在问题中共享日志文件。


下溢和上溢检测

<Tip>

此功能目前仅适用于PyTorch。

</Tip>

<Tip>

对于多GPU训练，需要DDP（`torch.distributed.launch`）。

</Tip>

<Tip>

此功能可与任何基于`nn.Module`的模型一起使用。

</Tip>

如果您开始出现`loss=NaN`或模型由于激活或权重中的`inf`或`nan`而出现其他异常行为，需要找出第一个下溢或上溢发生的位置以及导致它的原因。幸运的是，您可以通过激活自动检测的特殊模块轻松实现。

如果您正在使用[`Trainer`]，您只需要添加：

```bash
--debug underflow_overflow
```

到正常的命令行参数中，或者在创建[`TrainingArguments`]对象时传递`debug="underflow_overflow"`。

如果您正在使用自己的训练循环或另一个训练器，可以使用以下方式实现相同的效果：

```python
from transformers.debug_utils import DebugUnderflowOverflow

debug_overflow = DebugUnderflowOverflow(model)
```

[`~debug_utils.DebugUnderflowOverflow`]在模型中插入钩子，在每次`forward`调用之后立即测试输入和输出变量以及相应模块的权重。一旦在激活或权重的至少一个元素中检测到`inf`或`nan`，程序将进行断言并打印出如下报告（此报告是在fp16混合精度下使用`google/mt5-small`截获的）：

```
在batch_number=0期间检测到inf/nan
最后21个forward帧：
最小绝对值  最大绝对值  元数据
                  encoder.block.1.layer.1.DenseReluDense.dropout Dropout
0.00e+00 2.57e+02 input[0]
0.00e+00 2.85e+02 output
[...]
                  encoder.block.2.layer.0 T5LayerSelfAttention
6.78e-04 3.15e+03 input[0]
2.65e-04 3.42e+03 output[0]
             None output[1]
2.25e-01 1.00e+04 output[2]
                  encoder.block.2.layer.1.layer_norm T5LayerNorm
8.69e-02 4.18e-01 weight
2.65e-04 3.42e+03 input[0]
1.79e-06 4.65e+00 output
                  encoder.block.2.layer.1.DenseReluDense.wi_0 Linear
2.17e-07 4.50e+00 weight
1.79e-06 4.65e+00 input[0]
2.68e-06 3.70e+01 output
                  encoder.block.2.layer.1.DenseReluDense.wi_1 Linear
8.08e-07 2.66e+01 weight
1.79e-06 4.65e+00 input[0]
1.27e-04 2.37e+02 output
                  encoder.block.2.layer.1.DenseReluDense.dropout Dropout
0.00e+00 8.76e+03 input[0]
0.00e+00 9.74e+03 output
                  encoder.block.2.layer.1.DenseReluDense.wo Linear
1.01e-06 6.44e+00 weight
0.00e+00 9.74e+03 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.DenseReluDense T5DenseGatedGeluDense
1.79e-06 4.65e+00 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.dropout Dropout
3.18e-04 6.27e+04 input[0]
0.00e+00      inf output
```

示例输出由于篇幅原因进行了裁剪。

第二列显示绝对最大元素的值，因此如果您仔细查看最后几个帧，输入和输出的范围在`1e4`左右。因此，当此训练使用fp16混合精度进行时，最后一步溢出（因为在`fp16`下，在`inf`之前的最大数字是64e3）。为了避免在`fp16`下发生溢出，激活值必须保持远远低于`1e4`，因为`1e4 * 1e4 = 1e8`，所以任何具有大激活值的矩阵乘法都会导致数值溢出。

在跟踪的开始处，可以发现问题发生的批次号（这里的“在batch_number=0期间检测到inf/nan”意味着问题发生在第一批）。

每个报告帧都以报告此帧报告的相应模块的完全限定输入开头。如果仅查看此帧：

```
                  encoder.block.2.layer.1.layer_norm T5LayerNorm
8.69e-02 4.18e-01 weight
2.65e-04 3.42e+03 input[0]
1.79e-06 4.65e+00 output
```

在这里，`encoder.block.2.layer.1.layer_norm`表示它是第二个块的第一层的层归一化。而`forward`的具体调用是`T5LayerNorm`。

让我们看一下报告的最后几个帧：

```
在batch_number=0期间检测到inf/nan
最后21个forward帧：
绝对最小值  绝对最大值  元数据
[...]
                  encoder.block.2.layer.1.DenseReluDense.wi_0 Linear
2.17e-07 4.50e+00 weight
1.79e-06 4.65e+00 input[0]
2.68e-06 3.70e+01 output
                  encoder.block.2.layer.1.DenseReluDense.wi_1 Linear
8.08e-07 2.66e+01 weight
1.79e-06 4.65e+00 input[0]
1.27e-04 2.37e+02 output
                  encoder.block.2.layer.1.DenseReluDense.wo Linear
1.01e-06 6.44e+00 weight
0.00e+00 9.74e+03 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.DenseReluDense T5DenseGatedGeluDense
1.79e-06 4.65e+00 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.dropout Dropout
3.18e-04 6.27e+04 input[0]
0.00e+00      inf output
```

最后一个帧报告了`Dropout.forward`函数的情况，其中第一个条目是唯一的输入，第二个条目是唯一的输出。可以看到它是从`DenseReluDense`类内的一个名为`dropout`的属性中调用的。我们可以看到它发生在第一层的第2个块中，在第一批时发生。最后，绝对最大输入元素为`6.27e+04`，输出也是`inf`。

您可以在此处看到`T5DenseGatedGeluDense.forward`产生了输出激活值，其绝对最大值约为62.7K，非常接近fp16的上限64K。接下来，我们有`Dropout`，它在将一些元素置零后重新规范化权重，这将把绝对最大值推到超过64K，并导致溢出（`inf`）。

从这里可以看出，当fp16数字开始变得非常大时，我们需要查看之前的帧。

让我们将报告与`models/t5/modeling_t5.py`中的代码进行匹配：

```python
class T5DenseGatedGeluDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gelu_act = ACT2FN["gelu_new"]

    def forward(self, hidden_states):
        hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states
```

现在可以轻松地看到`dropout`调用以及所有先前的调用。

由于检测是在前向钩子中进行的，因此在每次`forward`返回后立即打印这些报告。

回到完整报告，要对其进行操作并修复问题，我们需要回到数字开始上升的前几个帧，并且很可能在此处切换到`fp32`模式，以便在相乘或求和时数字不会溢出。当然，可能还有其他解决方案。例如，如果启用了`amp`，我们可以在将原始`forward`移动到帮助器包装器之后，暂时关闭`amp`，如下所示：

```python
def _forward(self, hidden_states):
    hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
    hidden_linear = self.wi_1(hidden_states)
    hidden_states = hidden_gelu * hidden_linear
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.wo(hidden_states)
    return hidden_states


import torch


def forward(self, hidden_states):
    if torch.is_autocast_enabled():
        with torch.cuda.amp.autocast(enabled=False):
            return self._forward(hidden_states)
    else:
        return self._forward(hidden_states)
```

由于自动检测器仅报告完整帧的输入和输出，因此一旦知道要查找的位置，您可能还希望分析特定`forward`函数的中间阶段。在这种情况下，您可以使用`detect_overflow`辅助函数将检测器注入到所需的位置，例如：

```python
from debug_utils import detect_overflow


class T5LayerFF(nn.Module):
    [...]

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        detect_overflow(forwarded_states, "after layer_norm")
        forwarded_states = self.DenseReluDense(forwarded_states)
        detect_overflow(forwarded_states, "after DenseReluDense")
        return hidden_states + self.dropout(forwarded_states)
```

可以看到，我们添加了其中的2个，并且现在我们跟踪了在它们之间的任何指定的`forward`函数中是否检测到`inf`或`nan`。

实际上，检测器已经报告了这些问题，因为上面的每个调用都是一个`nn.Module`，但是假设如果您有一些局部的直接计算，那么您就可以这样做。

此外，如果在自己的代码中实例化了调试器，可以调整保存其默认值的帧数，例如：

```python
from transformers.debug_utils import DebugUnderflowOverflow

debug_overflow = DebugUnderflowOverflow(model, max_frames_to_save=100)
```

特定批次的绝对最小值和最大值跟踪

同一调试类可以用于关闭下溢/上溢检测功能时的每批次跟踪。

假设您想要观察给定批次的每个`forward`调用的所有因素的绝对最小值和最大值，并且只对第1批和第3批进行跟踪。然后，您可以将此类实例化为：

```python
debug_overflow = DebugUnderflowOverflow(model, trace_batch_nums=[1, 3])
```

现在，将使用与下溢/上溢检测器相同的格式跟踪完整的第1批和第3批。

批次编号从0开始索引。

这对于您知道程序在某个批次号之后开始出现问题时非常有用，因此您可以直接跳转到该区域。这里是这样一个配置的示例截断输出：

```
                  *** 开始批次号=1 ***
绝对最小值  绝对最大值  元数据
                  共享的Embedding
1.01e-06 7.92e+02 weight
0.00e+00 2.47e+04 input[0]
5.36e-05 7.92e+02 output
[...]
                  decoder.dropout Dropout
1.60e-07 2.27e+01 input[0]
0.00e+00 2.52e+01 output
                  decoder T5Stack
     不是一个张量 output
                  lm_head Linear
1.01e-06 7.92e+02 weight
0.00e+00 1.11e+00 input[0]
6.06e-02 8.39e+01 output
                   T5ForConditionalGeneration
     不是一个张量 output

                  *** 开始批次号=3 ***
绝对最小值  绝对最大值  元数据
                  共享的Embedding
1.01e-06 7.92e+02 weight
0.00e+00 2.78e+04 input[0]
5.36e-05 7.92e+02 output
[...]
```

在此配置中，您将获得大量转储的帧-与您的模型中的每个前向调用一样多的帧，因此可能是您想要的，也可能不是，但有时候与正常调试器一起使用比使用正常调试器更容易进行调试。例如，如果问题在第150批之后开始出现。因此，您可以转储149批和150批的跟踪，并比较数字开始发散的位置。

您还可以指定在哪个批次号之后停止训练，例如：

```python
debug_overflow = DebugUnderflowOverflow(model, trace_batch_nums=[1, 3], abort_after_batch_num=3)
```
