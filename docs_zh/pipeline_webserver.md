<!--⚠️ 注意: 这个文件是在Markdown中的，但是包含了我们的文档构建器（类似MDX）的特定语法，可能在你的Markdown查看器中无法正常呈现。-->

# 使用pipeline来构建web服务器

<Tip>
创建一个推理引擎是一个复杂的主题，而且“最佳”解决方案很可能取决于你的问题空间。你是在使用CPU还是GPU？你想要最低的延迟、最高的吞吐量、支持多个模型还是只优化一个特定模型？解决这个问题有很多方法，所以我们将要介绍的是一个很好的默认方法，但可能不一定是最优的解决方案。
</Tip>

重要的是要理解，我们可以使用一个迭代器，就像你在[数据集](pipeline_tutorial.md#using-pipelines-on-a-dataset)上使用的那样，在web服务器基本上是一个等待请求并随时处理请求的系统。

通常情况下，Web服务器是多路复用的（多线程、异步等等），以处理各种并发请求。另一方面，pipeline（和底层的模型）并不非常适合并行处理；它们会占用大量的RAM，所以最好在运行时为它们提供所有可用的资源，或者它是一项计算密集型任务。

我们将通过使Web服务器处理接收和发送请求的轻负载，并有一个单独的线程来处理实际的工作来解决这个问题。
这个示例将使用`starlette`。实际的框架并不是很重要，但如果你使用其他框架可能需要调整或更改代码以达到相同的效果。

创建`server.py`:

```py
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from transformers import pipeline
import asyncio


async def homepage(request):
    payload = await request.body()
    string = payload.decode("utf-8")
    response_q = asyncio.Queue()
    await request.app.model_queue.put((string, response_q))
    output = await response_q.get()
    return JSONResponse(output)


async def server_loop(q):
    pipe = pipeline(model="bert-base-uncased")
    while True:
        (string, response_q) = await q.get()
        out = pipe(string)
        await response_q.put(out)


app = Starlette(
    routes=[
        Route("/", homepage, methods=["POST"]),
    ],
)


@app.on_event("startup")
async def startup_event():
    q = asyncio.Queue()
    app.model_queue = q
    asyncio.create_task(server_loop(q))
```

现在你可以用以下命令运行它：
```bash
uvicorn server:app
```

然后你可以向它发出请求：
```bash
curl -X POST -d "test [MASK]" http://localhost:8000/
#[{"score":0.7742936015129089,"token":1012,"token_str":".","sequence":"test."},...]
```

到这里，你现在已经了解了如何创建一个Web服务器！

真正重要的是，我们只加载模型**一次**，所以Web服务器上没有多个模型的副本。这样，就不会浪费不必要的内存。
然后队列机制允许你做一些花哨的事情，比如在进行推理之前累积一些项目以实现动态批处理：

<Tip warning={true}>

下面的代码示例故意使用类似伪代码的方式编写，以提高可读性。
在没有检查它是否适合你的系统资源之前，请不要运行它！

</Tip>

```py
(string, rq) = await q.get()
strings = []
queues = []
while True:
    try:
        (string, rq) = await asyncio.wait_for(q.get(), timeout=0.001)  # 1ms
    except asyncio.exceptions.TimeoutError:
        break
    strings.append(string)
    queues.append(rq)
strings
outs = pipe(strings, batch_size=len(strings))
for rq, out in zip(queues, outs):
    await rq.put(out)
```

同样，推荐的代码是为了提高可读性而进行优化的，并不一定是最佳的代码。
首先，没有批处理大小的限制，通常并不是一个好主意。接下来，超时是在每次队列获取时重置的，这意味着在运行推理之前，你可能要等待比 1ms 更长的时间（从而延迟第一个请求的处理）。

最好是设置一个单独的1ms截止时间。

即使队列为空，它也将始终等待1ms，这可能并不是最佳的解决方法，因为如果队列中没有任何内容，你可能希望开始进行推理。
但是，如果批处理对你的用例非常重要，那么可能也是有道理的。
总之，没有一种最好的解决方案。

## 你可能想要考虑的几件事

### 错误检查

在生产环境中，有很多可能出错的地方：内存不足、空间不足、加载模型可能失败、查询可能错误、查询可能正确但由于模型配置错误而无法运行等等。

通常情况下，服务器将错误输出给用户是个好主意，因此，添加许多`try..except`语句以显示这些错误是个好主意。但要记住，根据你的安全上下文，这也可能是一个安全风险，因为它可能透露出所有这些错误。

### 断路器

Web服务器通常在超载时做断路处理，这意味着它们会返回适当的错误，而不是无限期地等待查询。返回503错误，而不是等待很长时间或在很长时间后返回504错误。

根据提议的代码，这在实现起来相对容易，因为只有一个队列。查看队列大小是开始在负载压力下返回错误的基本方法。

### 阻塞主线程

目前，PyTorch不支持异步，计算会在运行时阻塞主线程。这意味着如果能强制PyTorch在自己的线程/进程上运行会更好。这里没有这样做，因为代码变得更加复杂（主要是因为线程、异步和队列不太好地一起工作）。但最终的效果是相同的。

在推理时间较长（> 1秒）的情况下，这将非常重要，因为在这种情况下，意味着每个推理请求在收到错误之前必须等待1秒。

### 动态批处理

一般来说，批处理不一定比单个项目传递要好（有关更多信息，请参见[批处理细节](main_classes/pipelines#pipeline-batching)）。但在正确的情况下，它可以非常有效。API中默认没有提供动态批处理（过多的机会会导致速度变慢）。但对于BLOOM推理——一个非常大的模型——动态批处理对于为每个人提供良好的体验来说是**必不可少**的。