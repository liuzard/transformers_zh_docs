<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Share a model

The last two tutorials showed how you can fine-tune a model with PyTorch, Keras, and 🤗 Accelerate for distributed setups. The next step is to share your model with the community! At Hugging Face, we believe in openly sharing knowledge and resources to democratize artificial intelligence for everyone. We encourage you to consider sharing your model with the community to help others save time and resources.

In this tutorial, you will learn two methods for sharing a trained or fine-tuned model on the [Model Hub](https://huggingface.co/models):

- Programmatically push your files to the Hub.
- Drag-and-drop your files to the Hub with the web interface.

<iframe width="560" height="315" src="https://www.youtube.com/embed/XvSGPZFEjDY" title="YouTube video player"
frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope;
picture-in-picture" allowfullscreen></iframe>

<Tip>

To share a model with the community, you need an account on [huggingface.co](https://huggingface.co/join). You can also join an existing organization or create a new one.

</Tip>

## Repository features

Each repository on the Model Hub behaves like a typical GitHub repository. Our repositories offer versioning, commit history, and the ability to visualize differences.

The Model Hub's built-in versioning is based on git and [git-lfs](https://git-lfs.github.com/). In other words, you can treat one model as one repository, enabling greater access control and scalability. Version control allows *revisions*, a method for pinning a specific version of a model with a commit hash, tag or branch.

As a result, you can load a specific model version with the `revision` parameter:

```py
>>> model = AutoModel.from_pretrained(
...     "julien-c/EsperBERTo-small", revision="v2.0.1"  # tag name, or branch name, or commit hash
... )
```

Files are also easily edited in a repository, and you can view the commit history as well as the difference:

![vis_diff](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vis_diff.png)

## Setup

Before sharing a model to the Hub, you will need your Hugging Face credentials. If you have access to a terminal, run the following command in the virtual environment where 🤗 Transformers is installed. This will store your access token in your Hugging Face cache folder (`~/.cache/` by default):

```bash
huggingface-cli login
```

If you are using a notebook like Jupyter or Colaboratory, make sure you have the [`huggingface_hub`](https://huggingface.co/docs/hub/adding-a-library) library installed. This library allows you to programmatically interact with the Hub.

```bash
pip install huggingface_hub
```

Then use `notebook_login` to sign-in to the Hub, and follow the link [here](https://huggingface.co/settings/token) to generate a token to login with:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## Convert a model for all frameworks

To ensure your model can be used by someone working with a different framework, we recommend you convert and upload your model with both PyTorch and TensorFlow checkpoints. While users are still able to load your model from a different framework if you skip this step, it will be slower because 🤗 Transformers will need to convert the checkpoint on-the-fly.

Converting a checkpoint for another framework is easy. Make sure you have PyTorch and TensorFlow installed (see [here](installation) for installation instructions), and then find the specific model for your task in the other framework. 

<frameworkcontent>
<pt>
Specify `from_tf=True` to convert a checkpoint from TensorFlow to PyTorch:

```py
>>> pt_model = DistilBertForSequenceClassification.from_pretrained("path/to/awesome-name-you-picked", from_tf=True)
>>> pt_model.save_pretrained("path/to/awesome-name-you-picked")
```
</pt>
<tf>
Specify `from_pt=True` to convert a checkpoint from PyTorch to TensorFlow:

```py
>>> tf_model = TFDistilBertForSequenceClassification.from_pretrained("path/to/awesome-name-you-picked", from_pt=True)
```

Then you can save your new TensorFlow model with its new checkpoint:

```py
>>> tf_model.save_pretrained("path/to/awesome-name-you-picked")
```
</tf>
<jax>
If a model is available in Flax, you can also convert a checkpoint from PyTorch to Flax:

```py
>>> flax_model = FlaxDistilBertForSequenceClassification.from_pretrained(
...     "path/to/awesome-name-you-picked", from_pt=True
... )
```
</jax>
</frameworkcontent>

## Push a model during training

<frameworkcontent>
<pt>
<Youtube id="Z1-XMy-GNLQ"/>

Sharing a model to the Hub is as simple as adding an extra parameter or callback. Remember from the [fine-tuning tutorial](training), the [`TrainingArguments`] class is where you specify hyperparameters and additional training options. One of these training options includes the ability to push a model directly to the Hub. Set `push_to_hub=True` in your [`TrainingArguments`]:

```py
>>> training_args = TrainingArguments(output_dir="my-awesome-model", push_to_hub=True)
```

Pass your training arguments as usual to [`Trainer`]:

```py
>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=small_train_dataset,
...     eval_dataset=small_eval_dataset,
...     compute_metrics=compute_metrics,
... )
```

After you fine-tune your model, call [`~transformers.Trainer.push_to_hub`] on [`Trainer`] to push the trained model to the Hub. 🤗 Transformers will even automatically add training hyperparameters, training results and framework versions to your model card!

```py
>>> trainer.push_to_hub()
```
</pt>
<tf>
Share a model to the Hub with [`PushToHubCallback`]. In the [`PushToHubCallback`] function, add:

- An output directory for your model.
- A tokenizer.
- The `hub_model_id`, which is your Hub username and model name.

```py
>>> from transformers import PushToHubCallback

>>> push_to_hub_callback = PushToHubCallback(
...     output_dir="./your_model_save_path", tokenizer=tokenizer, hub_model_id="your-username/my-awesome-model"
... )
```

Add the callback to [`fit`](https://keras.io/api/models/model_training_apis/), and 🤗 Transformers will push the trained model to the Hub:

```py
>>> model.fit(tf_train_dataset, validation_data=tf_validation_dataset, epochs=3, callbacks=push_to_hub_callback)
```
</tf>
</frameworkcontent>

## Use the `push_to_hub` function

You can also call `push_to_hub` directly on your model to upload it to the Hub.

Specify your model name in `push_to_hub`:

```py
>>> pt_model.push_to_hub("my-awesome-model")
```

This creates a repository under your username with the model name `my-awesome-model`. Users can now load your model with the `from_pretrained` function:

```py
>>> from transformers import AutoModel

>>> model = AutoModel.from_pretrained("your_username/my-awesome-model")
```

If you belong to an organization and want to push your model under the organization name instead, just add it to the `repo_id`:

```py
>>> pt_model.push_to_hub("my-awesome-org/my-awesome-model")
```

The `push_to_hub` function can also be used to add other files to a model repository. For example, add a tokenizer to a model repository:

```py
>>> tokenizer.push_to_hub("my-awesome-model")
```

Or perhaps you'd like to add the TensorFlow version of your fine-tuned PyTorch model:

```py
>>> tf_model.push_to_hub("my-awesome-model")
```

Now when you navigate to your Hugging Face profile, you should see your newly created model repository. Clicking on the **Files** tab will display all the files you've uploaded to the repository.

For more details on how to create and upload files to a repository, refer to the Hub documentation [here](https://huggingface.co/docs/hub/how-to-upstream).

## Upload with the web interface

Users who prefer a no-code approach are able to upload a model through the Hub's web interface. Visit [huggingface.co/new](https://huggingface.co/new) to create a new repository:

![new_model_repo](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/new_model_repo.png)

From here, add some information about your model:

- Select the **owner** of the repository. This can be yourself or any of the organizations you belong to.
- Pick a name for your model, which will also be the repository name.
- Choose whether your model is public or private.
- Specify the license usage for your model.

Now click on the **Files** tab and click on the **Add file** button to upload a new file to your repository. Then drag-and-drop a file to upload and add a commit message.

![upload_file](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/upload_file.png)

## Add a model card

To make sure users understand your model's capabilities, limitations, potential biases and ethical considerations, please add a model card to your repository. The model card is defined in the `README.md` file. You can add a model card by:

* Manually creating and uploading a `README.md` file.
* Clicking on the **Edit model card** button in your model repository.

Take a look at the DistilBert [model card](https://huggingface.co/distilbert-base-uncased) for a good example of the type of information a model card should include. For more details about other options you can control in the `README.md` file such as a model's carbon footprint or widget examples, refer to the documentation [here](https://huggingface.co/docs/hub/models-cards).
