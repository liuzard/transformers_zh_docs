# 使用Trainer API进行超参数搜索

🤗 Transformers 提供了一个经过优化的 [`Trainer`] 类来训练🤗 Transformers 模型，可以让你在没有手动编写训练循环的情况下轻松开始训练。[`Trainer`] 提供了用于超参数搜索的 API。本文档将展示如何在示例中启用超参数搜索。

## 超参数搜索后端

[`Trainer`] 当前支持四种超参数搜索后端：
[optuna](https://optuna.org/)，[sigopt](https://sigopt.com/)，[raytune](https://docs.ray.io/en/latest/tune/index.html) 和 [wandb](https://wandb.ai/site/sweeps)。

在使用这些超参数搜索后端之前，你需要安装它们：
```bash
pip install optuna/sigopt/wandb/ray[tune] 
```

## 如何在示例中启用超参数搜索

定义超参数搜索空间，不同的后端需要不同的格式。

对于 sigopt，可以参考 sigopt 的 [object_parameter](https://docs.sigopt.com/ai-module-api-references/api_reference/objects/object_parameter)，示例如下：
```py
>>> def sigopt_hp_space(trial):
...     return [
...         {"bounds": {"min": 1e-6, "max": 1e-4}, "name": "learning_rate", "type": "double"},
...         {
...             "categorical_values": ["16", "32", "64", "128"],
...             "name": "per_device_train_batch_size",
...             "type": "categorical",
...         },
...     ]
```

对于 optuna，可以参考 optuna 的 [object_parameter](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html#sphx-glr-tutorial-10-key-features-002-configurations-py)，示例如下：

```py
>>> def optuna_hp_space(trial):
...     return {
...         "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
...         "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64, 128]),
...     }
```

Optuna 提供了多目标超参数优化。你可以在 `hyperparameter_search` 中传递 `direction` 并定义自己的 `compute_objective` 函数来返回多个目标值。Pareto 前沿（`List[BestRun]`）将在超参数搜索中返回，你可以参考 [test_trainer](https://github.com/huggingface/transformers/blob/main/tests/trainer/test_trainer.py) 中的 `TrainerHyperParameterMultiObjectOptunaIntegrationTest` 测试用例。示例如下

```py
>>> best_trials = trainer.hyperparameter_search(
...     direction=["minimize", "maximize"],
...     backend="optuna",
...     hp_space=optuna_hp_space,
...     n_trials=20,
...     compute_objective=compute_objective,
... )
```

对于 raytune，可以参考 raytune 的 [object_parameter](https://docs.ray.io/en/latest/tune/api/search_space.html)，示例如下：

```py
>>> def ray_hp_space(trial):
...     return {
...         "learning_rate": tune.loguniform(1e-6, 1e-4),
...         "per_device_train_batch_size": tune.choice([16, 32, 64, 128]),
...     }
```

对于 wandb，可以参考 wandb 的 [object_parameter](https://docs.wandb.ai/guides/sweeps/configuration)，示例如下：

```py
>>> def wandb_hp_space(trial):
...     return {
...         "method": "random",
...         "metric": {"name": "objective", "goal": "minimize"},
...         "parameters": {
...             "learning_rate": {"distribution": "uniform", "min": 1e-6, "max": 1e-4},
...             "per_device_train_batch_size": {"values": [16, 32, 64, 128]},
...         },
...     }
```

定义一个 `model_init` 函数并将其传递给 [`Trainer`]，示例如下：
```py
>>> def model_init(trial):
...     return AutoModelForSequenceClassification.from_pretrained(
...         model_args.model_name_or_path,
...         from_tf=bool(".ckpt" in model_args.model_name_or_path),
...         config=config,
...         cache_dir=model_args.cache_dir,
...         revision=model_args.model_revision,
...         use_auth_token=True if model_args.use_auth_token else None,
...     )
```

使用 `model_init` 函数、训练参数、训练集和测试集以及评估函数创建一个 [`Trainer`]：

```py
>>> trainer = Trainer(
...     model=None,
...     args=training_args,
...     train_dataset=small_train_dataset,
...     eval_dataset=small_eval_dataset,
...     compute_metrics=compute_metrics,
...     tokenizer=tokenizer,
...     model_init=model_init,
...     data_collator=data_collator,
... )
```

调用超参数搜索，获取最佳试验参数，后端可以是 `"optuna"`/`"sigopt"`/`"wandb"`/`"ray"`。direction 可以是 `"minimize"` 或 `"maximize"`，表示优化较大还是较小的目标。

如果没有定义自己的 `compute_objective` 函数，则会调用默认的 `compute_objective` 函数，并将评估度量（如 f1）的总和作为目标值返回。

```py
>>> best_trial = trainer.hyperparameter_search(
...     direction="maximize",
...     backend="optuna",
...     hp_space=optuna_hp_space,
...     n_trials=20,
...     compute_objective=compute_objective,
... )
```

## DDP 微调的超参数搜索
目前，DDP 的超参数搜索仅适用于 optuna 和 sigopt。只有 rank-zero 进程会生成搜索试验并将参数传递给其他进程。
