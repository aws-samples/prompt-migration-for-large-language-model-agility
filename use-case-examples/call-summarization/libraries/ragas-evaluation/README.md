# Evaluation framework for your Retrieval Augmented Generation (RAG) pipelines

> 🚀 Dedicated solutions to evaluate, monitor and improve performance of LLM & RAG application in production including custom models for production quality monitoring.[Talk to founders](https://calendly.com/shahules/30min)

Ragas is a framework that helps you evaluate your Retrieval Augmented Generation (RAG) pipelines. RAG denotes a class of LLM applications that use external data to augment the LLM’s context. There are existing tools and frameworks that help you build these pipelines but evaluating it and quantifying your pipeline performance can be hard. This is where Ragas (RAG Assessment) comes in.

Ragas provides you with the tools based on the latest research for evaluating LLM-generated text to give you insights about your RAG pipeline. Ragas can be integrated with your CI/CD to provide continuous checks to ensure performance.

## :shield: Installation

```bash
cd /home/ec2-user/SageMaker/openai-to-bedrock-migration-codebase-external/use-case-examples/call-summarization/libraries/ragas-evaluation
source activate python3
pip install -e . --quiet
pip install llama-index==0.9.6.post1
```

## :fire: Quickstart

This is a small example program you can run to see ragas in action!

```python

import os
import pandas as pd
from datasets import Dataset
from datasets import load_dataset
from ragas import evaluate
from ragas.metrics.answer_precision import answer_precision
from ragas.metrics.answer_recall import answer_recall
from ragas.metrics.answer_correctness import answer_correctness
from ragas.metrics.answer_relevance import answer_relevancy
from ragas.metrics.answer_similarity import answer_similarity
from ragas.metrics.context_precision import context_precision
from ragas.metrics.context_recall import context_recall
from ragas.metrics.faithfulness import faithfulness
from ragas.metrics.critique import AspectCritique

metrics = [
    answer_precision,
    answer_recall,
    answer_correctness,
    answer_similarity,
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall,
]

# prepare your huggingface dataset in the format
# Dataset({
#     features: ['question', 'contexts', 'answer', 'ground_truths'],
#     num_rows: 25
# })

result_csv_file = "result_test.csv"
result_df= pd.read_csv(result_csv_file)
result_ds = Dataset.from_pandas(result_df)

column_map = {
        "question": "Question",
        "contexts": "Contexts",
        "answer": "Bedrock Answer",
        "ground_truths": "Answer",
    }

# evaluate
eval_result = evaluate(result_ds, metrics=metrics, column_map=column_map)
```

**Important:**
Go into src/metrics_testing.ipynb for a walkthrough on how to call the functions and obtain all metrics needed for a particular dataset. Also for you to try it on your own.

Refer to our [documentation](https://docs.ragas.io/) to learn more.


## 🫂 Community

If you want to get more involved with Ragas, check out our [discord server](https://discord.gg/5djav8GGNZ). It's a fun community where we geek out about LLM, Retrieval, Production issues, and more.

## 🔍 Open Analytics

We track very basic usage metrics to guide us to figure out what our users want, what is working, and what's not. As a young startup, we have to be brutally honest about this which is why we are tracking these metrics. But as an Open Startup, we open-source all the data we collect. You can read more about this [here](https://github.com/explodinggradients/ragas/issues/49). **Ragas does not track any information that can be used to identify you or your company**. You can take a look at exactly what we track in the [code](./src/ragas/_analytics.py)

To disable usage-tracking you set the `RAGAS_DO_NOT_TRACK` flag to true.
