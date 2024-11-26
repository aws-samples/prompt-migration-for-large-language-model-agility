# Deep Eval Using Bedrock 

This repository demonstrates how to use Deep Eval by using Bedrock. This can be used for various LLM evaluation use cases, such as for summarization and question and answering. Here are the steps to run this: 

1. pip install deepeval

2. Make sure in your application to import the following user-created metrics: 
```
from bias.bias import BiasMetric
from faithfulness.faithfulness import FaithfulnessMetric
from answer_relevancy.answer_relevancy import AnswerRelevancyMetric
from toxicity.toxicity import ToxicityMetric
```

*** NOTE: Partial package that includes summarization and faithfulness only.