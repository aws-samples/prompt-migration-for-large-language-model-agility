# OpenAI to Bedrock Migration

This codebase contains useful notebooks and tools for conducting OpenAI-to-AWS Migration POCs. This repo is designed to help users migrate FMs to Amazon Bedrock including converting raw prompts from source model to destination Claude model following best practices. More importantly, this workshop act as a playbook for users to experiment and help users to make decisions on whether to migrate to bedrock by evaluating source and destination LLMs with various  performance metrics and generate a comparison report with cost and latency for two use cases. 1. Call summarization 2. An Intelligent RAG based Financial Analyst. 
 

![Alt text](images/image.png)
There are three main steps in the model evaluation and migration process.
  1. Evaluation for Source Model
  2. Prompt Migration and Optimization for Target Model
  3. Evaluation fro Target Model

In the end of the notebook, we will generate a full comparison report of the source and target model comparison with performance metrics, latency and cost. By walking through each notebooks in the repo, users will gain in-depth understanding of how to evaluate and make a decision to migrate an application from OpenAI to Amazon Bedrock. The same process can be used for evaluation and migration of one LLM family to another on Amazon Bedrock. 


## Getting started

### Bedrock Model Setup
Go to the AWS Console -> Amazon Bedrock -> Model Access 

Model acces is available on the left panel, at the bottom of the menu. Select the models you wish to get access to and request access. (Please note the AWS region that you are in when requesting model access.)
![Alt text](images/image-1.png)



### IAM Policy for conducting OpenAI to Amazon Bedrock Migration in AWS cloud
This codebase requires an IAM role that has access to Amazon Bedrock. Here is an example IAM policy to grant access to Bedrock APIs and the other policies you will need for this workshop (DynamoDB, Lambda, Opensearch Serverless, Bedrock, IAM, Delete S3 Bucket):

IAM Policy for Advanced Agents for Bedrock Workshop This codebase requires an IAM role that has access to Amazon Bedrock. Here is an example IAM policy to grant access to Bedrock APIs and the other policies you will need for this workshop (DynamoDB, Lambda, Opensearch Serverless, Bedrock, IAM, Delete S3 Bucket):
```{
   {
	"Version": "2012-10-17",
	"Statement": [
		{
			"Sid": "OtherServices",
			"Effect": "Allow",
			"Action": [
                "dynamodb:*",
                "bedrock:*",
                "aoss:*",
                "lambda:*",
                "iam:*"
			],
			"Resource": "*"
		},
		{
			"Sid": "S3Delete",
			"Effect": "Allow",
			"Action": "s3:DeleteObject",
			"Resource": "arn:aws:s3:::*/*"
		},
		{
			"Sid": "S3DeleteBucket",
			"Effect": "Allow",
			"Action": "s3:DeleteBucket",
			"Resource": "arn:aws:s3:::*"
		}
	]
}
}
```

  ##### IMPORTANT NOTE:
  ##### If you are running the notebooks in SageMaker Studio, add this policy to the SageMaker role. If you are running the notebooks in your own environment, ensure your user/role has these permissions.

### AWS SageMaker
In the AWS console, navigate to Amazon SageMaker service and choose "Notebooks" on the left hand side panel.

- Create notebook instance Click on "Create notebook Instance".
![Alt text](images/image-2.png)

- Add the necessary details such as Name, instance type and execution role as shown in the diagram. Please create a new execution role with all necessary permissions if you don't have already.

![Alt text](images/image-3.png)

- Once the notebook gets created, you will see the Status as "InService" as shown below. Click on "Open JupyterLab" and it will launch the JupyterLab on browser.
![Alt text](images/image-4.png)

- Downloading code samples from the GitHub repo

- Download the code repo in zip file and upload the file to Jupyeter Lab on Sagemaker.
![Alt text](images/image-5.png)

- Open the terminal and unzip the file.
```unzip openai-to-bedrock-migration-codebase-external-main.zip```

Now you will see all the files in this repo ready to use for the migration process.

### Repo Structure Instructions

There will be two folders the users can get started with:
1. Feature Examples
2. Use Case Examples

#### Feature Examples
For individual features such as functions to invoke Claude model and generating Claude prompts using metaprompts, please refer to the "feature-examples" folder.
![Alt text](images/image-10.png)

#### Use Case Examples
Click on the use-case-examples and choose the use case you would like to start experimenting an end-to-end migration process for different use cases:
![Alt text](images/image-9.png)
1.  Call Summarization Use Case
2.  Investment Analysis Assistant RAG Use Case

* Run through notebooks under the "notebooks" folder for a end-to-end migration pipeline.
* The input data is stored under "data/sample_docs", if you wish to bring in your own data please replace the documents there.
* The prompts are stored under ""data/prompts" for different LLM model (ie. OpenAI or Claude). 
* Please replace the prompts for your source model under the repected folder. The migrated prompts for target model will 	also be stored here after running the prompt_migration notebook.
* The Evaluation Reports are stored under "outputs/evaluation_reports/" for different frameworks(ie. Deepeval or RAGAS)


## Resources and References

* [In-house RAGAS evaluation package with BedRock](https://gitlab.aws.dev/genaiic-reusable-assets/engagement-artifacts/ragas-evaluation/-/tree/main?ref_type=heads) 
* [Official RAGAS pacakge](https://github.com/explodinggradients/ragas)
* [Anthropic metaprompt method](https://docs.anthropic.com/en/docs/helper-metaprompt-experimental) 
* [Anthropic documentation for Claude models](https://docs.anthropic.com/en/docs/intro-to-claude)


## Support
longchn@amazon.com, aviyadc@amazon.com, uelaine@amazon.com, aminikha@amazon.com, fccal@amazon.com, ravividy@amazon.com


## Contributing
Contributions are welcome!


