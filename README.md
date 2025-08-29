# AI-Powered Climate News & Adaptation Insights

## Build Status: BUILDING
## Overview
This project addresses the critical issue of climate change, aiming to enhance public understanding of both prominent and lesser-known environmental challenges. Recognizing the current overload of climate-related information, my goal is to develop a platform that condenses and presents this data in an accessible and engaging manner. This resource will be freely available, so as to allow replication/use for everyone.

## Features
This LLM-powered system is designed to perform the following key functions:

* **Summarize Climate Articles:** Generate concise summaries of lengthy news articles on climate change.
* **Q&A on Climate Impacts:** Answer specific questions regarding climate impacts on various sectors and regions.
* **Identify Adaptation Strategies:** Extract and categorize relevant climate adaptation strategies discussed in the text.
* **Spot Emerging Trends:** Identify recurring themes and emerging trends in climate discussions
* **Categorize Climate Information:** Classify articles and reports based on climate topics (e.g., policy, science, economics, regional impacts).

## Tech Stack
* **Programming Language:** Python
* **Retrival-Augmented-Generation (RAG):** ChromaDB, all-MiniLM-L6-v2, LangChain
* **Fine-tuning:** torch, transformers, peft
* **Web Scraping:** BeautifulSoup4, requests, lxml
* **Data Manipulation:** pandas, numpy, Hugging Face datasets
* **Dev Env:** Colab
* **Evaluation:** tqdm, loguru, scikit-learn, rouge-score

## Dataset
The model was trained and evaluated on a curated dataset of climate-related news articles and reports
* **Sources:** climate.gov, insideclimatenews.com, science.nasa.gov
* **Size:** over 4,500 articles
* **Curation:** The dataset underwent cleaning methods such as boilerplate removal, targetting specific content, filtering/minimum length check, normalization, and deduplication

## Model
I fine-tuned the open-source pre-trained Mistral 7b model for this project using QLoRA fine-tuning. But to do this, I first had to convert the cleaned data into intruction/input/output triplets. I did this by using GPT 3.5's API and running it through a series of possible summarization queries resulting in over 10,000 triplets ready for fine-tuning. To include RAG implementation, the article's content and metadata was embedded and stored in ChromaDB. I then went ahead with fine-tuning. Running the fine-tuning script on my local machine was inefficient and too weak, so I ran it remotely through a Colab notebook: 
https://colab.research.google.com/drive/17erg1mjtlQJcdwdLsXC40hmgMBpyT66e?usp=sharing.
I can now use the ChromaDB vector + fine-tuned Mistral 7b to query with RAG for better results.

## Results/Evaluation
This is where things got messy. When fine-tuning, it's typical to make a 80/10/10 split for training/evaluation/testing respectively. When creating my training and evaluation dataset, which both gets used during training for hyperparameter tweaking, monitoring training progress, and metric reporting, it turns out the 20% for eval/testing i had saved for later is what i actually used as my evaluation dataset. TLDR; I fine-tuned my model on 100% of my training data, so overfitting is possible. I don't have the resources to re-train from scratch, so I used gpt's API and selected 500 random examples from my training dataset to test my model on. This code is in test_dataset_creation.py under the model_training folder. I understand that benchmarking is now a 'bit' skewed, nonetheless progress had to be made somehow.

## Final Benchmarks
The final benchmarks can be found in results/eval_results.txt. Firstly, I'll explain what the 4 metrics represent:<br>
rouge1: Higher score means the model is using more of the same vocabulary as the ground truth answers<br>
rougeL: Higher score means the model is producing longer chunks of text that match the reference structure/order<br>
cosine: Refers to the semantic similarity between the model's answers and that of the ground truth<br>
latency: refers to the speed at which the model responds.

My model outperformed the base Mistral7b model in all metrics except latency.

rouge1: +16.6%
rougeL: +13.3%
cosine: +3.2%
latency: -38.3%

## Contact
* **Arya Maran**
* **LinkedIn:** https://www.linkedin.com/in/arya-maran
* **Email:** maran0@purdue.edu
