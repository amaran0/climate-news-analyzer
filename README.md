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
I fine-tuned the open-source pre-trained Mistral 7b model for this project using QLoRA fine-tuning. But to do this, I first had to convert the cleaned data into intruction/input/output triplets. I did this by using GPT 3.5's API and running it through a series of possible summarization queries resulting in over 10,000 triplets ready for fine-tuning.

## Contact/About Me
* **Your Name:** Arya Maran
* **LinkedIn:** https://www.linkedin.com/in/arya-maran
* **Email:** maran0@purdue.edu
