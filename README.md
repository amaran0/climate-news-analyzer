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
* **Sources:** climate.gov, insideclimatenews.com, noaa.org
* **Size:** over 5,500 articles
* **Curation:** The dataset underwent cleaning methods such as boilerplate removal, targetting specific content, filtering/minimum length check, normalization, and deduplication

## Model
I fine-tuned the open-source pre-trained Mistral 7b model for this project using QLoRA

## How to Use/Demo
To explore our project locally or interact with the demo:

### Local Setup
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the demo script:**
    ```bash
    python run_demo.py
    ```
    *(Adjust script name as necessary, e.g., `app.py` for Streamlit apps)*

### Example Commands
* **Summarize an article:**
    ```bash
    python scripts/summarize.py --article_path data/sample_article.txt
    ```
* **Ask a question about a document:**
    ```bash
    python scripts/qa_system.py --document_path data/ipcc_report_excerpt.pdf --question "What are the primary drivers of sea-level rise?"
    ```
    *(Adjust script paths as necessary)*

### Video Demo
[Embed Link to your video demo (e.g., YouTube, Loom). Example: `https://www.youtube.com/watch?v=your_video_id`]

## Results & Evaluation
Our model demonstrates promising performance in extracting and synthesizing climate information.

* **Performance Metrics:**
    * **Summarization:** Achieved a ROUGE-1 score of [XX.X] and ROUGE-L score of [YY.Y] on our test set.
    * **Q&A:** Attained an F1-score of [AA.A]% and Exact Match (EM) score of [BB.B]% on our climate Q&A dataset.
    * [Include any other relevant metrics, e.g., classification accuracy].
* **Concrete Examples:**
    * **Original Article Excerpt:**
        > "Recent studies indicate a significant increase in the frequency and intensity of extreme weather events, including heatwaves and heavy rainfall, directly attributable to human-induced climate change. These changes pose severe threats to agricultural productivity and urban infrastructure..."
    * **Model-Generated Summary:**
        > "Human-induced climate change is causing more frequent and intense extreme weather, like heatwaves and heavy rainfall, threatening agriculture and urban infrastructure."
    * **Q&A Pair:**
        * **Question:** "What are some adaptation strategies for coastal communities?"
        * **Model Answer:** "Coastal communities can adapt by building seawalls, restoring natural coastal ecosystems like mangroves, and implementing early warning systems for storms."
* **Visualizations:**
    * * ![Most Frequent Adaptation Strategies](images/adaptation_strategies_chart.png)
    * ![Key Themes Word Cloud](images/themes_wordcloud.png)

## Future Work/Extensions
We envision several exciting avenues for future development:

* **Real-Time Processing:** Integrating with live news feeds for continuous, up-to-the-minute climate insights.
* **Broader Dataset Integration:** Expanding the dataset to include more diverse sources, such as social media, policy documents from various governments, and regional climate reports.
* **Full Web Deployment:** Developing a user-friendly web interface for wider accessibility and interaction.
* **Multi-modal Analysis:** Incorporating climate-related images, videos, and geospatial data for richer analysis.
* **Personalized Feeds:** Allowing users to customize their climate information feeds based on specific interests or regional focus.

## Contact/About Me
* **Your Name:** Arya Maran
* **LinkedIn:** https://www.linkedin.com/in/arya-maran
* **Email:** maran0@purdue.edu
