# Oncology Journal Article Scraper and Vector Storage

## Introduction

This project involves scraping oncology journal articles, summarizing them, extracting keywords, and storing both the metadata and vector embeddings in a MySQL database and a Milvus vector database. The system is designed to allow efficient similarity-based queries for journal articles based on the vector embeddings of the titles.

The scraper fetches articles from a specified website, processes them to extract relevant metadata, and stores both the article details and their vector embeddings for future queries. The system supports batch processing to handle large volumes of data efficiently.

## Requirements

Before running the scripts, ensure that you have the necessary dependencies installed. You can install the required libraries using the `requirements.txt` file.

### To install dependencies:
1. Create a virtual environment (optional but recommended):
   ```
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

2. Install the required libraries:
 ```
  pip install -r requirements.txt
 ```
3. Configuration File
You need to create a configuration file named config.ini in the project directory. This file contains the connection details for MySQL and Milvus, as well as the URL for the web scraper and batch size for processing.

config.ini Example:
```
[mysql]
host = 
user = 
password = 
database = oncology

[milvus]
host = 
port = 

[web_scraper]
url = https://www.nature.com/subjects/oncology
batch_size = 10
```

### Approach
1. MySQL Database: The project uses a MySQL database to store metadata for oncology articles. The table includes columns for title, authors, publication date, abstract, summary, and keywords.

2. Milvus Vector Database: The project uses Milvus to store vector embeddings of article titles. These vectors enable similarity searches, allowing for efficient retrieval of articles based on their content.

2. Article Crawling and Summarization: The system crawls a specified URL for oncology articles, extracts metadata, summarizes abstracts, and generates keywords using NLP techniques. The embeddings of article titles are generated using a pre-trained BERT model.

4. Batch Processing: To optimize memory usage, the articles are processed in batches. Each batch is stored in both MySQL and Milvus, reducing memory consumption during large crawls.

5. Similarity Querying: Once the data is stored, you can query the system using a text query, which will return articles with the closest matching titles based on their vector similarity.

### Tech Used
+ Python 3.x: The programming language used for the project.
+ MySQL: A relational database management system to store metadata.
+ Milvus: A vector database used for storing and searching vector embeddings.
+ BeautifulSoup: A Python library for scraping web content.
+ NLTK & spaCy: Libraries for natural language processing, used for tokenization, stopword removal, and keyword extraction.
+ Hugging Face Transformers: Used for the pre-trained BERT model to generate vector embeddings.
+ Torch: The deep learning framework used for working with the BERT model.

### How to Run
1. Set up MySQL and Milvus: Ensure MySQL and Milvus are running on your local machine or use a remote instance.
2. Create a database in MySQL (e.g., oncology).
3. Ensure Milvus is running and accessible.
4. Create the Configuration File: Ensure config.ini is present with the correct connection details.
5. Run the Script: You can run the script via the command line with different commands based on your requirements.
   ```
   chmod +x run_script.sh
   ./run_script.sh
   ```

## Approach to Solving the Problem

1. **Crawler Application**: 
   - I created a Python-based crawler application to scrape data from the website "https://www.nature.com/subjects/oncology". The crawler specifically targets articles related to oncology, ensuring that no articles from other specialties are included.
   
2. **Data Extraction**: 
   - The crawler extracts essential metadata from each article, including:
     - **Title**: The article title.
     - **Author(s)**: A list of authors associated with the article.
     - **Publication Date**: The date when the article was published.
     - **Abstract**: A short summary of the article's content.
   - This data is then stored in a MySQL database for further use.

3. **Summarization and Keywords Extraction**: 
   - I implemented a summarization technique using a pre-trained language model to generate a short summary of each article’s abstract.
   - Additionally, I extracted relevant keywords from the article abstracts using NLP techniques (NLTK and spaCy).
   - Both the summary and keywords are stored in the MySQL database for each article.

4. **Vectorization and Storage in Milvus**: 
   - For efficient querying, I used a pre-trained BERT model to create vector embeddings of the article titles.
   - These vector embeddings are stored in Milvus, a vector database, which allows for efficient similarity-based searches.

5. **Querying**: 
   - The system allows users to query articles based on free-text English queries, such as "Give me the journal those are published last week".
   - The system performs a similarity search in the Milvus database and retrieves the most relevant articles based on the vector similarity of the query to the stored article titles.


### Future Scope
* Scaling Up: The system can be extended to scrape more websites or handle larger datasets by optimizing the crawling and storage process.
* Real-Time Updates: Implement real-time crawling and updating to keep the database up-to-date with new articles.
* Advanced Querying: Enhance the querying capabilities by adding more complex filters and searching based on other metadata fields (e.g., author, publication date).
* Integration with Other Systems: Integrate the system with other research databases or tools for advanced analysis and visualization of the data.
