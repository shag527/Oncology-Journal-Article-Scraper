import requests
import sys
from bs4 import BeautifulSoup
import mysql.connector
import configparser
from datetime import datetime
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import nltk
from nltk.tokenize import word_tokenize
import spacy
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

# Load configuration from config.ini
config = configparser.ConfigParser()
config.read('config.ini')

# Download NLTK and spaCy models
nltk.download("punkt")
nltk.download("stopwords")
nlp = spacy.load("en_core_web_sm")

# Initialize summarization pipeline and embedding model
summarizer = pipeline("summarization", model="t5-small")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
embedding_model = AutoModel.from_pretrained("bert-base-uncased")

def connect_to_mysql():
    """
    Establishes a connection to the MySQL database using configuration details.
    Ensures proper connection handling for database interactions.
    """
    return mysql.connector.connect(
        host=config['mysql']['host'],
        user=config['mysql']['user'],
        password=config['mysql']['password'],
        database=config['mysql']['database']
    )

def create_mysql_table():
    """
    Creates the MySQL table `oncology_articles` if it does not exist.
    This table stores metadata about oncology journal articles.
    """
    connection = connect_to_mysql()
    cursor = connection.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS oncology_articles (
            id INT AUTO_INCREMENT PRIMARY KEY,
            title VARCHAR(255),
            authors TEXT,
            publication_date DATE,
            abstract TEXT,
            summary TEXT,
            keywords TEXT
        );
    ''')
    connection.close()

def connect_to_milvus():
    """
    Connects to the Milvus vector database using configuration details.
    Ensures compatibility for vector-based data storage and retrieval.
    """
    connections.connect(
        'default', 
        host=config['milvus']['host'], 
        port=config['milvus']['port']
    )

def create_milvus_collection():
    """
    Creates a Milvus collection named `article_vectors` for storing vector embeddings.
    Ensures the collection uses an IVF_FLAT index for efficient similarity searches.
    """
    if utility.has_collection("article_vectors"):
        utility.drop_collection("article_vectors")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="title_vector", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=255)
    ]
    schema = CollectionSchema(fields, description="Oncology Journal Embeddings")
    collection = Collection("article_vectors", schema)
    collection.create_index("title_vector", {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})
    return collection

def extract_keywords(text):
    """
    Extracts keywords from the provided text using NLTK and spaCy.
    Focuses on nouns and proper nouns while removing stopwords.
    """
    stop_words = set(nltk.corpus.stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filtered_tokens = [word.lower() for word in word_tokens if word.isalnum() and word not in stop_words]
    doc = nlp(" ".join(filtered_tokens))
    keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    return ", ".join(list(set(keywords[:10])))

def generate_embeddings(text):
    """
    Generates vector embeddings for a given text using a pre-trained BERT model.
    Returns a 768-dimensional vector representation.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

def crawl_articles():
    """
    Scrapes oncology journal articles from a specified website.
    Extracts metadata such as title, authors, publication date, abstract, and summary.
    """
    url = config['web_scraper']['url']
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to fetch the page.")
        return []

    soup = BeautifulSoup(response.content, "html.parser")
    articles = soup.find_all("article")
    data = []

    for article in articles:
        title_element = article.find("h3", class_="c-card__title")
        title = title_element.get_text(strip=True) if title_element else None
        authors_element = article.find("ul", class_="c-author-list")
        authors = ", ".join([li.text.strip() for li in authors_element.find_all("li")]) if authors_element else None
        date_element = article.find("time")
        publication_date = datetime.strptime(date_element["datetime"], "%Y-%m-%d").date() if date_element else None
        abstract_element = article.find("div", class_="c-card__summary")
        abstract = abstract_element.text.strip() if abstract_element else None

        if title and abstract:
            summary = summarizer(abstract, max_length=50, min_length=25, do_sample=False)[0]["summary_text"]
            keywords = extract_keywords(abstract)
            title_vector = generate_embeddings(title)
            data.append((title, authors, publication_date, abstract, summary, keywords, title_vector))

    return data

def save_data_to_mysql(data):
    """
    Inserts crawled article metadata into the MySQL database.
    Ensures that each record is stored with proper structure and avoids duplicates.
    """
    connection = connect_to_mysql()
    cursor = connection.cursor()
    query = '''
        INSERT INTO oncology_articles (title, authors, publication_date, abstract, summary, keywords)
        VALUES (%s, %s, %s, %s, %s, %s);
    '''
    for record in data:
        cursor.execute(query, record[:-1])  # Exclude vector for MySQL
    connection.commit()
    connection.close()

def save_vectors_to_milvus(data, collection):
    """
    Inserts vector embeddings into the Milvus collection for similarity searches.
    Ensures vectors are stored along with associated article titles.
    """
    milvus_data = [[], []]
    for record in data:
        title_vector = record[-1]
        title = record[0]

        if title_vector and title:
            milvus_data[0].append(title_vector)
            milvus_data[1].append(title)

    if not milvus_data[0]:
        print("No valid data to insert into Milvus.")
        return

    collection.insert(milvus_data)
    collection.flush()
    print("Data successfully inserted into Milvus.")

def query_articles(text_query, collection):
    """
    Performs a similarity search in Milvus using the provided text query.
    Retrieves and displays relevant articles based on vector similarity.
    """
    collection.load()
    vector = generate_embeddings(text_query)
    
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10}
    }

    results = collection.search(
        data=[vector],
        anns_field="title_vector",
        param=search_params,
        limit=5,
        output_fields=['title']
    )
    for result in results[0]:
        print(f"Title: {result.entity.get('title')}, Distance: {result.distance}")

def crawl_and_store_in_batches(batch_size=10):
    """
    Crawls articles and stores them in batches. 
    Each batch is saved to MySQL and Milvus, reducing memory consumption during large crawls.
    """
    global crawled_data  

    oncology_articles = crawl_articles()

    if oncology_articles:
        for i in range(0, len(oncology_articles), batch_size):
            batch = oncology_articles[i:i + batch_size]
            save_data_to_mysql(batch)
            save_vectors_to_milvus(batch, collection)

        crawled_data = oncology_articles  
        print(f"Crawled and stored {len(crawled_data)} articles.")
    else:
        print("No articles found or failed to crawl the data.")



if __name__ == "__main__":
    """
    Main function handles commands provided via the command line. 
    Commands correspond to steps in the shell script and orchestrate the following:
    1. create_mysql_table: Creates the MySQL table.
    2. crawl_and_summarize: Crawls articles, summarizes them with keyword extraction and stores data in Mysql and Milvus.
    3. query_articles: Queries articles based on a provided query.
    """

    if len(sys.argv) < 2:
        print("Usage: python3 main_script.py <command> [arguments]")
        sys.exit(1)

    command = sys.argv[1]

    # Establish Milvus connection here, once
    connect_to_milvus()
    collection = create_milvus_collection()

    if command == "create_mysql_table":
        create_mysql_table()
        print("MySQL table created successfully.")

    elif command == "query_articles":
        crawl_and_store_in_batches(batch_size=int(config['web_scraper']['batch_size'])) 
        query_results = query_articles("Give me the journal those are published last week", collection)

    else:
        print(f"Unknown command: {command}. Please use a valid command.")
        sys.exit(1)