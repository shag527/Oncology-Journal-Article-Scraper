#!/bin/bash

# Step 1: Create MySQL Table
echo "Creating MySQL table..."
python3 main.py create_mysql_table

# Step 2: Crawl Articles, Summarize and store Data in MySQL and Milvus. Also Query Articles
python3 main.py query_articles "Give me the journal those are published last week"

echo "Process completed successfully."
