# QA Chatbot

## Overview

The QA Chatbot is designed to answer user queries based on the data provided in an Excel file. It utilizes a vector database to store and retrieve the data. The chatbot also offers APIs to interact with the vector database, including insert, update, and delete operations. It supports multilingual responses, ensuring accuracy and context in both English and other languages.

## Objective

The goal is to create a chatbot capable of responding to questions based on the data in the provided Excel file. The chatbot leverages advanced NLP techniques to generate accurate responses, ensuring users receive relevant information.

## Features

- **Multilingual Support**: The chatbot can respond in multiple languages, translating answers as needed.
- **Data Operations**: APIs to insert, update, and delete data in the vector database.
- **Structured Responses**: Concise, accurate, and relevant responses for all user queries.
- **Retrieval-Augmented Generation (RAG)**: Optionally, fine-tune pre-trained models to answer questions based on the dataset.

## Setup

### 1. Clone the Repository

``` bash
git clone https://github.com/Hirenkhokhar/QA_Chatbot.git
```
### 2 Activate Environment
```bash
conda activate ./{venv_name}
```
### 3. Install Dependencies
   ```bash
   pip install -r requirements.txt
   ```

### 4. **Run the Application**:

   ```bash
   python app.py
   ```
