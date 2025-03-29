# Laws-prediction--Top-k-retreival

This project predicts the laws applicable for given case document input by retreiving top k similar cases.
First the textual data is converted into word embeddings using sentence_transformers' 'all-MiniLM-L6-v2' model. Data also includes information like language, law_area which are one hot encoded. Theese data are concatenated and used to create a vectordatabase.
Used 2 different vector databases, vilvus and faiss for retreival.

This projects allows the judicial officials to get the laws applicable for given case straight away.
