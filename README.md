# Laws-prediction--Top-k-retreival

This project predicts the laws applicable for given case document input by retreiving top k similar cases.
First the textual data is converted into word embeddings using sentence_transformers' 'all-MiniLM-L6-v2' model. Data also includes information like language, law_area which are one hot encoded. Theese data are concatenated and used to create a vectordatabase.
Used 2 different vector databases, vilvus and faiss for retreival.

To improve the quality of word embeddings generated and decrease the compute needed & the time for processing, the textual data is sent into nlp_pattern.ipynb which is a data pre processing pipeline that uses basic nlp methods to decrease the number of tokens while conserving most of the important information.

This projects allows the judicial officials to get the laws applicable for given case straight away.
