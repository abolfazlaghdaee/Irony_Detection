# Irony_Detection

This project is dedicated to irony detection in Persian tweets, utilizing advanced Natural Language Processing (NLP) techniques. The dataset for this research was gathered from X (formerly Twitter) and uniquely annotated based on the reactions to each tweet. This approach allows the project to delve into how audience reactions can serve as implicit signals for identifying ironic statements within the often nuanced and context-dependent nature of social media communication. The goal is to develop robust models capable of distinguishing ironic from non-ironic text, which is crucial for improving downstream NLP tasks like sentiment analysis and opinion mining, especially in a morphologically rich language like Persian.


### Methodology Highlights
To capture the semantic and contextual nuances of Persian tweets for irony detection, we utilize three distinct embedding techniques:

**TF-IDF (Term Frequency-Inverse Document Frequency):**  classic statistical measure reflecting how important a word is to a document in a collection or corpus.

**FastText:** An open-source, free library from Meta AI for efficient learning of word representations and text classification. It's particularly effective for languages with rich morphology like Persian as it considers character n-grams.

**ParsBERT:** A state-of-the-art transformer-based model pre-trained specifically on a large corpus of Persian text. As a BERT-like model, it captures deep contextual relationships, which is highly beneficial for understanding complex linguistic phenomena like irony.


### Modeling Approach
Our project explores a range of neural network architectures to effectively learn and classify ironic patterns from Persian tweet data. We have experimented with both recurrent neural networks (RNNs) and advanced transformer-based models:

**Recurrent Neural Network (RNN) Based Models**

We utilized various architectures from the RNN family, which are well-suited for sequential data like text, allowing them to capture dependencies across words in a tweet:

- RNN (Recurrent Neural Network): Basic recurrent layers to process sequences.

- GRU (Gated Recurrent Unit): A lighter alternative to LSTMs, designed to capture long-range dependencies more effectively.

- LSTM (Long Short-Term Memory): A powerful variant of RNNs capable of learning long-term dependencies, crucial for understanding complex sentence structures and context in irony.

**Transformer-Based Models**

Leveraging the recent advancements in self-attention mechanisms, we fine-tuned large pre-trained transformer models, which excel at understanding deep contextual relationships within text:

- ParsBERT: A BERT-like model specifically pre-trained on a massive corpus of Persian text. Its pre-training on a relevant language corpus provides a strong foundation for understanding the nuances of Persian grammar, semantics, and idiomatic expressions.

- XLM-RoBERTa Large (XLM-R Large): A multilingual transformer model pre-trained on text in 100 languages. Its vast pre-training data allows it to capture cross-lingual and general linguistic patterns, which can be highly beneficial for a global understanding of text, including complex sentiments like irony in Persian.

### Results
| Model / Embedding | Accuracy | F1-Score | Precision |
|-------------------|:--------:|:--------:|:---------:|
| RNN | 0.62 | 0.60| 0.59 | 
|  LSTM             | 0.69     | 0.68     | 0.63      |
| GRU    | 0.81     | 0.80     | 0.82     | 0.78     | 
| ParsBERT          | 0.802793	     | 0.802674| 0.802868      | 
| XLM-R             | 0.800409    | 0.800657    |0.802546      | 


### License
This project is licensed under the MIT License.