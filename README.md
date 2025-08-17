# Content-Based-Recommender-System-using-NLP
Implements a Content Based Anime Recommender System using NLP techniques like **Stopwords Removal**,**Lemmatization** as preprocessing; **Count Vectorizer** and **Cosine Similarity** to grab **Top-N Rankings**. 
# Anime Recommender System (Content-Based Filtering)

## 📌 Overview

This project implements a **content-based anime recommender system**. The system suggests anime titles that are similar to a given input by analyzing their textual metadata (such as genres, synopsis, and tags). Instead of relying on user ratings or collaborative filtering, the model focuses on the content of the anime itself to make recommendations.

The notebook walks through the entire pipeline, from preprocessing text to computing similarity scores and generating recommendations.

---

## ⚙️ Techniques Used

* **Natural Language Processing (NLP)**

  * **Tokenization**: Splitting text into individual words.
  * **Stopword Removal**: Removing common words that add little meaning (e.g., "the", "and").
  * **Lemmatization**: Reducing words to their root form for normalization.

* **Feature Extraction**

  * **CountVectorizer**: Transforms text into a sparse matrix of token counts.

* **Similarity Computation**

  * **Cosine Similarity**: Measures the angle between vectors to determine how similar two anime descriptions are.

* **Recommendation Logic**

  * For a given anime, the system ranks and retrieves the top N most similar anime based on content.

---

## 🛠 Tech Stack

* **Programming Language**: Python 3
* **Libraries & Tools**:

  * `pandas` → Data manipulation
  * `numpy` → Numerical operations
  * `scikit-learn` → Vectorization & similarity computations
  * `nltk` / `spacy` → Text preprocessing (stopwords, lemmatization)
  * `kaggle` to import the necessary dataset.

---

## 🚀 Workflow

1. **Load Dataset** → Import anime dataset with metadata (titles, genres, synopsis, etc.).
2. **Preprocess Text** → Clean and normalize text for feature extraction.
3. **Vectorize Data** → Use CountVectorizer/TF-IDF to convert text into numerical vectors.
4. **Compute Similarity Matrix** → Calculate cosine similarity across all anime.
5. **Generate Recommendations** → Return the most relevant anime for a given input.

---

## 📖 Example

* Input: *"Naruto"*
* Output: Similar recommendations like *"Bleach"*, *"One Piece"*, *"Fairy Tail"*, etc.

---

## 🔮 Future Enhancements

* Combine with collaborative filtering for hybrid recommendations.
* Use advanced embeddings (Word2Vec, BERT) for semantic similarity.
* Deploy as a **web application** using Streamlit or Flask for interactive usage.
