# 📌 Resume–JD Semantic Matching (NLP-Based System)

This project simulates an intelligent talent-matching engine using a combination of **Named Entity Recognition (NER)** and **semantic similarity** techniques. It supports extracting structured information from resumes and job descriptions (JDs) and computes relevance scores to rank top-matching candidates or positions.

## 🔍 Why this matters

Recruitment workflows often rely on manual screening of resumes and keyword-based search, which limits efficiency and often overlooks candidate–job fit beyond surface-level terms. By applying NLP techniques, this project enhances the matching process with a more **context-aware**, **semantic**, and **scalable** approach — especially valuable in high-volume hiring or talent recommendation platforms.

---

## 🧭 System Overview

```mermaid
flowchart TD
    A[Input: Raw Resume + JD] --> B[NER Extraction (spaCy Transformer)]
    B --> C[Entity Standardization]
    C --> D[TF-IDF / SBERT Embedding]
    D --> E[Cosine Similarity Score]
    E --> F[Top-K Matching Output]
```
![Architecture Diagram](./resume_jd_matching.png)
---

## ⚙️ Key Components

* **NER Extraction**: Fine-tuned `spaCy` Transformer model identifies and labels key attributes in resumes and JDs, such as skills, education, experience, and certifications.

* **Entity Normalization**: Harmonizes variations (e.g., “Python dev” vs. “Python engineer”) to improve downstream semantic comparison.

* **Semantic Embeddings**:

  * **TF-IDF**: Captures term-level statistical features.
  * **SBERT (Sentence-BERT)**: Captures contextual similarity between resume and JD sentences.

* **Ranking Logic**: Combines embedding-based similarity scores and optional keyword weights to produce a Top-K recommendation list.

---

## 🧪 Project Modules

| Script                          | Description                                               |
| ------------------------------- | --------------------------------------------------------- |
| `scripts/ner_predictor.py`      | Extracts structured entities from resumes/JDs using spaCy |
| `scripts/similarity_matcher.py` | Computes semantic similarity between parsed inputs        |
| `scripts/eval_metrics.py`       | Evaluates matching quality (Precision\@K, etc.)           |

---

## 📂 Folder Structure

```bash
resume-jd-matching-nlp/
├── data/
│   └── sample/             # Sample resume & JD files
├── scripts/
│   ├── ner_predictor.py    # spaCy-based NER pipeline
│   ├── similarity_matcher.py
│   └── eval_metrics.py
├── results/
│   └── output/             # Matching output & score files
├── configs/
│   └── spacy_config.cfg    # Fine-tune config
└── README.md
```

---

## 📊 Sample Output

| Resume        | JD                | Top Match | Score |
| ------------- | ----------------- | --------- | ----- |
| resume\_A.txt | jd\_marketing.txt | ✅         | 0.91  |
| resume\_B.txt | jd\_software.txt  | ❌         | 0.42  |

---

## 💡 Use Cases

* Internal HR recommendation engine
* Candidate–job matching for recruiting platforms
* JD benchmarking and similarity clustering

---

## 🛠 Tech Stack

* Python 3.10+
* spaCy (Transformer NER)
* Sentence-BERT (`sentence-transformers`)
* scikit-learn (TF-IDF, cosine similarity)

---

## 🚀 Future Directions

* Add interactive web interface (Flask or Streamlit)
* Enable bi-directional matching (JD → Resume / Resume → JD)
* Explore cross-lingual matching support
* Integrate recruiter feedback loop to refine scoring

---

## ✍️ Author

**Debby Yeh**
NLP Application Engineer in Transition
📌 Focus: AI for HR, legal NLP, retrieval-based systems
🔗 [Portfolio (Notion)](https://mango-mapusaurus-5df.notion.site/Debby-Yeh-Portfolio-1ca5118474d2801caa58de564fb53e38)

