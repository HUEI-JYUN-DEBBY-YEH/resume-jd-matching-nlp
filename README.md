# 📌 NLP Resume-JD Semantic Matching 專案成果展示

## 📘 專案簡介
本專案旨在模擬真實人力媒合系統，透過自動生成之履歷與職缺資料進行命名實體辨識（NER）訓練，並結合語意嵌入與相似度評估技術，達成履歷與職缺間的智能匹配推薦。

---

## 📌 模型訓練成果摘要

### 🔹 JD NER 模型表現
| 指標       | 數值   |
|------------|--------|
| ENTS_F     | 0.51   |
| ENTS_P     | 0.76   |
| ENTS_R     | 0.38   |
| 模型       | model-best (transformer fine-tuned) |

### 🔹 Resume NER 模型表現
| 指標       | 數值   |
|------------|--------|
| ENTS_F     | 0.30   |
| ENTS_P     | 0.43   |
| ENTS_R     | 0.23   |
| 模型       | model-best (transformer fine-tuned) |

---

## 📌 語意匹配推薦系統評估指標 (Top-3)

### 🔹 Recommendation Evaluation
| 評估指標     | 數值   |
|--------------|--------|
| Precision@3  | 0.410  |
| MAP@3        | 0.367  |
| NDCG@3       | 0.549  |

- 評估方式：對每個 JD 推薦前 3 筆履歷，並依人工標記結果計算上述指標。
- 工具腳本：`eval_metrics.py`

---

## 📌 技術亮點摘要

- 🔧 **SpaCy Transformer 微調 NER 模型**
- 🧠 **SBERT 語意嵌入產生語句向量**
- 📊 **KMeans 聚類評估語意結構品質**
- 📈 **TF-IDF / BM25 / SBERT 三法匹配比較**
- 💡 **具可解釋性的履歷推薦邏輯**

---

## 📁 推薦結果範例（JD ➜ Top-3 Resumes）

📄 *可參考檔案：* `top_k_recommendations.csv`

| JD內容摘要                      | Top1履歷         | Top2履歷         | Top3履歷         |
|--------------------------------|------------------|------------------|------------------|
| Backend工程師需熟Python與API設計 | 張O倫（Python）  | 林O妤（Flask）   | 陳O宇（Django）  |

---

## 🧠 未來可擴充方向
- 加入更多履歷/職缺數據擴大訓練集
- 結合 LLM 模型實現語意問答或推薦解釋
- 前後端部署可視化推薦平台（Streamlit + FastAPI）

---
作者：Debby Yeh｜專案完成日：2025年3月