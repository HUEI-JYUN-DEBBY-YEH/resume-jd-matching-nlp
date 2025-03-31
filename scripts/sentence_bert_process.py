import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 讀取已處理好的資料 (例如：履歷)
input_file = r'D:\NLP_Resume_Parsing\data\processed\all_JD_for_vectorization.csv'
df = pd.read_csv(input_file, encoding='utf-8-sig')

# 確認有 text_for_NLP 欄位，作為向量化文本來源
if 'text_for_NLP' not in df.columns:
    raise ValueError("找不到 'text_for_NLP' 欄位，請檢查資料格式。")

# 將每筆文本提取出來 (每行一筆資料)
texts = df['text_for_NLP'].tolist()

# 使用 Sentence-BERT 模型進行向量化
model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
embeddings = model.encode(texts)

print("Sentence-BERT embeddings shape:", embeddings.shape)  # (100, 768) 例如

# 使用 KMeans 進行聚類分析
num_clusters = 5  # 可根據資料特性調整
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(embeddings)
cluster_labels = kmeans.labels_

# 將聚類結果加入原始資料中
df['cluster'] = cluster_labels

# 計算聚類效果評估 (Silhouette Score)
score = silhouette_score(embeddings, cluster_labels)
print("Silhouette Score:", score)

# 儲存聚類結果，方便後續檢視
output_file = r'D:\NLP_Resume_Parsing\data\processed\all_JDs_sentenceBERT_clustered.csv'
df.to_csv(output_file, index=False, encoding='utf-8-sig')
print("聚類結果已儲存至", output_file)
