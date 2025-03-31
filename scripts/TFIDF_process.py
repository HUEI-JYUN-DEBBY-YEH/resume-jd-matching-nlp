import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 讀取合併後的履歷資料 CSV 檔案（注意檔案路徑根據實際情況修改）
input_file = r'D:\NLP_Resume_Parsing\data\processed\all_JD_for_vectorization.csv'
df = pd.read_csv(input_file, encoding='utf-8-sig')

# 確認 text_for_NLP 欄位是否存在，這是後續向量化的文本內容
if 'text_for_NLP' not in df.columns:
    raise ValueError("找不到 'text_for_NLP' 欄位，請檢查資料。")

# 將每一筆資料的文本取出 (每一行代表一筆資料，獨立向量化)
texts = df['text_for_NLP'].tolist()

# 1. TF-IDF 向量化
vectorizer = TfidfVectorizer(max_features=500)  # 可根據需要調整 max_features
tfidf_matrix = vectorizer.fit_transform(texts)
print("TF-IDF 矩陣形狀:", tfidf_matrix.shape)

# 2. KMeans 聚類分析
num_clusters = 5  # 可以根據資料特性調整聚類數量
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(tfidf_matrix)

# 將聚類結果 (cluster labels) 加入原始 DataFrame
df['cluster'] = kmeans.labels_

# 3. 聚類效果評估 (使用 Silhouette Score)
score = silhouette_score(tfidf_matrix, kmeans.labels_)
print("Silhouette Score:", score)

# 4. 儲存聚類結果，方便後續檢查與分析
output_file = r'D:\NLP_Resume_Parsing\data\processed\all_JDs_clustered.csv'
df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"聚類結果已儲存至 {output_file}")
