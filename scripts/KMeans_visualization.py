import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib

# 設定 matplotlib 使用中文字體（例如SimHei）
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字體
matplotlib.rcParams['axes.unicode_minus'] = False    # 正確顯示負號

# 設定要讀取的JD資料檔案路徑 (請根據實際情況修改)
input_file = r'D:\NLP_Resume_Parsing\data\processed\all_resumes_sentenceBERT_clustered.csv'

# 讀取 CSV 檔案，並檢查必要欄位
df = pd.read_csv(input_file, encoding='utf-8-sig')
if 'text_for_NLP' not in df.columns or 'cluster' not in df.columns:
    raise ValueError("檔案中缺少 'text_for_NLP' 或 'cluster' 欄位，請確認資料格式。")

# 將每筆文本提取出來
texts = df['text_for_NLP'].tolist()

# 使用 Sentence-BERT 模型進行向量化
model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
embeddings = model.encode(texts)
print("Sentence-BERT embeddings shape:", embeddings.shape)  # 應為 (100, 512)

# 可選：計算 Silhouette Score 驗證聚類效果
score = silhouette_score(embeddings, df['cluster'])
print("Silhouette Score:", score)

# 使用 t-SNE 將高維向量降到 2 維
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# 產生散點圖
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                      c=df['cluster'], cmap='viridis', alpha=0.7)
plt.title("t-SNE 降維後的 resumes Sentence-BERT 聚類結果")
plt.xlabel("t-SNE 第一維")
plt.ylabel("t-SNE 第二維")
plt.colorbar(scatter, label='聚類標籤')
plt.show()
