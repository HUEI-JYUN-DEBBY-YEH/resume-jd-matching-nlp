import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from pathlib import Path

# === Resume-JD 相似度計算 ===
JD_KEYWORD_PATH = Path("D:/NLP_Resume_Parsing/data/processed/jd_keywords/all_jd_keywords.csv")
RESUME_EMBED_PATH = Path("D:/NLP_Resume_Parsing/data/processed/resume_embeddings/resume_embeddings.npy")
RESUME_META_PATH = RESUME_EMBED_PATH.with_suffix(".csv")
OUTPUT_PATH = Path("D:/NLP_Resume_Parsing/data/processed/similarity/top_k_matches.csv")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# 載入語意模型（與嵌入同款）
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# 載入資料
jd_df = pd.read_csv(JD_KEYWORD_PATH)
resume_embeddings = np.load(RESUME_EMBED_PATH)
resume_meta = pd.read_csv(RESUME_META_PATH)

# 計算每份 JD 的嵌入
jd_texts = jd_df["抽取關鍵詞"].fillna("").tolist()
jd_embeddings = model.encode(jd_texts, batch_size=16, show_progress_bar=True)

# 計算相似度矩陣
cosine_scores = util.cos_sim(jd_embeddings, resume_embeddings).numpy()  # shape: (num_jd, num_resume)

# 取得 top-k resume for each JD
top_k = 3
match_results = []

for jd_idx, jd_row in jd_df.iterrows():
    top_indices = np.argsort(cosine_scores[jd_idx])[::-1][:top_k]
    for rank, resume_idx in enumerate(top_indices):
        match_results.append({
            "JD檔案": jd_row["檔案"],
            "JD索引": jd_row["索引"],
            "JD原文": jd_row["原文"],
            "Resume檔案": resume_meta.loc[resume_idx, "檔案"],
            "Resume索引": resume_meta.loc[resume_idx, "索引"],
            "Resume原文": resume_meta.loc[resume_idx, "原文"],
            "相似度": cosine_scores[jd_idx][resume_idx],
            "排名": rank + 1
        })

# 儲存匹配結果
match_df = pd.DataFrame(match_results)
match_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print(f"✅ 匹配完成，共 {len(jd_df)} 筆 JD x Top-{top_k} Resume => {len(match_df)} 筆推薦結果")
