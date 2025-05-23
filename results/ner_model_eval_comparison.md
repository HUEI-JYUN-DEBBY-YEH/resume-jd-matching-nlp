
# 📊 JD vs Resume NER 模型成效比較報告

## 🧪 測試資料與背景
- 測試資料：自製履歷資料（100 份）與 JD 資料（100 份）
- 模型架構：`spaCy + bert-base-chinese` transformer 模型
- 任務目標：抽取履歷與職缺中的結構化資訊（如職稱、技能、學歷、年資等）

## 📌 成果比較表

| 項目              | JD 模型結果                                      | Resume 模型結果                                  | 阿祖解析與建議                         |
|-------------------|--------------------------------------------------|--------------------------------------------------|----------------------------------------|
| **公司名稱**       | ❌「聯強國際股份有限公司」未抽出                | ❌ 完全未辨識                                     | ✴️ 雙方皆弱，屬泛用詞向缺乏樣本類型     |
| **職稱**           | ✅「資深軟體工程師」被成功抽出（含標點）        | ✘「擔任資」→ 被錯切                              | ✅ JD 模型語意解析穩定，Resume tokenizer 偏切 |
| **技能**           | ✴️「Python」抽出，但標為「加分條件」            | ✴️「Python」包入經驗描述                          | ✴️ 混標明顯，建議 matcher 融合同義欄位 |
| **經驗描述 / 年資** | ✅「5 年」有抽出，但分類偏差                    | ✅「5 年」抽出但標為 [學經歷要求]                | ✴️ 可接受，後處理時合併經驗欄位即可     |
| **證照**           | ✅「Google 資料工程師證照」被抽出               | ❌ 完全未抽出                                     | 📌 Resume樣本偏少，建議強化訓練資料     |
| **教育背景**       | ✅ 國立台灣大學資訊工程學系，標為「必備技能」    | ✅ 同樣辨識出來，但被錯切成三段                  | ✅ 語意有識，span 處理略須修正          |
| **整體表現評估**   | ⭐⭐⭐⭐☆ 已可作為應用 matcher 輸入                | ⭐⭐☆☆☆ 僅作為人工對照、非主流程依據             | ✅ 主推使用 JD 模型，Resume 模型備查    |


---

## 🧭 下一階段：Matcher 設計與應用流程規劃

### ✅ 應用流程圖

```
Resume 原文
   ↓
（可選：Rule-based NER 補強）
   ↓
SBERT 向量表示
   ↓                                   
                 → cosine similarity → 推薦分數
   ↑                                     ↓
JD 關鍵詞（來自 JD NER 模型）
   ↓
TF-IDF / BM25 關鍵詞表示
```

### 🧩 模組與步驟

| 步驟 | 模組名稱         | 功能描述 |
|------|------------------|----------|
| 1    | `jd_ner_extractor.py` | 將 JD 文件轉為結構化關鍵詞 |
| 2    | `resume_embedder.py` | 將 Resume 原文轉為 SBERT 向量 |
| 3    | `matcher.py`     | 結合關鍵詞 / 向量進行匹配與打分 |
| 4    | `evaluator.py`   | 使用 Precision@k, MAP, NDCG 等評估指標 |

---

