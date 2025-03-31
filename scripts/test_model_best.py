import spacy

# ❶ 載入訓練完成的模型
nlp = spacy.load("D:/NLP_Resume_Parsing/models/resume_model/transformer_ner_weighted/model-best")

# ❷ 測試文本（可替換為任一 Resume 或 JD 原文）
test_text = """
王小明曾任職於聯強國際股份有限公司，擔任資深軟體工程師，具備 Python、機器學習開發經驗超過 5 年，並持有 Google 資料工程師證照，畢業於國立台灣大學資訊工程學系。
"""

# ❸ 進行推論
doc = nlp(test_text)

# ❹ 輸出結果
print("📄 原文:")
print(test_text)
print("\n🔍 NER 模型抽取結果:")

for ent in doc.ents:
    print(f"[{ent.label_}] {ent.text}")
