import matplotlib.pyplot as plt

# 定義方法與對應的 Silhouette Score
methods = ['TF-IDF (Resume)', 'Sentence-BERT (Resume)', 'TF-IDF (JD)', 'Sentence-BERT (JD)']
scores = [0.0305, 0.0783, 0.0410, 0.1142]

plt.figure(figsize=(8, 6))
bars = plt.bar(methods, scores, color=['skyblue', 'salmon', 'skyblue', 'salmon'])
plt.title("Silhouette Score 比較")
plt.ylabel("Silhouette Score")
plt.ylim(0, 0.15)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f'{yval:.3f}', ha='center', va='bottom')
plt.show()
