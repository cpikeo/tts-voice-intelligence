盲听听评说明
1. 按 listen/order.txt 试听；只向评员发 listen/ 与 ratings.csv，不发映射。
2. 在 ratings.csv 中为每个样本打 1–5 分（5 最好），不要修改 sample 列。
   accuracy=发音准确度 | naturalness=自然度 | prosody=韵律 | expressiveness=表现力 | timbre_consistency=音色一致性 | continuity=段间连续性
3. 音色一致性与段间连续性请听完整个样本再评；样本应是完整的段落序列。
4. 评完前不要打开 blind_map.json；评完后运行 unblind 揭示对应关系。
