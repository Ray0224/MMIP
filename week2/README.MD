Quiz1

Logistic Regression
指標	Threshold = 0.5	Threshold = 0.7
TN              41	        41
FP	            1	        1
FN	            1	        5
TP	            71	        67
Accuracy	    98.25%	    94.74%
Precision	    98.61%	    98.53%
Recall	        98.61%	    93.06%
F1-Score	    98.61%	    95.71%

Random Forest
指標	Threshold = 0.5	Threshold = 0.7
TN	            39	        42
FP	            3	        0
FN	            2	        8
TP	            70	        64
Accuracy	    95.61%	    92.98%
Precision	    95.89%	    100.00%
Recall	        97.22%	    88.89%
F1-Score	    96.55%	    94.12%

Threshold = 0.5之比較
指標	Logistic Regression	Random Forest
Accuracy	        98.25%	95.61%
Precision	        98.61%	95.89%
Recall	            98.61%	97.22%
F1-Score	        98.61%	96.55%

本實驗使用 Logistic Regression 與 Random Forest 兩種分類模型，並使用相同的 Validation Dataset 進行比較。在 Threshold = 0.5 的情況下，Logistic Regression 的 Accuracy、Precision、Recall 與 F1-Score 分別為 98.25%、98.61%、98.61% 與 98.61%；Random Forest 則分別為 95.61%、95.89%、97.22% 與 96.55%。在 Precision 方面，Logistic Regression 的結果較高，主要是因為其 False Positive 為 1，而 Random Forest 的 False Positive 為 3。在 Recall 方面，Logistic Regression 同樣較高，其 False Negative 為 1，而 Random Forest 為 2。錯誤類型方面，Logistic Regression 的 FP 與 FN 各為 1；Random Forest 則為 FP = 3、FN = 2，因此 Random Forest 的 False Positive 較多。


Quiz2
1. 超參數設定

本實驗使用 MLP 進行信用卡違約預測，模型架構及超參數如下：

項目	            設定
Epoch	            50
Batch Size	        32
Learning Rate	    0.001
Optimizer	        Adam
Loss Function	    Binary Crossentropy
Hidden Layer	    2 層
Neuron 數量	        64 → 32
Activation Function	ReLU
Output	            1 Neuron + Sigmoid
Dropout	            0.3（改善模型）

23 Features → 64 → 32 → 1
23 Features → 64 → Dropout(0.3) → 32 → Dropout(0.3) → 1

2. 模型改善前後比較
2.1 Training / Validation 結果
指標	原始 MLP	Dropout MLP
Training Loss	0.3864	0.4235
Validation Loss	0.4599	0.4351
Training Accuracy	83.42%	82.15%
Validation Accuracy	80.65%	81.92%

加入 Dropout 後，Training Loss 上升，但 Validation Loss 由 0.4599 降至 0.4351；Validation Accuracy 則由 80.65% 提升至 81.92%。

Training Accuracy 與 Validation Accuracy 的差距：

原始 MLP：83.42% − 80.65% = 2.77%
Dropout MLP：82.15% − 81.92% = 0.23%

因此可以觀察到，加入 Dropout 後 Training 與 Validation 的表現差距縮小，過擬合現象有所改善，模型在 Validation Dataset 上的表現也較穩定。

2.2 Classification Metrics
指標	原始 MLP	Dropout MLP
Accuracy	80.65%	81.92%
Precision	59.98%	68.06%
Recall	37.60%	34.36%
F1-Score	46.23%	45.67%

加入 Dropout 後，Accuracy 與 Precision 提升，但 Recall 與 F1-Score 略微下降。因此 Dropout 並非讓所有評估指標都提升，而是在降低過擬合的同時，使不同指標之間產生不同的變化。
另外，進一步測試 Classification Threshold 後，Threshold = 0.30 時：
Accuracy  = 79.58%
Precision = 53.99%
Recall    = 52.00%
F1-Score  = 52.98%
相較於 Dropout 模型原本 Threshold = 0.5 的 F1-Score 45.67%，提升至 52.98%，顯示 Threshold 的選擇會明顯影響 Precision、Recall 與 F1-Score。
