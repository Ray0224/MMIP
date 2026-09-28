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

