# AI_term_project
【人工智慧】期末作業

> 可以在 Github 上面[查看本專案](https://github.com/coke5151/114-1-ai-hw-submit)

## 開發環境
1. 初次使用請先同步環境（電腦裡沒 uv 的[去安裝一下](https://docs.astral.sh/uv/guides/install-python/)）

```bash
uv sync
```

2. 後續開發：

```bash
uv run python main.py # 執行 main.py 檔案
uv run jupyterlab # 開啟 Jupyter Notebook
```

## 下載資料集
前往 [Google Drive](https://drive.google.com/drive/folders/1i6mAZvKNhrwkA6QRkaZcScXeyADqqmnZ) 下載資料集，解壓縮後放在 `data` 資料夾中。

data 資料夾應包含以下資料：
- `task1_dataset_kotae.csv` # 可由 `CityA Ground Truth Data.csv.gz` 解壓縮得到

---

## Jupyter Notebooks 說明

本專案的 Jupyter Notebook 分為以下幾類：

### 資料探索與基準模型
| 檔案 | 說明 |
|------|------|
| [model1.ipynb](notebooks/model1.ipynb) | 資料探索分析，計算人流統計與視覺化 |
| [model2.ipynb](notebooks/model2.ipynb) | 進階資料分析與人流預測模型開發 |
| [baseline_moving_average.ipynb](notebooks/baseline_moving_average.ipynb) | **Baseline 模型**：使用移動平均法 (Moving Average) 進行人流預測 |

### Seq2Seq 時間序列預測模型
| 檔案 | 說明 |
|------|------|
| [train_multivariate_seq2seq.ipynb](notebooks/train_multivariate_seq2seq.ipynb) | **多變數 Seq2Seq 模型**：結合多地點資料與週末標籤進行預測 |
| [train_multivariate_timeperiod_seq2seq.ipynb](notebooks/train_multivariate_timeperiod_seq2seq.ipynb) | **多變數時段 Seq2Seq 模型**：加入 One-Hot Encoding 的時段特徵 (早/午/晚/深夜) |
| [hyperparameter_experiment.ipynb](notebooks/hyperparameter_experiment.ipynb) | **超參數實驗**：Grid Search 測試不同 hidden_size、num_layers、learning_rate 組合 |

### 分類模型 (週間/週末判別)
| 檔案 | 說明 |
|------|------|
| [auto_label_weekend_and_train_dnn.ipynb](notebooks/auto_label_weekend_and_train_dnn.ipynb) | **K-Means 自動標籤 + DNN 分類器**：使用非監督學習自動標記週末，並訓練 DNN 分類模型 |
| [model2_dnn_cnn_comparison.ipynb](notebooks/model2_dnn_cnn_comparison.ipynb) | **DNN vs CNN 比較**：比較深度神經網路與卷積神經網路在時段分類任務的表現 |

### 其他
| 檔案 | 說明 |
|------|------|
| [test_script.ipynb](notebooks/test_script.ipynb) | 測試用腳本，可查看當前環境是否能調用到 CUDA |

---

## Models 檔案說明

訓練完成的模型與相關輸出檔案：

### 預測模型 (Seq2Seq)
| 檔案 | 說明 |
|------|------|
| [seq2seq_model.pth](models/seq2seq_model.pth) | 單變數 Seq2Seq 模型權重 |
| [seq2seq_multivariate.pth](models/seq2seq_multivariate.pth) | 多變數 Seq2Seq 模型權重 |
| [seq2seq_multivariate_timeperiod.pth](models/seq2seq_multivariate_timeperiod.pth) | 多變數 + 時段特徵 Seq2Seq 模型權重 |

### 分類模型
| 檔案 | 說明 |
|------|------|
| [dnn_time_classifier.pth](models/dnn_time_classifier.pth) | DNN 時段分類器權重 |
| [cnn_time_classifier.pth](models/cnn_time_classifier.pth) | CNN 時段分類器權重 |

### 資料預處理
| 檔案 | 說明 |
|------|------|
| [scaler.pkl](models/scaler.pkl) | 單變數模型的 MinMaxScaler |
| [scaler_multivariate.pkl](models/scaler_multivariate.pkl) | 多變數模型的 MinMaxScaler |
| [scaler_multivariate_timeperiod.pkl](models/scaler_multivariate_timeperiod.pkl) | 多變數時段模型的 MinMaxScaler |

### 評估結果
| 檔案 | 說明 |
|------|------|
| [eval_log_baseline_ma.txt](models/eval_log_baseline_ma.txt) | Baseline (Moving Average) 評估指標 |
| [eval_log_multivariate.txt](models/eval_log_multivariate.txt) | 多變數 Seq2Seq 評估指標 |
| [eval_log_multivariate_timeperiod.txt](models/eval_log_multivariate_timeperiod.txt) | 多變數時段 Seq2Seq 評估指標 |

### 視覺化結果
| 檔案 | 說明 |
|------|------|
| [prediction_result_baseline_ma.png](models/prediction_result_baseline_ma.png) | Baseline 模型預測結果圖 |
| [prediction_result_multivariate.png](models/prediction_result_multivariate.png) | 多變數模型預測結果圖 |
| [prediction_result_multivariate_timeperiod.png](models/prediction_result_multivariate_timeperiod.png) | 多變數時段模型預測結果圖 |
| [training_loss_univariate.png](models/training_loss_univariate.png) | 單變數模型訓練 Loss 曲線 |
| [confusion_matrix_dnn.png](models/confusion_matrix_dnn.png) | DNN 分類器混淆矩陣 |
| [confusion_matrix_dnn_cnn.png](models/confusion_matrix_dnn_cnn.png) | DNN vs CNN 混淆矩陣比較 |
| [dnn_architecture_comparison.png](models/dnn_architecture_comparison.png) | DNN 架構比較圖 |
| [dnn_cnn_training_comparison.png](models/dnn_cnn_training_comparison.png) | DNN vs CNN 訓練過程比較 |
| [dnn_cnn_validation_comparison.png](models/dnn_cnn_validation_comparison.png) | DNN vs CNN 驗證結果比較 |

### 超參數實驗結果
| 檔案 | 說明 |
|------|------|
| [hyperparameter_results/](models/hyperparameter_results/) | 超參數搜尋結果資料夾 |
| [hyperparameter_results/hyperparameter_results.csv](models/hyperparameter_results/hyperparameter_results.csv) | 各超參數組合的實驗結果 |
| [hyperparameter_results/best_hyperparameters.txt](models/hyperparameter_results/best_hyperparameters.txt) | 最佳超參數配置 |
| [hyperparameter_results/hyperparameter_comparison.png](models/hyperparameter_results/hyperparameter_comparison.png) | 超參數比較視覺化圖表 |
