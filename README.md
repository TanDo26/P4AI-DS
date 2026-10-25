# Nhóm DL — FT-Transformer & TabNet cho dự đoán giá nhà

Hiện thực [`implement_plan.md`](implement_plan.md) (code trong [`dl/`](dl/)): 2 model DL (FT-Transformer, TabNet) trên Housing (545 dòng) và King County (21.613 dòng), cùng split/fold với nhóm ML để so sánh công bằng.

## Cài đặt

```bash
conda create -n house-dl python=3.12 -y          # 3.14 cũng được: mọi version đã pin đều có wheel cp314
conda activate house-dl
cd dl
# cài torch TRƯỚC — wheel torch trên PyPI cho Windows chỉ là bản CPU
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu     # CPU
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126   # GPU NVIDIA (vd. RTX 4060)
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

Nếu `nvidia-smi` báo `CUDA Version` ≥ 13.0, có thể dùng `cu130` thay cho `cu126`. Có GPU thì chạy King County ngay tại máy với `--device cuda`, không cần Kaggle.

Hai file CSV gốc (`Housing.csv`, `kc_house_data.csv`) mặc định nằm ở thư mục cha của `dl/`. Nếu để chỗ khác, đặt biến môi trường `HOUSE_DATA_DIR`.

## Cách chạy (từ thư mục `dl/`)

| Lệnh | Việc |
|---|---|
| `python run.py splits` | Tạo split 80/20, 5 fold, tập con cho learning curve (chạy 1 lần; giữ nguyên nếu đã có) |
| `python run.py tune --dataset housing --model ft_transformer --n-trials 50` | Optuna 5-fold CV; trial 0 là cấu hình mặc định; chạy tiếp được nếu bị ngắt |
| `python run.py final --dataset housing --model ft_transformer` | Train lại trên toàn bộ train, đánh giá test **1 lần**, giải thích model, vẽ hình |
| `python run.py all --dataset housing` | `tune` + `final` cho cả 2 model |
| `python run.py ablation --dataset housing` | Ablation preprocessing (cấu hình mặc định, CV) |
| `python run.py size-curve` | Learning curve theo số dòng train trên King County |
| `python run.py summary [--ml-results ml.csv]` | Bảng tổng hợp + hình so sánh + phân tích sai số trên test; thêm kết quả nhóm ML để tính gap |
| `python run.py replot` | Vẽ lại hình của từng model từ kết quả đã lưu (không train lại) |

Tuỳ chọn chung: `--device auto|cpu|cuda`, `--threads N`.
Trên CPU nên dùng `--threads 1` và chạy nhiều job song song: với model nhỏ, thêm luồng làm **chậm** đi (đo được: FT-Transformer trên Housing 0,54 s/epoch với 1 luồng, 1,2 s/epoch với 8 luồng).

**King County cần GPU.** Trên CPU, FT-Transformer mất ~46 s/epoch và TabNet ~6 s/epoch, tức hàng giờ cho một trial. Dùng [`dl/notebooks/kaggle_kc_gpu.ipynb`](dl/notebooks/kaggle_kc_gpu.ipynb), rồi giải nén `dl_results_kc.zip` vào `dl/`.

## Cấu trúc

```
dl/
├── run.py                     # CLI
├── src/
│   ├── config.py              # đường dẫn, SEED_DATA=42, SEED_MODEL=36
│   ├── data.py                # đọc CSV + feature engineering stateless
│   ├── splits.py              # split / fold / inner split / tập con learning curve
│   ├── preprocess.py          # TabularPreprocessor, TargetTransformer (stateful) + preprocessing.md
│   ├── metrics.py             # MAE, RMSE, R², NRMSE, bootstrap CI
│   ├── cv.py                  # 5-fold CV với inner_val cho early stopping
│   ├── tune.py                # Optuna (TPE seed 42, MedianPruner, SQLite để resume)
│   ├── final.py               # model cuối + test (module DUY NHẤT đọc test split)
│   ├── explain.py             # permutation importance, attention [CLS], mask TabNet
│   ├── plots.py, summary.py
│   └── models/                # ft_transformer.py, tabnet.py (cùng interface BaseModel)
├── experiments/               # ablation.py, size_curve.py
├── tests/                     # test_no_leakage.py, test_smoke.py, test_pipeline.py
├── notebooks/kaggle_kc_gpu.ipynb
├── splits/                    # dùng chung với nhóm ML
├── results/<dataset>/<model>/ # kết quả từng model (xem dưới)
└── figures/                   # hình tổng hợp, ablation, learning curve theo kích thước
```

## Kết quả của mỗi model × dataset (`results/<dataset>/<model>/`)

| File | Nội dung |
|---|---|
| `best_params.json` | Hyperparams tốt nhất, CV mean ± std, số epoch cho model cuối, thông tin Optuna, seed, version thư viện, phần cứng |
| `optuna_trials.csv`, `optuna.db` | Toàn bộ trial (db dùng để chạy tiếp) |
| `cv_folds.csv`, `cv_histories.json` | Metric và best epoch từng fold; learning curve từng fold của trial tốt nhất |
| `test_metrics.json` | MAE, RMSE, R², NRMSE trên test + 95% CI bootstrap, `fit_time`, `predict_time` |
| `predictions_test.csv` | `row, y_true, y_pred` |
| `preprocessing.md` | Các bước preprocessing, fit trên tập nào, lý do |
| `feature_importance_*.csv` | Permutation importance; attention (FT-Transformer) hoặc mask (TabNet) |
| `figures/` | learning curve, Optuna history / param importance, dự đoán vs thực tế, residual, importance, mask TabNet |

Tổng hợp: `results/summary.csv`, `results/summary.md`, `figures/summary/*.png`, `results/<dataset>/ablation.csv`, `results/kc/size_curve.csv`.

`summary.md` có thêm bảng **phân tích sai số trên test**: tỉ lệ tổng bình phương sai số đến từ căn sai nhất / 5 căn sai nhất, tỉ lệ ŷ/y lớn nhất, và RMSE/R² khi bỏ căn sai nhất. Bảng này **chỉ để phân tích**; metric chính không bị thay đổi. Mỗi model còn có `largest_errors_test.csv` liệt kê 10 căn sai nhiều nhất kèm feature gốc. Nhóm đã quyết định **không** kẹp (clip) dự đoán; các trường hợp ngoại suy được báo cáo như một kết quả.

## Quy trình đánh giá (tóm tắt)

```
split 80/20 (seed 42) → train | test (khoá)
train → KFold(5, shuffle, seed 42); mỗi fold: fold_train → inner_train | inner_val (15%, seed 42)
   fit scaler / encoder / target scaler trên inner_train → train (seed 36), early stopping trên inner_val
   → dự đoán fold_val → expm1 về giá gốc → MAE / RMSE / R²
Optuna: chọn theo mean CV RMSE (giá gốc)
Final: fit trên toàn bộ train, n_epochs = median(best_epoch của 5 fold) → test đúng 1 lần
```

## Cho nhóm ML: dùng chung split

`row` trong các file `splits/<dataset>/*.csv` là vị trí dòng (0-based) trong CSV gốc. Các lệnh sklearn thông thường cho ra **đúng cùng split và fold** (đã kiểm tra trong `tests/test_no_leakage.py`):

```python
df = pd.read_csv("kc_house_data.csv")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
folds = KFold(5, shuffle=True, random_state=42).split(train_df)
```

`size_subsets.csv` (cột `fold, row, rank`): tập con cỡ `n` của fold `k` là các dòng có `fold == k` và `rank < n`. Nhóm ML có thể chạy model tốt nhất của mình trên cùng các tập con này để vẽ chung learning curve.

Để có cột gap so với ML trong bảng tổng hợp: `python run.py summary --ml-results ml.csv`, với `ml.csv` có ít nhất các cột `dataset, model, test_rmse` (`dataset` là `housing` hoặc `kc`).

## Test

```bash
python -m pytest -q tests                      # ~6 phút, gồm cả test end-to-end
python -m pytest -q tests -k "not pipeline"    # ~10 giây
```

## Những điểm cụ thể hoá so với plan

- **Cấu hình train của FT-Transformer theo số dòng train:**
  - Dưới 2.000 dòng: batch 32, tối đa 400 epoch, patience 30.
  - Từ 2.000 dòng trở lên: batch 256, tối đa 200 epoch, patience 16.

  Với thí nghiệm chính, quy tắc này trùng với bảng trong plan (Housing / King County). Nó chỉ thực sự khác khi chạy các tập con nhỏ của King County trong learning curve.
- **Learning curve theo kích thước:** hyperparams cố định, trừ batch size của TabNet được hạ về 64 / 16 khi tập con dưới 2.000 dòng. Batch ≥ 256 tune cho King County chỉ còn 1–2 bước/epoch trên 436 dòng.
- **Số epoch của model cuối** là median của best epoch (đếm từ 1) qua 5 fold.
- **Ablation "không feature engineering":** với Housing là không lấy log `area`; với King County là dùng các cột gốc (chỉ bỏ `id`, `date`).
- **Attention của FT-Transformer:** `rtdl_revisiting_models` không trả attention, nên `explain.py` dùng forward pre-hook để tính lại softmax(QKᵀ/√d) của truy vấn [CLS].
