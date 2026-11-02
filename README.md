# Nhóm DL — FT-Transformer, TabNet & MLP cho dự đoán giá nhà

3 model DL trên Housing (545 dòng) và King County (21.613 dòng), cùng split/fold với nhóm ML để so sánh công bằng:

- FT-Transformer và TabNet, trong [`dl/`](dl/): hiện thực [`implement_plan.md`](implement_plan.md).
- MLP, trong [`dl_mlp_project/`](dl_mlp_project/): xem mục [MLP](#mlp-dl_mlp_project).

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

Hai file CSV gốc (`Housing.csv`, `kc_house_data.csv`) mặc định nằm ở thư mục cha của `dl/`. Nếu để chỗ khác, đặt biến môi trường `HOUSE_DATA_DIR`. Env `house-dl` đã đủ thư viện cho cả MLP, không cần cài thêm.

## Chạy toàn bộ từ đầu đến cuối

Chạy lần lượt các bước dưới đây, tất cả trong env `house-dl`. Kết quả cuối cùng là bảng so sánh 3 model × 2 dataset ở `dl/results/summary.md`.

`tune` lưu tiến độ vào `results/<dataset>/<model>/optuna.db`. Nếu bị ngắt giữa chừng, chạy lại đúng lệnh đó để làm tiếp. Muốn tune lại từ đầu thì xoá file `optuna.db` đó trước khi chạy.

### Bước 0 — Kiểm tra (từ `dl/`)

```bash
conda activate house-dl
cd dl
python run.py splits                          # splits/ đã có thì giữ nguyên, không tạo lại
python -m pytest -q tests -k "not pipeline"   # ~15 giây
```

### Bước 1 — FT-Transformer và TabNet trên Housing (CPU là đủ)

```bash
python run.py --threads 1 tune  --dataset housing --model ft_transformer --n-trials 50
python run.py --threads 1 final --dataset housing --model ft_transformer
python run.py --threads 1 tune  --dataset housing --model tabnet --n-trials 50
python run.py --threads 1 final --dataset housing --model tabnet
python run.py --threads 1 ablation --dataset housing
```

Thời gian đo trên i5-1135G7: tune FT-Transformer khoảng 1 giờ, TabNet khoảng 50 phút. Có thể mở 2 terminal để chạy 2 model song song.

### Bước 2 — FT-Transformer và TabNet trên King County (cần GPU)

**Cách A: máy có GPU NVIDIA** (vd. RTX 4060, đã cài torch bản `cu126` như ở mục Cài đặt), chạy từ `dl/`:

```bash
python run.py --device cuda tune  --dataset kc --model ft_transformer --n-trials 30
python run.py --device cuda final --dataset kc --model ft_transformer
python run.py --device cuda tune  --dataset kc --model tabnet --n-trials 30
python run.py --device cuda final --dataset kc --model tabnet
python run.py --device cuda size-curve          # cần best_params.json của cả 2 model trên kc
python run.py --device cuda ablation --dataset kc
```

**Cách B: Kaggle** (GPU miễn phí). Notebook [`dl/notebooks/kaggle_kc_gpu.ipynb`](dl/notebooks/kaggle_kc_gpu.ipynb) chạy đúng các lệnh trên:

1. Nén toàn bộ thư mục `dl/` (gồm `splits/` và `results/`) thành `dl.zip`, rồi tạo Kaggle Dataset tên `house-dl-code` từ file đó.
2. Mở notebook, làm theo phần *Chuẩn bị* trong đó. Một session Kaggle tối đa 12 giờ, nên FT-Transformer và TabNet chạy ở 2 session riêng.
3. Mỗi session cho ra một file `dl_results_kc.zip`. Tải về và giải nén vào `dl/`.

### Bước 3 — MLP trên cả 2 dataset (từ `dl_mlp_project/`)

```bash
cd ../dl_mlp_project
python check_datasets.py      # kiểm tra 2 CSV ở thư mục gốc
python smoke_test.py          # phải in "SMOKE TEST PASSED"; điểm số ở bước này là dữ liệu giả, không dùng
python dl_mlp_experiment.py --housing_csv ../Housing.csv --king_county_csv ../kc_house_data.csv --output_dir results_mlp
```

- Lệnh cuối chạy cả 2 dataset, mỗi dataset 5 cấu hình × 5 fold, rồi train model cuối và đánh giá test.
- Script tự dùng GPU nếu có; thêm `--cpu` để ép chạy CPU.
- Giữ nguyên `--output_dir results_mlp`, vì bước tổng hợp đọc từ `dl_mlp_project/results_mlp/`.

### Bước 4 — Tổng hợp 3 model (từ `dl/`)

```bash
cd ../dl
python run.py summary                         # thêm --ml-results ml.csv khi nhóm ML gửi kết quả
```

Kết quả nằm ở:
- `results/summary.md`: bảng metric và bảng phân tích sai số.
- `results/summary.csv`.
- `figures/summary/*.png`: hình so sánh CV và hình thời gian train vs RMSE.

Nếu log có dòng `skip MLP ...` nghĩa là kết quả MLP được chạy bằng code cũ (seed CV 36), phải chạy lại Bước 3.

Không cần train lại khi chỉ muốn vẽ lại hình: dùng `python run.py replot` rồi `python run.py summary`.

## Các lệnh của `dl/run.py` (chạy từ thư mục `dl/`)

| Lệnh | Việc |
|---|---|
| `python run.py splits` | Tạo split 80/20, 5 fold, tập con cho learning curve (chạy 1 lần; giữ nguyên nếu đã có) |
| `python run.py tune --dataset housing --model ft_transformer --n-trials 50` | Optuna 5-fold CV; trial 0 là cấu hình mặc định; chạy tiếp được nếu bị ngắt |
| `python run.py final --dataset housing --model ft_transformer` | Train lại trên 85% train (early stopping trên 15% còn lại), giải thích model trên 15% đó, đánh giá test **1 lần**, vẽ hình |
| `python run.py all --dataset housing` | `tune` + `final` cho cả 2 model |
| `python run.py ablation --dataset housing` | Ablation preprocessing (cấu hình mặc định, CV) |
| `python run.py size-curve` | Learning curve theo số dòng train trên King County |
| `python run.py summary [--ml-results ml.csv]` | Bảng tổng hợp + hình so sánh + phân tích sai số trên test cho FT-Transformer, TabNet và MLP (nếu đã có `dl_mlp_project/results_mlp/`); thêm kết quả nhóm ML để tính gap |
| `python run.py replot` | Vẽ lại hình của từng model từ kết quả đã lưu (không train lại) |

Tuỳ chọn chung: `--device auto|cpu|cuda`, `--threads N`.
Trên CPU nên dùng `--threads 1` và chạy nhiều job song song: với model nhỏ, thêm luồng làm **chậm** đi (đo được: FT-Transformer trên Housing 0,54 s/epoch với 1 luồng, 1,2 s/epoch với 8 luồng).

**King County cần GPU.** Trên CPU, FT-Transformer mất ~46 s/epoch và TabNet ~6 s/epoch, tức hàng giờ cho một trial (xem Bước 2 ở trên).

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
| `best_params.json` | Hyperparams tốt nhất, CV mean ± std, thông tin Optuna, seed, version thư viện, phần cứng |
| `optuna_trials.csv`, `optuna.db` | Toàn bộ trial (db dùng để chạy tiếp) |
| `cv_folds.csv`, `cv_histories.json` | Metric và best epoch từng fold; learning curve từng fold của trial tốt nhất |
| `test_metrics.json` | MAE, RMSE, R², NRMSE trên test + 95% CI bootstrap, `fit_time`, `predict_time`, best epoch của model cuối, số dòng fit / val / test |
| `final_history.json` | Train loss và val RMSE từng epoch của model cuối |
| `predictions_test.csv` | `row, y_true, y_pred` |
| `preprocessing.md` | Các bước preprocessing, fit trên tập nào, lý do |
| `feature_importance_*.csv` | Permutation importance; attention (FT-Transformer) hoặc mask (TabNet). Tất cả tính trên 15% validation của model cuối, cùng các dòng với MLP |
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
Final: train → fit (85%) | val (15%, seed 42); fit preprocessing trên fit, early stopping trên val
   → giải thích model trên val → test đúng 1 lần
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

## MLP (`dl_mlp_project/`)

Code MLP chép từ [`Focus-In-Self-Coding/P4AI-DS/dl_mlp_project`](https://github.com/Focus-In-Self-Coding/P4AI-DS/tree/d90149bcc93e8d9debfc6a03128d690f957c1d45/dl_mlp_project) (commit `d90149b`) để sửa tiếp trong repo này. So với bản gốc, `KFold` của CV và 2 lần tách 15% để early stopping (trong mỗi fold, và cho model cuối) đổi seed từ 36 sang 42, đường dẫn CSV trỏ về thư mục gốc; cách chạy xem [`dl_mlp_project/README.md`](dl_mlp_project/README.md). MLP dùng **đúng cùng các dòng** với FT-Transformer / TabNet ở mọi bước: test, 5 fold, inner_val trong từng fold, và 15% validation của model cuối (fold `k` của MLP, đếm từ 1, là fold `k − 1` trong `folds.csv`).

Sau khi chạy MLP (`--output_dir results_mlp`), `python run.py summary` tự thêm MLP vào bảng tổng hợp và hình so sánh. Metric test, 95% CI, NRMSE và phân tích sai số của MLP được tính lại từ `test_predictions.csv` bằng cùng code với 2 model kia; CV lấy từ `cv_fold_details.csv`. MLP chạy với seed CV cũ (36) sẽ bị bỏ qua kèm cảnh báo.

## Test

```bash
python -m pytest -q tests                      # ~2 phút, gồm cả test end-to-end
python -m pytest -q tests -k "not pipeline"    # ~15 giây
```

## Những điểm cụ thể hoá so với plan

- **Cấu hình train của FT-Transformer theo số dòng train:**
  - Dưới 2.000 dòng: batch 32, tối đa 400 epoch, patience 30.
  - Từ 2.000 dòng trở lên: batch 256, tối đa 200 epoch, patience 16.

  Với thí nghiệm chính, quy tắc này trùng với bảng trong plan (Housing / King County). Nó chỉ thực sự khác khi chạy các tập con nhỏ của King County trong learning curve.
- **Learning curve theo kích thước:** hyperparams cố định, trừ batch size của TabNet được hạ về 64 / 16 khi tập con dưới 2.000 dòng. Batch ≥ 256 tune cho King County chỉ còn 1–2 bước/epoch trên 436 dòng.
- **Model cuối** dùng chung protocol với MLP: train trên 85% train, early stopping trên 15% còn lại (seed 42). Permutation importance, attention và mask TabNet tính trên 15% đó, nên test chỉ dùng cho metric và phân tích sai số.
- **Ablation "không feature engineering":** với Housing là không lấy log `area`; với King County là dùng các cột gốc (chỉ bỏ `id`, `date`).
- **Attention của FT-Transformer:** `rtdl_revisiting_models` không trả attention, nên `explain.py` dùng forward pre-hook để tính lại softmax(QKᵀ/√d) của truy vấn [CLS].
