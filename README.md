# Nhóm DL — MLP, TabM, TabNet, FT-Transformer & ModernNCA cho dự đoán giá nhà

5 model DL trên Housing (545 dòng) và King County (21.613 dòng), cùng split/fold với nhóm ML để so sánh công bằng. Toàn bộ code nằm trong [`dl/`](dl/); FT-Transformer và TabNet hiện thực [`implement_plan.md`](implement_plan.md), MLP lấy từ code MLP gốc của nhóm (xem mục [MLP](#mlp)), TabM và ModernNCA là 2 model năm 2025 (xem mục [TabM](#tabm) và [ModernNCA](#modernnca)). Cả 5 model được tune bằng Optuna với cùng số trial.

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

Nếu `nvidia-smi` báo `CUDA Version` ≥ 13.0, có thể dùng `cu130` thay cho `cu126`.

Hai file CSV gốc (`Housing.csv`, `kc_house_data.csv`) mặc định nằm ở thư mục cha của `dl/`. Nếu để chỗ khác, đặt biến môi trường `HOUSE_DATA_DIR`.

## Chạy toàn bộ: `run_all.py`

Một lệnh chạy hết mọi thứ, từ thư mục `dl/`:

```bash
python run_all.py --device cuda
```

Các bước, theo thứ tự:

1. Tạo split. File split đã có thì giữ nguyên, không tạo lại.
2. Chạy các test nhanh (~15 giây). Nếu fail thì dừng luôn, không train.
3. Với từng dataset (Housing rồi King County) và từng model (FT-Transformer, TabNet, MLP, TabM, ModernNCA): tune, rồi train model cuối và đánh giá test.
4. Ablation preprocessing trên 2 dataset.
5. Learning curve theo số dòng train (size-curve) trên King County.
6. `summary`: bảng so sánh ở `results/summary.md`.

Mỗi bước in giờ bắt đầu, nên xem log là biết bước nào tốn bao lâu.

**Chạy lại sau khi bị ngắt:** chạy lại đúng lệnh đó, các phần đã xong sẽ không bị làm lại.
- `tune` chạy tiếp từ `results/<dataset>/<model>/optuna.db`. Trial bị ngắt giữa chừng sẽ được chạy lại, kể cả trial cấu hình mặc định và 5 cấu hình gốc của MLP.
- Model cuối bị bỏ qua nếu `test_metrics.json` đã khớp với hyperparams tốt nhất. Dùng `--force` để ép train lại.
- Ablation và size-curve chỉ chạy những dòng còn thiếu trong file CSV kết quả.

Muốn tune lại một model từ đầu thì xoá `results/<dataset>/<model>/optuna.db` trước khi chạy.

| Tuỳ chọn | Ý nghĩa |
|---|---|
| `--device auto\|cpu\|cuda` | Mặc định `auto`: có GPU thì dùng GPU |
| `--threads N` | Số luồng CPU của torch. Chỉ dùng khi chạy CPU (`--threads 1` là nhanh nhất) |
| `--datasets housing kc` | Chỉ chạy các dataset được liệt kê |
| `--models ft_transformer tabnet mlp tabm modernnca` | Chỉ chạy các model được liệt kê |
| `--force` | Train lại model cuối kể cả khi đã khớp |
| `--skip-tests` | Bỏ bước test nhanh |
| `--ml-results ml.csv` | Thêm cột chênh lệch so với model ML tốt nhất vào bảng tổng hợp |

Ví dụ:

```bash
python run_all.py --device cpu --threads 1 --datasets housing   # chỉ Housing, trên CPU
python run_all.py --device cuda --datasets kc --models tabm     # chỉ TabM trên King County
```

**Thời gian (RTX 4060 + i5-12400F):** tune King County đo được khoảng 2,4 giờ cho FT-Transformer và 4,8 giờ cho TabNet (30 trial mỗi model). MLP, TabM và ModernNCA ước tính 0,5–3 giờ mỗi model, chưa đo. Khi FT-Transformer và TabNet đã xong, phần còn lại (3 model kia trên 2 dataset, ablation, size-curve) mất khoảng 7–12 giờ, nên chạy qua đêm. Trên CPU (i5-1135G7), tune Housing mất khoảng 1 giờ cho FT-Transformer và 50 phút cho TabNet. King County trên CPU mất hàng giờ cho một trial, nên cần GPU.

**Không có GPU:** dùng Kaggle với notebook [`dl/notebooks/kaggle_kc_gpu.ipynb`](dl/notebooks/kaggle_kc_gpu.ipynb).
1. Nén toàn bộ thư mục `dl/` (gồm `splits/` và `results/`) thành `dl.zip`, rồi tạo Kaggle Dataset tên `house-dl-code` từ file đó.
2. Làm theo phần *Chuẩn bị* trong notebook. Một session Kaggle tối đa 12 giờ, nên notebook chia làm 2 session.
3. Mỗi session cho ra một file `dl_results_kc.zip`. Tải về, giải nén vào `dl/`, rồi chạy `python run.py summary`.

## Các lệnh lẻ của `dl/run.py` (chạy từ thư mục `dl/`)

| Lệnh | Việc |
|---|---|
| `python run.py splits` | Tạo split 80/20, 5 fold, tập con cho learning curve (chạy 1 lần; giữ nguyên nếu đã có) |
| `python run.py tune --dataset housing --model ft_transformer` | Optuna 5-fold CV (mặc định 50 trial Housing / 30 trial King County, mọi model); chạy tiếp được nếu bị ngắt |
| `python run.py final --dataset housing --model mlp` | Train lại trên 85% train (early stopping trên 15% còn lại), giải thích model trên 15% đó, đánh giá test **1 lần**, vẽ hình |
| `python run.py all --dataset housing` | `tune` + `final` cho cả 5 model |
| `python run.py ablation --dataset housing` | Ablation preprocessing (cấu hình mặc định, CV) |
| `python run.py size-curve` | Learning curve theo số dòng train trên King County |
| `python run.py summary [--ml-results ml.csv]` | Bảng tổng hợp + hình so sánh + phân tích sai số trên test; thêm kết quả nhóm ML để tính gap |
| `python run.py replot` | Vẽ lại hình của từng model từ kết quả đã lưu (không train lại) |

Tuỳ chọn chung: `--device auto|cpu|cuda`, `--threads N`.
Trên CPU nên dùng `--threads 1` và chạy nhiều job song song: với model nhỏ, thêm luồng làm **chậm** đi (đo được: FT-Transformer trên Housing 0,54 s/epoch với 1 luồng, 1,2 s/epoch với 8 luồng).

## Cấu trúc

```
dl/
├── run.py                     # CLI từng bước
├── run_all.py                 # chạy toàn bộ từ đầu đến cuối
├── src/
│   ├── config.py              # đường dẫn, SEED_DATA=42, SEED_MODEL=36
│   ├── data.py                # đọc CSV + feature engineering stateless (dùng chung cho 5 model)
│   ├── splits.py              # split / fold / inner split / tập con learning curve
│   ├── preprocess.py          # TabularPreprocessor (numcat / tabnet / onehot), TargetTransformer + preprocessing.md
│   ├── metrics.py             # MAE, RMSE, R², NRMSE, bootstrap CI
│   ├── cv.py                  # 5-fold CV với inner_val cho early stopping
│   ├── tune.py                # Optuna (TPE seed 42, MedianPruner, SQLite để resume)
│   ├── final.py               # model cuối + test (module DUY NHẤT đọc test split)
│   ├── explain.py             # permutation importance, attention [CLS], mask TabNet, láng giềng ModernNCA
│   ├── plots.py, summary.py
│   └── models/                # ft_transformer.py, tabnet.py, mlp.py, tabm.py, modernnca.py (cùng interface BaseModel)
├── experiments/               # ablation.py, size_curve.py
├── tests/                     # test_no_leakage.py, test_smoke.py, test_pipeline.py
├── notebooks/kaggle_kc_gpu.ipynb
├── splits/                    # dùng chung với nhóm ML
├── results/<dataset>/<model>/ # kết quả từng model (xem dưới)
└── figures/                   # hình tổng hợp, ablation, learning curve theo kích thước
```

## Kết quả của mỗi model × dataset (`results/<dataset>/<model>/`)

Cả 5 model ghi cùng một bộ file:

| File | Nội dung |
|---|---|
| `best_params.json` | Hyperparams tốt nhất, CV mean ± std, thông tin Optuna, seed, version thư viện, phần cứng |
| `optuna_trials.csv`, `optuna.db` | Toàn bộ trial (db dùng để chạy tiếp) |
| `cv_folds.csv`, `cv_histories.json` | Metric và best epoch từng fold; learning curve từng fold của trial tốt nhất |
| `test_metrics.json` | MAE, RMSE, R², NRMSE trên test + 95% CI bootstrap, `fit_time`, `predict_time`, best epoch của model cuối, số dòng fit / val / test |
| `final_history.json` | Train loss và val RMSE từng epoch của model cuối |
| `predictions_test.csv` | `row, y_true, y_pred` |
| `preprocessing.md` | Các bước preprocessing, fit trên tập nào, lý do |
| `feature_importance_*.csv` | Permutation importance (cả 5 model); attention (FT-Transformer); mask (TabNet). Tất cả tính trên 15% validation của model cuối |
| `neighbors_val.csv` | Chỉ ModernNCA: với 5 căn đầu của 15% validation, 5 căn train có trọng số lớn nhất (giá của chúng được lấy trung bình để dự đoán) |
| `figures/` | learning curve, Optuna history / param importance, dự đoán vs thực tế, residual, importance, mask TabNet |

Tổng hợp: `results/summary.csv`, `results/summary.md`, `figures/summary/*.png`, `results/<dataset>/ablation.csv`, `results/kc/size_curve.csv`.

`summary.md` có thêm bảng **phân tích sai số trên test**: tỉ lệ tổng bình phương sai số đến từ căn sai nhất / 5 căn sai nhất, tỉ lệ ŷ/y lớn nhất, và RMSE/R² khi bỏ căn sai nhất. Bảng này **chỉ để phân tích**; metric chính không bị thay đổi. Mỗi model còn có `largest_errors_test.csv` liệt kê 10 căn sai nhiều nhất kèm feature gốc. Nhóm đã quyết định **không** kẹp (clip) dự đoán; các trường hợp ngoại suy được báo cáo như một kết quả.

## Quy trình đánh giá (tóm tắt)

```
split 80/20 (seed 42) → train | test (khoá)
train → KFold(5, shuffle, seed 42); mỗi fold: fold_train → inner_train | inner_val (15%, seed 42)
   fit scaler / encoder / target scaler trên inner_train → train (seed 36), early stopping trên inner_val
   → dự đoán fold_val → expm1 về giá gốc → MAE / RMSE / R²
Tuning: Optuna, chọn theo mean CV RMSE (giá gốc); cấu hình mặc định (MLP: 5 cấu hình gốc) chạy trước
Final: train → fit (85%) | val (15%, seed 42); fit preprocessing trên fit, early stopping trên val
   → giải thích model trên val → test đúng 1 lần
```

## MLP

MLP lấy từ code MLP gốc của nhóm ([`Focus-In-Self-Coding/P4AI-DS/dl_mlp_project`](https://github.com/Focus-In-Self-Coding/P4AI-DS/tree/d90149bcc93e8d9debfc6a03128d690f957c1d45/dl_mlp_project), commit `d90149b`), đưa vào [`dl/src/models/mlp.py`](dl/src/models/mlp.py) để chạy chung pipeline, chung định dạng kết quả với 2 model kia.

**Giữ nguyên từ bản gốc:**
- Kiến trúc: các lớp Linear → ReLU → Dropout, đầu ra 1 giá trị.
- Cách train: AdamW, loss MSE, ReduceLROnPlateau (factor 0,5, patience 7, lr tối thiểu 1e-5), gradient clipping 5,0, early stopping với patience 22 và tối đa 200 epoch.
- Seed model 36.
- 5 cấu hình của bản gốc. Chúng giờ là 5 trial đầu tiên của Optuna (xem *Khác bản gốc*):

| Cấu hình | Hidden layers | Dropout | Learning rate | Weight decay | Batch |
|---|---|---|---|---|---|
| `mlp_64_32_d0_lr1e-3_wd1e-4_b32` | 64, 32 | 0,0 | 1e-3 | 1e-4 | 32 |
| `mlp_128_64_d0.1_lr1e-3_wd1e-4_b64` | 128, 64 | 0,1 | 1e-3 | 1e-4 | 64 |
| `mlp_128_64_32_d0.2_lr5e-4_wd1e-3_b128` | 128, 64, 32 | 0,2 | 5e-4 | 1e-3 | 128 |
| `mlp_256_128_d0.1_lr5e-4_wd1e-4_b128` | 256, 128 | 0,1 | 5e-4 | 1e-4 | 128 |
| `mlp_64_32_d0.2_lr1e-3_wd1e-2_b64` | 64, 32 | 0,2 | 1e-3 | 1e-2 | 64 |

**Khác bản gốc (để so sánh công bằng với các model khác; cần ghi trong report):**
- **Tuning bằng Optuna** với cùng số trial như các model khác (50 Housing / 30 King County), thay cho việc chỉ chạy 5 cấu hình.
  - Không gian tìm kiếm lấy theo MLP của Gorishniy et al. (2021): độ rộng lớp đầu / các lớp giữa / lớp cuối (16–512, thang log), tối đa 4 lớp, dropout 0–0,5, learning rate 1e-5–1e-2, batch 32–256, scaler StandardScaler hoặc Quantile.
  - Weight decay được nới tới 1e-2 (bài báo dừng ở 1e-3) để 5 cấu hình gốc nằm trong không gian. 5 cấu hình gốc luôn chạy đầu tiên.
  - Kết quả của lần chạy chỉ có 5 cấu hình cũ được tự động chuyển sang `results/<dataset>/mlp_grid/` khi tune MLP lần đầu bằng code mới (kể cả các dòng MLP trong `results/kc/size_curve.csv`), nên vẫn dùng được trong report.
- **Feature:** dùng chung feature engineering của [`dl/src/data.py`](dl/src/data.py), ví dụ log `sqft_*` / `area`, sửa `bedrooms = 33`, thêm `dist_downtown_km` và `house_age`, bỏ `sqft_above`.
- **Mã hoá đầu vào:** cấu hình gốc dùng StandardScaler cho cột số; categorical (`zipcode`, `furnishingstatus`) dùng one-hot vì MLP không có lớp embedding.
  - Preprocessing gốc của MLP (cột gốc) gần với biến thể "Không feature engineering" trong ablation. Vì vậy ablation cho thấy feature engineering giúp MLP bao nhiêu.
  - Biến thể "StandardScaler" không chạy cho MLP vì baseline của MLP đã dùng StandardScaler.
- **Split và seed:** dùng chung với 2 model kia. Seed 42 cho KFold và cho 2 lần tách 15% để early stopping; permutation importance cũng dùng seed 42 (bản gốc dùng 36 cho cả hai).
- **Đánh giá:** model cuối, giải thích model, test, CI và phân tích sai số đi qua đúng code của các model kia.

## TabM

TabM (Gorishniy et al., ICLR 2025) là **một mạng chứa ensemble k = 32 MLP dùng chung phần lớn trọng số**; dự đoán là trung bình của 32 dự đoán. Code: gói `tabm` của tác giả, trong [`dl/src/models/tabm.py`](dl/src/models/tabm.py).

- **Input giống hệt MLP:** one-hot, không dùng embedding cho cột số. So sánh MLP với TabM vì thế chỉ khác ở phần ensemble.
- **Train theo notebook ví dụ của tác giả:**
  - AdamW, MSE tính riêng cho từng thành viên rồi lấy trung bình (không phải MSE của dự đoán trung bình).
  - Gradient clipping 1,0, early stopping trên dự đoán trung bình.
  - Batch 256 và patience 16; dưới 2.000 dòng train thì batch 32, patience 30, như FT-Transformer.
- **Mặc định** (trial đầu tiên): 3 khối × 512, dropout 0,1, lr 2e-3, weight decay 3e-4, Quantile.
- **Không gian tìm kiếm** lấy theo README của TabM (k cố định 32): số khối 1–5, độ rộng 64–1024, lr 1e-4–5e-3, weight decay 0 hoặc 1e-4–1e-1. Có thêm dropout 0–0,5 và scaler. Với Housing, thu hẹp còn 1–3 khối và độ rộng 64–512.

## ModernNCA

ModernNCA (Ye et al., ICLR 2025) thuộc nhánh **dự đoán dựa trên láng giềng**: giá một căn là trung bình có trọng số softmax(−khoảng cách) của giá các căn train, với khoảng cách đo trong không gian embedding học được. Code viết lại theo bản của tác giả trong LAMDA-TALENT, ở [`dl/src/models/modernnca.py`](dl/src/models/modernnca.py).

- **Encoder:** cột liên tục và cột thứ tự đi qua embedding PLR-lite (`rtdl_num_embeddings`); cột 0/1 và one-hot giữ nguyên; sau đó qua Linear rồi 0–2 khối MLP. Khoảng cách Euclid, temperature 1.
- **Lúc train** (như TALENT): mỗi batch so với một phần ngẫu nhiên (`sample_rate`) của các căn train khác cùng các căn còn lại trong batch. Nhãn của chính căn đó bị che.
- **Lúc dự đoán:** dùng toàn bộ căn train làm candidate.
- **Không rò rỉ:** candidate chỉ là các dòng model được train (inner_train trong CV, phần fit 85% ở model cuối), không bao giờ là dòng val hay test. File `model.pt` lưu cả tập candidate.
- **Cách train:** AdamW, MSE, early stopping. Batch 1024 và patience 20 cho King County; batch 32 và patience 30 cho dữ liệu dưới 2.000 dòng.
- **Mặc định và không gian tìm kiếm** lấy theo config ModernNCA của TALENT. Với Housing, thu hẹp `dim` / `d_block` còn tối đa 512.
- **Giải thích riêng:** `neighbors_val.csv` liệt kê các căn train mà model dựa vào để dự đoán một căn validation.

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
python -m pytest -q tests                      # ~3–4 phút, gồm cả test end-to-end
python -m pytest -q tests -k "not pipeline"    # ~20 giây
```

## Những điểm cụ thể hoá so với plan

- **Cấu hình train của FT-Transformer theo số dòng train:**
  - Dưới 2.000 dòng: batch 32, tối đa 400 epoch, patience 30.
  - Từ 2.000 dòng trở lên: batch 256, tối đa 200 epoch, patience 16.

  Với thí nghiệm chính, quy tắc này trùng với bảng trong plan (Housing / King County). Nó chỉ thực sự khác khi chạy các tập con nhỏ của King County trong learning curve.
- **Learning curve theo kích thước:** hyperparams cố định, trừ batch size của TabNet được hạ về 64 / 16 khi tập con dưới 2.000 dòng. Batch ≥ 256 tune cho King County chỉ còn 1–2 bước/epoch trên 436 dòng.
- **Model cuối** dùng protocol của code MLP gốc cho cả 5 model: train trên 85% train, early stopping trên 15% còn lại (seed 42). Permutation importance, attention, mask TabNet và láng giềng ModernNCA tính trên 15% đó, nên test chỉ dùng cho metric và phân tích sai số.
- **Trial cố định:** cấu hình mặc định của FT-Transformer / TabNet / TabM / ModernNCA và 5 cấu hình gốc của MLP luôn được xếp vào hàng chờ trước các trial do Optuna sinh. Trial bị ngắt giữa chừng sẽ được xếp lại khi chạy tiếp.
- **Size-curve** ghi thêm cột `params_hash`. Các dòng tính với hyperparams khác hyperparams tốt nhất hiện tại sẽ tự bị tính lại.
- **Ablation "không feature engineering":** với Housing là không lấy log `area`; với King County là dùng các cột gốc (chỉ bỏ `id`, `date`).
- **Attention của FT-Transformer:** `rtdl_revisiting_models` không trả attention, nên `explain.py` dùng forward pre-hook để tính lại softmax(QKᵀ/√d) của truy vấn [CLS].
