# Implementation Plan — Nhóm DL: FT-Transformer, TabNet

> **Mục tiêu project:** dự đoán giá nhà và **so sánh hiệu quả giữa Classical ML và Deep Learning trên 2 dataset có quy mô khác nhau**.
> Tài liệu này mô tả cách hiện thực 2 model DL (không thuộc họ MLP) sao cho kết quả **so sánh công bằng được** với nhóm ML.

| Mục | Nội dung |
|---|---|
| Dataset 1 | Housing Prices [D1] — 545 dòng, 12 feature, không missing (*quy mô nhỏ*) |
| Dataset 2 | King County House Sales [D2] — 21.613 dòng, 20 feature, không missing (*quy mô vừa*) |
| Model | FT-Transformer [1], TabNet [6] |
| Seed | **42** cho mọi thao tác chia / lấy mẫu dữ liệu; **36** cho phần ngẫu nhiên bên trong model (theo quy định nhóm) |
| Môi trường | conda env `house-dl`: Python 3.12, torch 2.14.1 (CPU), scikit-learn 1.9.1, optuna 5.0.0, pytorch-tabnet 4.1.0, rtdl_revisiting_models. Local chạy CPU; tuning King County nên chạy trên Kaggle/Colab GPU |

---

## 1. Câu hỏi nghiên cứu và vai trò của 2 model

**RQ1.** Trên dataset nhỏ (Housing, ~436 dòng train), DL có cạnh tranh được với Classical ML không?
**RQ2.** Trên dataset vừa (King County, ~17.3k dòng train), khoảng cách DL–ML thay đổi thế nào?
**RQ3.** Đánh đổi giữa độ chính xác, thời gian train/inference và khả năng giải thích.

2 model đại diện cho **2 hướng kiến trúc khác nhau** của DL cho dữ liệu bảng. Cả hai đều **học hoàn toàn từ dữ liệu train** (không có prior học sẵn), nên phù hợp để kiểm tra giả thuyết "DL cần nhiều dữ liệu":

| Model | Hướng kiến trúc | Đối chiếu với model ML | Giả thuyết trên Housing (nhỏ) | Giả thuyết trên King County (vừa) |
|---|---|---|---|---|
| **FT-Transformer** [1] | Self-attention giữa mọi cặp feature, học tương tác toàn cục | Linear models (tương tác phải tự tạo tay) và GBDT | Dễ overfit, khó thắng Ridge/GBDT | Cạnh tranh được với GBDT; [1] cho thấy FT-Transformer là DL mạnh nhất nhưng GBDT vẫn thắng ở một số dataset |
| **TabNet** [6] | Chọn feature tuần tự và thưa, mô phỏng cây quyết định một cách khả vi | Decision Tree, Random Forest, GBDT | Yếu: [6] thiết kế và đánh giá chủ yếu trên dataset lớn | Trung bình; [12] cho thấy TabNet thường kém XGBoost ngoài các dataset của paper gốc |

Bối cảnh cho phần Related Work: GBDT thường vẫn thắng DL trên dữ liệu bảng cỡ vừa [11, 12], và khác biệt DL–GBDT phụ thuộc vào đặc tính dataset, trong đó kích thước là một yếu tố quan trọng [13]. Survey tổng quan: [14].

> ⚠️ **Lưu ý phương pháp luận:** 2 dataset khác nhau **không chỉ về quy mô** mà còn về feature, thị trường và đơn vị tiền. So trực tiếp "Housing vs King County" sẽ **lẫn lộn** hiệu ứng quy mô với các yếu tố khác. Vì vậy plan có thêm thí nghiệm **learning curve theo kích thước train trên King County** (Mục 6.1): giữ nguyên dataset, chỉ thay số dòng train. Thí nghiệm này tách riêng được hiệu ứng quy mô.

---

## 2. Quy định chung của nhóm và cách hiện thực

| Quy định | Cách hiện thực |
|---|---|
| Split 80/20, `random_seed = 42` | `train_test_split(test_size=0.2, random_state=42, shuffle=True)`. Lưu `train_idx.csv`, `test_idx.csv` và **gửi cả nhóm ML dùng chung** |
| Test chỉ dùng đánh giá cuối | `final.py` là nơi **duy nhất** đọc test; mỗi model chỉ chạy test **1 lần** với hyperparams đã chốt |
| Tuning bằng 5-fold CV trên train | `KFold(5, shuffle=True, random_state=42)`. Lưu `folds.csv` để ML và DL dùng **cùng 5 fold** |
| DL dùng validation riêng để early stopping | Trong mỗi fold: tách 15% fold-train làm `inner_val` (seed 42). Early stopping trên `inner_val`, chấm điểm trên `fold_val` |
| Transform target thì đổi về giá gốc | Train trên `log1p(price)`; trước khi tính metric gọi `expm1` |
| Metric: MAE, RMSE, R², training time | Tính trên giá gốc. Ghi thêm *predict time*, bootstrap 95% CI và NRMSE (Mục 3.6) |
| Seed | **42** cho mọi thao tác chia hoặc lấy mẫu dữ liệu: split, KFold, `inner_val`, tập con learning curve, bootstrap, permutation importance, Optuna `TPESampler(seed=42)`. **36** cho phần ngẫu nhiên trong model: `torch.manual_seed(36)`, `np.random.seed(36)`, `random.seed(36)` trước khi train, TabNet `seed=36` |
| Lưu đầy đủ để tổng hợp | Mỗi model/dataset có 1 thư mục `results/` chuẩn (Mục 8) |

---

## 3. Pipeline chung cho cả 2 model

### 3.1 Cấu trúc thư mục

```
dl/
├── README.md                    # cách chạy lại toàn bộ
├── requirements.txt             # PIN version (đặc biệt pytorch-tabnet)
├── splits/{housing,kc}/         # train_idx.csv, test_idx.csv, folds.csv, size_subsets.csv
├── src/
│   ├── config.py                # đường dẫn, SEED_DATA=42, SEED_MODEL=36, danh sách cột
│   ├── data.py                  # load_raw(), make_features_housing(), make_features_kc()   (stateless)
│   ├── splits.py                # tạo + lưu split/fold (chạy 1 lần duy nhất)
│   ├── preprocess.py            # TabularPreprocessor(kind=...), TargetTransformer       (stateful)
│   ├── metrics.py               # mae/rmse/r2 trên giá gốc, nrmse, bootstrap_ci()
│   ├── cv.py                    # run_cv(model, params, dataset) → metric từng fold, best_epoch, history
│   ├── tune.py                  # Optuna: objective = mean CV RMSE
│   ├── final.py                 # train lại trên full train → test 1 lần → lưu kết quả
│   ├── explain.py               # permutation importance + giải thích riêng của từng model
│   ├── plots.py
│   └── models/
│       ├── ft_transformer.py    # build model + training loop có early stopping
│       └── tabnet.py            # wrapper quanh TabNetRegressor
├── experiments/size_curve.py    # learning curve theo kích thước train (Mục 6.1)
├── tests/test_no_leakage.py
├── results/<dataset>/<model>/
└── figures/<dataset>/
```

Cả 2 model wrapper theo **cùng 1 interface**, để `cv.py`, `tune.py` và `final.py` không phải viết riêng cho từng model:

```python
class BaseModel:
    def fit(self, X_tr, y_tr, X_val=None, y_val=None) -> "BaseModel": ...
    def predict(self, X) -> np.ndarray: ...          # dự đoán trên thang target đã transform
    best_epoch_: int                                 # epoch tốt nhất theo inner_val
    history_: dict                                   # loss train/val theo epoch (để vẽ learning curve)
```

### 3.2 Preprocessing

**Bước stateless** (`data.py`, chạy trên từng dòng, không học tham số nên làm trước split được):

| Dataset | Xử lý |
|---|---|
| Housing | 6 cột yes/no → 0/1. `area` → `log`. Giữ `bedrooms`, `bathrooms`, `stories`, `parking` dạng số. `furnishingstatus` là categorical (3 mức) |
| King County | Bỏ `id`. Từ `date` tạo `days_since_start` và `sale_month`. Sửa `bedrooms=33 → 3` (typo: 1.620 sqft, 1.75 phòng tắm). Tạo `house_age` (clip ≥ 0), `years_since_update`, `is_renovated`, `has_basement`, `dist_downtown_km`. `log1p` các cột `sqft_*`. Bỏ `sqft_above` (bằng đúng `sqft_living − sqft_basement` ở 100% dòng), `yr_built`, `yr_renovated`. `zipcode` là categorical (70 mức) |

**Bước stateful** (`preprocess.py`, **fit trên `inner_train` của từng fold**, sau đó transform `inner_val`, `fold_val` và test). Có 2 kiểu đầu ra:

| `kind` | Đầu ra | Dùng cho |
|---|---|---|
| `"numcat"` | `x_num` (float, đã scale) + `x_cat` (index nguyên) + `cat_cardinalities` | FT-Transformer |
| `"tabnet"` | 1 ma trận: cột số đã scale + cột categorical dạng index; kèm `cat_idxs`, `cat_dims` | TabNet |

- **Scaler cho cột số:** `QuantileTransformer(output_distribution="normal")` theo [1], hoặc `StandardScaler`. Chọn bằng CV.
- **Categorical:** `OrdinalEncoder` với index 0 dành cho giá trị chưa thấy.

> ⚠️ `Housing.csv` đang **sắp xếp giảm dần theo giá**. Mọi bước chia dữ liệu bắt buộc `shuffle=True`: KFold, inner split và DataLoader.

### 3.3 Target

```
train:   y' = (log1p(price) − μ) / σ        # μ, σ tính trên inner_train
predict: price_hat = expm1(ŷ' · σ + μ)      # đổi về giá gốc rồi mới tính metric
```

### 3.4 Protocol CV → tuning → final

```
data ─ split(0.2, seed 42) ─► train (80%) | test (20%, khoá)
train ─ KFold(5, shuffle, seed 42) ─► với mỗi fold k:
    fold_train ─ split(0.15, seed 42) ─► inner_train | inner_val
    fit preprocessor + target scaler trên inner_train
    train model (seed 36), early stopping trên inner_val  → ghi best_epoch_k
    predict fold_val → đổi về giá gốc → MAE_k, RMSE_k, R²_k
Optuna chọn hyperparams theo mean(RMSE_k)              [chỉ dùng train]
Final: fit preprocessor trên full train
       train với n_epochs = median(best_epoch_k) của cấu hình tốt nhất (không cần val)
       predict test đúng 1 lần → metric + bootstrap CI + thời gian
```

Lý do dùng `inner_val` thay vì early stopping thẳng trên `fold_val`: nếu cùng một tập vừa dùng để dừng train vừa dùng để chấm điểm, điểm CV sẽ bị lạc quan. Khi đó so với CV của nhóm ML sẽ không công bằng.

### 3.5 Hyperparameter tuning

- **Thuật toán:** Optuna [15] với `TPESampler(seed=42)` [16] và `MedianPruner`. Mỗi fold báo cáo 1 giá trị trung gian, nên các trial tệ bị dừng sớm.
- **Objective:** mean CV RMSE trên **giá gốc**, cùng metric dùng để so sánh cuối cùng.
- **Budget:** Housing khoảng 50 trial (chạy CPU local). King County khoảng 30 trial (chạy GPU).
- **Lưu lại:** toàn bộ trial (`optuna_trials.csv`), `best_params.json`, và 2 hình optimization history và param importance.

### 3.6 Đánh giá

| Nội dung | Chi tiết |
|---|---|
| Metric bắt buộc | MAE, RMSE, R² trên giá gốc. CV báo `mean ± std` qua 5 fold; test báo 1 giá trị |
| Bootstrap 95% CI trên test | Resample test 1.000 lần (seed 42) [21]. **Quan trọng với Housing**, vì test chỉ có 109 dòng; chênh lệch nhỏ giữa các model có thể chỉ là nhiễu |
| So sánh giữa 2 dataset | MAE và RMSE **không so được giữa 2 dataset** (khác đơn vị và thang giá). Để so chéo, dùng R², `NRMSE = RMSE / mean(y_test)` và **gap tương đối so với model ML tốt nhất**: `(RMSE_DL − RMSE_bestML) / RMSE_bestML` |
| Thời gian | `fit_time` (train final) và `predict_time` (toàn bộ test). Ghi rõ phần cứng (CPU hay GPU). Chỉ so thời gian giữa các model chạy cùng phần cứng |
| Lưu dự đoán | `predictions_test.csv` (index, y_true, y_pred) để nhóm trưởng tự làm thêm phân tích hoặc kiểm định |

### 3.7 Giải thích model (chung cho cả 2)

- **Permutation importance** [19] trên test, đo bằng mức tăng RMSE khi xáo trộn từng feature, lặp 5 lần (seed 42). Phương pháp này không phụ thuộc model, nên so được trực tiếp với feature importance của nhóm ML.
- Mỗi model có thêm cách giải thích riêng: attention map cho FT-Transformer, mask cho TabNet (mô tả ở mục của từng model).
- Tuỳ chọn nếu dư thời gian: SHAP [20] cho model tốt nhất.

### 3.8 Kiểm tra chống leakage (`tests/test_no_leakage.py`)

- Không index test nào nằm trong train hoặc fold.
- Các fold không chồng lấn và phủ đủ toàn bộ train.
- Preprocessor chỉ thấy `inner_train` (kiểm tra `scaler.mean_` khớp với mean của `inner_train`).
- `final.py` là module duy nhất đọc `test_idx.csv`.

---

## 4. FT-Transformer

### 4.1 Phương pháp

FT-Transformer (Feature Tokenizer + Transformer) [1] biến **mỗi feature thành 1 token** rồi cho các token tương tác qua self-attention [2]:

- **Feature Tokenizer:** feature số $x_j$ thành $T_j = b_j + x_j \cdot W_j$. Feature categorical thành $T_j = b_j + e_j^\top W_j$ (lookup embedding). Thêm token `[CLS]` vào đầu chuỗi.
- **Transformer:** $L$ block, mỗi block gồm Multi-Head Self-Attention và FFN dạng ReGLU [3], dùng PreNorm (LayerNorm trước sub-layer; block đầu tiên bỏ LayerNorm ở nhánh attention) và residual dropout.
- **Đầu ra:** $\hat{y} = \text{Linear}(\text{ReLU}(\text{LayerNorm}(T^{[CLS]}_L)))$.
- **Độ phức tạp:** $O(k^2 d)$ theo số feature $k$. Không đáng lo ở đây vì $k$ chỉ khoảng 12–22.
- **Bối cảnh:** khác TabTransformer [4] (chỉ đưa feature categorical qua Transformer), FT-Transformer token hoá **cả feature số**. Điều này phù hợp với 2 dataset này, nơi đa số feature là số. Hướng mở rộng: thay tokenizer tuyến tính bằng embedding tuần hoàn/piecewise-linear cho feature số [5].

### 4.2 Vì sao chọn

- [1] kết luận FT-Transformer là kiến trúc DL tổng quát mạnh nhất trong benchmark của họ. Benchmark đó có **California Housing**, bài toán giá nhà cỡ ~20k dòng, tương đương King County.
- FT-Transformer là đại diện "DL thuần học từ dữ liệu", nên trả lời trực tiếp được RQ1 và RQ2.

### 4.3 Input

- Dùng `kind="numcat"`.
- **Housing:** 11 cột số (gồm cả 6 cột 0/1); `furnishingstatus` là categorical (cardinality 3).
- **King County:** khoảng 21 cột số; `zipcode` là categorical (cardinality 70, cộng 1 index cho giá trị lạ).

### 4.4 Hiện thực

Dùng package chính thức của tác giả `rtdl_revisiting_models` [1] (đã chạy thử constructor và forward pass trong env `house-dl`). Training loop tự viết trong `models/ft_transformer.py`.

```python
from rtdl_revisiting_models import FTTransformer

model = FTTransformer(
    n_cont_features=x_num.shape[1],
    cat_cardinalities=cat_cardinalities,     # [] nếu không có categorical
    d_out=1,
    n_blocks=3, d_block=192, attention_n_heads=8,
    attention_dropout=0.2, ffn_d_hidden=None, ffn_d_hidden_multiplier=4 / 3,
    ffn_dropout=0.1, residual_dropout=0.0,   # = cấu hình mặc định của [1]
)
optimizer = torch.optim.AdamW(model.make_parameter_groups(), lr=1e-4, weight_decay=1e-5)
y_pred = model(x_num, x_cat).squeeze(-1)
```

`make_parameter_groups()` không áp weight decay cho bias, LayerNorm và embedding, giống paper gốc [1, 17].

Training loop:
1. Shuffle mỗi epoch, loss MSE trên target đã chuẩn hoá.
2. Cuối mỗi epoch tính RMSE trên `inner_val`; giữ weight tốt nhất.
3. Dừng nếu sau `patience` epoch không cải thiện [18].
4. Ghi `history_` (train loss, val RMSE) và `best_epoch_`.

### 4.5 Cấu hình training

| | Housing | King County |
|---|---|---|
| Batch size | 32 | 256 |
| Max epochs / patience | 400 / 30 | 200 / 16 |
| Optimizer | AdamW [17] | AdamW |
| Thiết bị | CPU local | GPU (Kaggle/Colab) |

### 4.6 Không gian tuning

Dựa theo không gian tìm kiếm trong phụ lục của [1], thu hẹp lại cho Housing vì dữ liệu quá nhỏ.

| Hyperparameter | Housing | King County | Mặc định [1] |
|---|---|---|---|
| `n_blocks` | {1, 2, 3} | {1, …, 4} | 3 |
| `d_block` (chia hết cho 8 heads) | {32, 64, …, 256} | {64, 128, …, 512} | 192 |
| `attention_dropout` | U[0, 0.5] | U[0, 0.5] | 0.2 |
| `ffn_d_hidden_multiplier` | U[2/3, 8/3] | U[2/3, 8/3] | 4/3 |
| `ffn_dropout` | U[0, 0.5] | U[0, 0.5] | 0.1 |
| `residual_dropout` | {0, U[0, 0.2]} | {0, U[0, 0.2]} | 0.0 |
| `lr` | LogU[1e-5, 1e-3] | LogU[1e-5, 1e-3] | 1e-4 |
| `weight_decay` | LogU[1e-6, 1e-3] | LogU[1e-6, 1e-3] | 1e-5 |
| Scaler | {quantile, standard} | {quantile, standard} | quantile |

Chạy **cấu hình mặc định trước** để có điểm tham chiếu, rồi mới tune.

### 4.7 Giải thích

- **Attention map:** lấy trung bình attention từ token `[CLS]` tới từng feature, qua mọi head, layer và mẫu test. [1] đề xuất cách này như một thước đo feature importance rẻ.
  Lấy attention bằng forward hook trên module attention; nếu package không trả về attention probabilities thì tự viết lại lớp attention (~30 dòng).
- **Permutation importance** (Mục 3.7) để đối chiếu với attention map.

### 4.8 Rủi ro

| Rủi ro | Cách xử lý |
|---|---|
| Overfit nặng trên Housing | Dùng ít block, `d_block` nhỏ, dropout và weight decay cao, batch nhỏ. Nếu vẫn thua thì **đó là kết quả cần báo cáo** (ủng hộ RQ1), không phải lỗi |
| Chạy chậm trên CPU với King County | Chạy tuning trên GPU. Bật `MedianPruner` |
| Kết quả không tái lập | Cố định seed; `torch.use_deterministic_algorithms(True)` khi chạy CPU |

---

## 5. TabNet

### 5.1 Phương pháp

TabNet [6] xử lý dữ liệu qua $N_{steps}$ **bước quyết định tuần tự**. Ở mỗi bước, model chọn một tập nhỏ feature để "nhìn". Cách hoạt động này giống một chuỗi cây quyết định, nhưng khả vi:

- **Attentive transformer** sinh mask thưa $M[i] = \text{sparsemax}(P[i-1] \cdot h_i(a[i-1]))$ [7].
  Prior scale $P[i] = \prod_{j \le i} (\gamma - M[j])$ hạn chế việc dùng lại cùng một feature ở nhiều bước; $\gamma$ càng lớn thì càng dễ dùng lại.
- **Feature transformer** là các khối Linear + Ghost BatchNorm [10] + GLU [9]. Một phần chia sẻ giữa các bước (`n_shared`), một phần riêng từng bước (`n_independent`).
- **Đầu ra:** $\hat{y} = W_{final} \sum_i \text{ReLU}(d[i])$.
- **Loss:** MSE cộng regularization độ thưa $\lambda_{sparse} \cdot \frac{1}{N_{steps} B} \sum_{i,b,j} -M_{b,j}[i] \log(M_{b,j}[i] + \epsilon)$.
- **Giải thích có sẵn:** tổng hợp các mask, có trọng số theo đóng góp của từng bước, cho ra feature importance cho cả từng mẫu lẫn toàn cục.

### 5.2 Vì sao chọn

- TabNet là cầu nối **ý tưởng của cây quyết định với DL**. Vì vậy so sánh trực tiếp với Decision Tree, Random Forest và GBDT của nhóm ML rất có ý nghĩa.
- Có khả năng giải thích **sẵn trong kiến trúc** (đáp ứng yêu cầu "model explanation").
- [12] chỉ ra TabNet thường cần tune nhiều và hay kém XGBoost trên dataset mới. Kiểm chứng điều này trên 2 quy mô là một đóng góp phân tích tốt.

### 5.3 Input

- Dùng `kind="tabnet"`: một ma trận với `cat_idxs` (vị trí các cột categorical) và `cat_dims` (số giá trị của mỗi cột).
- **Embedding categorical:** `cat_emb_dim` = 2 cho `furnishingstatus`, 4–8 cho `zipcode`.
- **Target** phải có shape `(n, 1)`.

### 5.4 Hiện thực

Dùng thư viện `pytorch-tabnet` (DreamQuark), hiện thực của [6]. Wrapper đặt trong `models/tabnet.py`.

```python
from pytorch_tabnet.tab_model import TabNetRegressor

model = TabNetRegressor(
    n_d=16, n_a=16, n_steps=4, gamma=1.3, lambda_sparse=1e-4,
    n_independent=2, n_shared=2, momentum=0.02, mask_type="sparsemax",
    cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=cat_emb_dims,
    optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=2e-2),
    scheduler_fn=torch.optim.lr_scheduler.StepLR, scheduler_params=dict(step_size=50, gamma=0.9),
    seed=36, verbose=0,
)
model.fit(
    X_tr, y_tr.reshape(-1, 1),
    eval_set=[(X_val, y_val.reshape(-1, 1))], eval_metric=["rmse"],
    max_epochs=300, patience=30,
    batch_size=256, virtual_batch_size=64, drop_last=False,
)
best_epoch = model.best_epoch                  # lưu lại để train final
importance = model.feature_importances_        # importance toàn cục
explain_matrix, masks = model.explain(X_test)  # importance từng mẫu + mask từng bước
```

> ⚠️ Mặc định của thư viện là `batch_size=1024`, `virtual_batch_size=128`. Các giá trị này **lớn hơn cả fold-train của Housing** (~300 dòng). Phải hạ xuống, ví dụ batch 32–128 và virtual batch 16–64, và giữ `drop_last=False`.
>
> Final model: gọi `fit` trên full train **không truyền `eval_set`**, với `max_epochs = median(best_epoch_k)`.

### 5.5 Không gian tuning

Dựa theo khuyến nghị của [6] và của thư viện (n_d = n_a trong khoảng 8–64, n_steps 3–10, γ 1.0–2.0, n_independent và n_shared 1–5, momentum 0.01–0.4, lr lớn khoảng 0.02 kèm decay).

| Hyperparameter | Housing | King County |
|---|---|---|
| `n_d = n_a` | {8, 16, 24} | {8, 16, 32, 64} |
| `n_steps` | {3, 4, 5} | {3, …, 8} |
| `gamma` | U[1.0, 2.0] | U[1.0, 2.0] |
| `n_independent`, `n_shared` | {1, 2} | {1, 2, 3} |
| `lambda_sparse` | LogU[1e-6, 1e-2] | LogU[1e-6, 1e-2] |
| `momentum` | U[0.01, 0.4] | U[0.01, 0.4] |
| `lr` | {0.005, 0.01, 0.02} | {0.005, 0.01, 0.02, 0.025} |
| `mask_type` | {sparsemax [7], entmax [8]} | {sparsemax, entmax} |
| `batch_size` / `virtual_batch_size` | {32, 64, 128} / {16, 32} | {256, 512, 1024} / {64, 128, 256} |

### 5.6 Giải thích

- Lấy `feature_importances_` (toàn cục), rồi vẽ heatmap `masks` của từng bước quyết định cho một số căn nhà mẫu.
- Đối chiếu với permutation importance. TabNet có tiêu chí chọn feature *được học* riêng, nên hai kết quả có thể khác nhau; sự khác biệt này cũng đáng phân tích.

### 5.7 Rủi ro

| Rủi ro | Cách xử lý |
|---|---|
| `pytorch-tabnet` ít được bảo trì (bản 4.1.0), có thể xung đột với numpy 2 / torch mới | Pin version trong `requirements.txt`; chạy smoke test ngay khi cài. Phương án dự phòng: TabNet trong `pytorch_tabular` |
| Training không ổn định trên dữ liệu nhỏ (Ghost BN với batch nhỏ) | Giảm `n_steps`, tăng `momentum`, giảm lr |
| Early stopping trên thang target đã chuẩn hoá thay vì giá gốc | Chấp nhận được, vì chỉ dùng để chọn epoch. Metric báo cáo vẫn tính trên giá gốc |

---

## 6. Thí nghiệm bổ sung để trả lời câu hỏi về quy mô

### 6.1 Learning curve theo kích thước train (King County)

**Mục tiêu:** tách riêng hiệu ứng **quy mô dữ liệu** khỏi các khác biệt khác giữa 2 dataset.

- Trong mỗi fold của CV trên King County, lấy mẫu con từ fold-train với $n \in \{436, 1000, 2500, 5000, 10000, \text{full} \approx 13.8\text{k}\}$ (seed 42). Mốc $n = 436$ bằng đúng kích thước train của Housing.
- Train với **hyperparams cố định** (cấu hình tốt nhất trên full data), rồi chấm điểm trên **toàn bộ `fold_val`**. Test set vẫn không bị chạm tới.
- **Hình:** RMSE và R² theo $n$ (trục x dạng log), mỗi model một đường, có dải ±std qua 5 fold.
- **Đề xuất với nhóm trưởng:** gửi `size_subsets.csv` cho nhóm ML để họ chạy cùng các tập con với model tốt nhất của mình (ví dụ Ridge, LightGBM). Khi đó trên cùng một hình sẽ thấy rõ điểm giao nhau giữa DL và ML theo quy mô.
- **Hạn chế cần ghi trong report:** hyperparams được tune trên full data có thể chưa tối ưu cho $n$ nhỏ.

### 6.2 Ablation preprocessing (chạy bằng CV, cấu hình mặc định)

| Ablation | FT-Transformer | TabNet |
|---|---|---|
| Target thô vs `log1p` | ✓ | ✓ |
| `QuantileTransformer` vs `StandardScaler` | ✓ | ✓ |
| Có / không có feature engineering (King County) | ✓ | ✓ |

---

## 7. Hình và bảng cần xuất

**Cho từng model × dataset:**
1. Learning curve (train loss và val RMSE theo epoch) cho 1 fold đại diện.
2. Predicted vs actual trên test (trục log) và histogram residual.
3. Feature importance: permutation importance kèm attention (FT-Transformer) hoặc mask (TabNet).
4. Optuna: optimization history và param importance.

**Tổng hợp 2 model (gửi nhóm trưởng ghép với kết quả ML):**
5. Bảng: model × dataset → CV (mean ± std) và test (giá trị + 95% CI) cho MAE, RMSE, R², kèm `fit_time` và `predict_time`.
6. Biểu đồ cột CV RMSE có error bar, tách theo dataset.
7. Scatter thời gian vs RMSE (đánh đổi accuracy–chi phí).
8. Learning curve theo kích thước (Mục 6.1).

---

## 8. Bàn giao cho nhóm trưởng

```
results/<dataset>/<model>/
├── preprocessing.md        # các bước, áp dụng cho cột nào, fit trên tập nào, lý do
├── best_params.json        # hyperparams tốt nhất + seed + version thư viện
├── optuna_trials.csv       # toàn bộ trial
├── cv_folds.csv            # MAE / RMSE / R² / best_epoch từng fold
├── test_metrics.json       # MAE, RMSE, R², NRMSE, CI 95%, fit_time, predict_time, phần cứng
├── predictions_test.csv    # index, y_true, y_pred
└── figures/
```

Kèm theo: `splits/` (dùng chung với nhóm ML), `requirements.txt` đã pin version, `README.md` với lệnh chạy lại từng bước.

---

## 9. Tài liệu tham khảo

**FT-Transformer và nền tảng Transformer**

[1] Y. Gorishniy, I. Rubachev, V. Khrulkov, A. Babenko. *Revisiting Deep Learning Models for Tabular Data.* NeurIPS 34, 2021. arXiv:2106.11959. Code: https://github.com/yandex-research/rtdl-revisiting-models
[2] A. Vaswani et al. *Attention Is All You Need.* NeurIPS 30, 2017. arXiv:1706.03762.
[3] N. Shazeer. *GLU Variants Improve Transformer.* arXiv:2002.05202, 2020.
[4] X. Huang, A. Khetan, M. Cvitkovic, Z. Karnin. *TabTransformer: Tabular Data Modeling Using Contextual Embeddings.* arXiv:2012.06678, 2020.
[5] Y. Gorishniy, I. Rubachev, A. Babenko. *On Embeddings for Numerical Features in Tabular Deep Learning.* NeurIPS 35, 2022. arXiv:2203.05556.

**TabNet**

[6] S. Ö. Arik, T. Pfister. *TabNet: Attentive Interpretable Tabular Learning.* Proceedings of the AAAI Conference on Artificial Intelligence, 35(8), 6679–6687, 2021. arXiv:1908.07442. Thư viện: https://github.com/dreamquark-ai/tabnet
[7] A. F. T. Martins, R. F. Astudillo. *From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification.* ICML 2016. arXiv:1602.02068.
[8] B. Peters, V. Niculae, A. F. T. Martins. *Sparse Sequence-to-Sequence Models.* ACL 2019. arXiv:1905.05702. *(entmax)*
[9] Y. N. Dauphin, A. Fan, M. Auli, D. Grangier. *Language Modeling with Gated Convolutional Networks.* ICML 2017. arXiv:1612.08083. *(GLU)*
[10] E. Hoffer, I. Hubara, D. Soudry. *Train Longer, Generalize Better: Closing the Generalization Gap in Large Batch Training of Neural Networks.* NeurIPS 30, 2017. arXiv:1705.08741. *(Ghost Batch Normalization)*

**DL vs Classical ML trên dữ liệu bảng (Related Work)**

[11] L. Grinsztajn, E. Oyallon, G. Varoquaux. *Why Do Tree-Based Models Still Outperform Deep Learning on Typical Tabular Data?* NeurIPS 35 Datasets and Benchmarks Track, 2022. arXiv:2207.08815.
[12] R. Shwartz-Ziv, A. Armon. *Tabular Data: Deep Learning Is Not All You Need.* Information Fusion, 81, 84–90, 2022. arXiv:2106.03253.
[13] D. McElfresh, S. Khandagale, J. Valverde, V. Prasad C, G. Ramakrishnan, M. Goldblum, C. White. *When Do Neural Nets Outperform Boosted Trees on Tabular Data?* NeurIPS 36 Datasets and Benchmarks Track, 2023. arXiv:2305.02997.
[14] V. Borisov, T. Leemann, K. Seßler, J. Haug, M. Pawelczyk, G. Kasneci. *Deep Neural Networks and Tabular Data: A Survey.* IEEE Transactions on Neural Networks and Learning Systems, 35(6), 7499–7519, 2024. arXiv:2110.01889.

**Training, tuning, đánh giá, giải thích**

[15] T. Akiba, S. Sano, T. Yanase, T. Ohta, M. Koyama. *Optuna: A Next-generation Hyperparameter Optimization Framework.* KDD 2019. arXiv:1907.10902.
[16] J. Bergstra, R. Bardenet, Y. Bengio, B. Kégl. *Algorithms for Hyper-Parameter Optimization.* NeurIPS 24, 2011. *(TPE)*
[17] I. Loshchilov, F. Hutter. *Decoupled Weight Decay Regularization.* ICLR 2019. arXiv:1711.05101. *(AdamW)*
[18] L. Prechelt. *Early Stopping — But When?* In *Neural Networks: Tricks of the Trade*, LNCS 1524, 55–69, Springer, 1998.
[19] L. Breiman. *Random Forests.* Machine Learning, 45(1), 5–32, 2001. *(permutation importance)*
[20] S. M. Lundberg, S.-I. Lee. *A Unified Approach to Interpreting Model Predictions.* NeurIPS 30, 2017. arXiv:1705.07874. *(SHAP, tuỳ chọn)*
[21] B. Efron, R. J. Tibshirani. *An Introduction to the Bootstrap.* Chapman & Hall, 1993.

**Dataset**

[D1] Yasser H. *Housing Prices Dataset.* Kaggle. https://www.kaggle.com/datasets/yasserh/housing-prices-dataset
[D2] harlfoxem. *House Sales in King County, USA.* Kaggle. https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
