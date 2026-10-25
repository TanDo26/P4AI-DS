# Preprocessing — housing / tabnet

## Bước stateless (từng dòng, không học tham số)
- 6 cột yes/no → 0/1
- `area` → `log(area)`
- `bedrooms`, `bathrooms`, `stories`, `parking` giữ dạng số
- `furnishingstatus` là categorical

## Bước stateful (fit trên inner_train trong CV; trên toàn bộ train cho model cuối)

| Bước | Cột | Fit trên | Lý do |
|---|---|---|---|
| StandardScaler | area | inner_train / train | Cột liên tục, lệch phải; scaler chọn bằng CV |
| StandardScaler | bedrooms, bathrooms, stories, parking | inner_train / train | Cột thứ tự, ít giá trị |
| Giữ nguyên 0/1 | mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea | — | Cờ nhị phân |
| OrdinalEncoder (+1, index 0 = giá trị lạ) | furnishingstatus | inner_train / train | index nguyên → embedding của TabNet (`cat_idxs`, `cat_dims`) |
| Target: `log1p(price)` rồi chuẩn hoá (μ, σ) | price | inner_train / train | Giá lệch phải; dự đoán được đổi về giá gốc bằng `expm1` trước khi tính metric |
