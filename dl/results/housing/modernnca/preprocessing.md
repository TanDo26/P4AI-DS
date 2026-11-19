# Preprocessing — housing / modernnca

## Bước stateless (từng dòng, không học tham số)
- 6 cột yes/no → 0/1
- `area` → `log(area)`
- `bedrooms`, `bathrooms`, `stories`, `parking` giữ dạng số
- `furnishingstatus` là categorical

## Bước stateful (fit trên inner_train trong CV; trên phần fit 85% của train cho model cuối)

| Bước | Cột | Fit trên | Lý do |
|---|---|---|---|
| QuantileTransformer(output_distribution='normal') | area | inner_train / phần fit 85% | Cột liên tục, lệch phải; scaler chọn bằng CV; sau đó (cùng các cột thứ tự) qua PLR embedding trong ModernNCA |
| StandardScaler | bedrooms, bathrooms, stories, parking | inner_train / phần fit 85% | Cột thứ tự, ít giá trị |
| Giữ nguyên 0/1 | mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea | — | Cờ nhị phân |
| OneHotEncoder (giá trị lạ → toàn 0) | furnishingstatus | inner_train / phần fit 85% | One-hot như cách ModernNCA xử lý categorical trong TALENT |
| Target: `log1p(price)` rồi chuẩn hoá (μ, σ) | price | inner_train / phần fit 85% | Giá lệch phải; dự đoán được đổi về giá gốc bằng `expm1` trước khi tính metric |
