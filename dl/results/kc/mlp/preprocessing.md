# Preprocessing — kc / mlp

## Bước stateless (từng dòng, không học tham số)
- Bỏ `id`; từ `date` tạo `days_since_start` (số ngày từ 02/05/2014) và `sale_month`
- Sửa `bedrooms = 33 → 3` (lỗi nhập liệu)
- `house_age = năm bán − yr_built` (clip ≥ 0); `years_since_update = năm bán − max(yr_built, yr_renovated)` (clip ≥ 0)
- `is_renovated`, `has_basement` (cờ 0/1); `dist_downtown_km` (khoảng cách tới trung tâm Seattle)
- `log1p` cho `sqft_living`, `sqft_lot`, `sqft_basement`, `sqft_living15`, `sqft_lot15`
- Bỏ `sqft_above` (= sqft_living − sqft_basement), `yr_built`, `yr_renovated`
- `zipcode` là categorical

## Bước stateful (fit trên inner_train trong CV; trên phần fit 85% của train cho model cuối)

| Bước | Cột | Fit trên | Lý do |
|---|---|---|---|
| StandardScaler | sqft_living, sqft_lot, sqft_basement, sqft_living15, sqft_lot15, lat, long, dist_downtown_km, days_since_start, house_age, years_since_update | inner_train / phần fit 85% | StandardScaler cố định như code MLP gốc |
| StandardScaler | bedrooms, bathrooms, floors, view, condition, grade, sale_month | inner_train / phần fit 85% | Cột thứ tự, ít giá trị |
| Giữ nguyên 0/1 | waterfront, is_renovated, has_basement | — | Cờ nhị phân |
| OneHotEncoder (giá trị lạ → toàn 0) | zipcode | inner_train / phần fit 85% | MLP không có lớp embedding, như code MLP gốc |
| Target: `log1p(price)` rồi chuẩn hoá (μ, σ) | price | inner_train / phần fit 85% | Giá lệch phải; dự đoán được đổi về giá gốc bằng `expm1` trước khi tính metric |
