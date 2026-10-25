| Dataset | Model | Trials | CV RMSE | CV R² | Test MAE | Test RMSE [95% CI] | Test R² [95% CI] | NRMSE | Fit (s) | Predict (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| housing | FT-Transformer | 15 + 35 pruned | 1,010,087 ± 61,494 | 0.6607 ± 0.0271 | 1,000,278 | 1,409,501 [1,115,264, 1,704,752] | 0.6070 [0.465, 0.722] | 0.281 | 9.6 | 0.01 |
| housing | TabNet | 35 + 15 pruned | 1,085,143 ± 112,323 | 0.6074 ± 0.0654 | 1,130,911 | 1,622,363 [1,275,743, 1,958,275] | 0.4793 [0.312, 0.599] | 0.324 | 5.7 | 0.02 |
| kc | TabNet | 1 (mặc định, chưa tune) | 119,408 ± 3,584 | 0.8888 ± 0.0177 | 71,900 | 263,139 [121,221, 420,777] | 0.5420 [-0.301, 0.903] | 0.479 | 370.3 | 0.33 |

**Phân tích sai số trên test** (chỉ để phân tích — metric chính là bảng trên; chi tiết 10 căn sai nhiều nhất trong `largest_errors_test.csv`)

| Dataset | Model | % SSE từ căn sai nhất | % SSE từ 5 căn sai nhất | max ŷ/y | RMSE nếu bỏ căn sai nhất | R² nếu bỏ căn sai nhất |
|---|---|---|---|---|---|---|
| housing | FT-Transformer | 16.9% | 43.1% | 1.67 | 1,291,089 | 0.6385 |
| housing | TabNet | 12.6% | 43.9% | 2.88 | 1,524,081 | 0.4963 |
| kc | TabNet | 76.1% | 79.1% | 7.77 | 128,777 | 0.8899 |
