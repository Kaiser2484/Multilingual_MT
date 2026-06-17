# Multilingual Machine Translation with Transformer

Hệ thống dịch thuật máy đa ngôn ngữ (Multilingual Machine Translation) sử dụng kiến trúc Transformer tự xây dựng từ đầu (from scratch) bằng PyTorch. Hệ thống hỗ trợ dịch thuật **12 chiều** giữa 4 ngôn ngữ: Tiếng Anh, Tiếng Việt, Tiếng Nhật, và Tiếng Trung — hoàn toàn offline, không phụ thuộc bất kỳ API dịch thuật ngoài nào.

Giao diện trực quan được xây dựng trên nền tảng Web App (FastAPI + HTML/CSS/JS thuần).

---

## 📌 Tính Năng Nổi Bật

- **Kiến trúc Transformer**: Xây dựng toàn bộ mạng Neural (Encoder, Decoder, Multi-Head Attention) bằng PyTorch thuần, không dùng framework có sẵn.
- **Dịch 12 Chiều (Full Multi-way)**: Hỗ trợ tất cả các cặp ngôn ngữ nhờ Language Tags (`<2vi>`, `<2ja>`, `<2zh>`, `<2en>`) và cơ chế Self-Pivot qua Tiếng Anh.
- **Beam Search Decoding**: Sử dụng Beam Search (beam=5, length penalty=0.7) thay vì Greedy, cho chất lượng dịch tốt hơn.
- **Hoàn toàn Offline**: Không phụ thuộc `deep-translator` hay bất kỳ API dịch ngoài nào — tất cả đều dùng chính model của hệ thống.
- **Giao diện thân thiện**: Web UI trực quan, tự động hoán đổi ngôn ngữ nguồn/đích, hỗ trợ phím tắt `Ctrl + Enter`.
- **E2E Testing**: Tích hợp Playwright để kiểm thử tự động toàn bộ luồng giao diện UI.

---

## 🌐 Các Cặp Ngôn Ngữ Được Hỗ Trợ

| Chiều | Phương thức | Mô tả |
|-------|-------------|-------|
| EN ↔ VI | Trực tiếp | Model dịch thẳng với tag `<2vi>` / `<2en>` |
| EN ↔ JA | Trực tiếp | Model dịch thẳng với tag `<2ja>` / `<2en>` |
| EN ↔ ZH | Trực tiếp | Model dịch thẳng với tag `<2zh>` / `<2en>` |
| VI ↔ JA | Self-Pivot | VI → EN → JA (dùng chính model, 2 bước) |
| VI ↔ ZH | Self-Pivot | VI → EN → ZH (dùng chính model, 2 bước) |
| JA ↔ ZH | Self-Pivot | JA → EN → ZH (dùng chính model, 2 bước) |

---

## 📂 Cấu Trúc Thư Mục

```text
Multilingual_MT/
├── data/
│   ├── raw/                  # Dữ liệu gốc
│   └── processed/            # Dữ liệu đã tiền xử lý (train.txt, val.txt, test.txt)
├── model_assets/             # Trọng số mô hình (quản lý bởi Git LFS)
│   ├── best_transformer_averaged.pt   # Model tốt nhất (Checkpoint Averaging)
│   ├── best_transformer_model.pt      # Model tốt nhất theo Val Loss
│   └── best_baseline_model.pt         # LSTM Baseline để so sánh
├── notebooks/                # Jupyter Notebook để train/evaluate
│   ├── Model_2.ipynb         # Notebook huấn luyện chính (Kaggle/Colab)
│   └── evaluate.ipynb        # Notebook đánh giá BLEU Score
├── src/                      # Source code chính
│   ├── models/               # Định nghĩa kiến trúc Transformer, LSTM Baseline
│   ├── static/               # Frontend UI (index.html)
│   ├── api.py                # Backend API (FastAPI) – Beam Search, Self-Pivot
│   ├── prepare_data.py       # Script thu thập và xử lý dữ liệu đa ngôn ngữ
│   └── data_utils.py         # DataLoader và các tiện ích dữ liệu
├── tests/                    # Thư mục kiểm thử (E2E với Playwright)
├── tokenizer/                # BPE Tokenizer dùng chung (32,000 token)
├── requirements.txt          # Các thư viện phụ thuộc
└── README.md
```

---

## 🛠️ Cài Đặt (Installation)

**1. Clone dự án và tạo môi trường ảo:**
```bash
git clone https://github.com/Kaiser2484/Multilingual_MT.git
cd Multilingual_MT

python -m venv .venv
# Trên Windows:
.venv\Scripts\activate
# Trên Linux/Mac:
source .venv/bin/activate
```

**2. Cài đặt các thư viện cần thiết:**
```bash
pip install -r requirements.txt
```

> **Lưu ý:** Các file model (`.pt`) được lưu trữ qua **Git LFS**. Nếu bạn chưa cài Git LFS, hãy chạy:
> ```bash
> git lfs install
> git lfs pull
> ```

**3. Tải trình duyệt cho E2E Test (nếu cần):**
```bash
playwright install
```

---

## 🚀 Hướng Dẫn Sử Dụng (Inference)

Khởi chạy Backend (FastAPI server):

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Sau khi chạy, truy cập: [http://localhost:8000](http://localhost:8000)

API tự động ưu tiên dùng `best_transformer_averaged.pt` (chất lượng tốt nhất). Nếu không tìm thấy, sẽ fallback về `best_transformer_model.pt`.

---

## 🧠 Hướng Dẫn Huấn Luyện (Training & Fine-Tuning)

Mô hình được huấn luyện trên **Kaggle** (Tesla T4 GPU) qua Notebook `Model_2.ipynb`.

**Quy trình huấn luyện:**

1. Upload dataset lên Kaggle Dataset (bao gồm thư mục `src`, `data`, `tokenizer`, `model_assets`).
2. Mở `notebooks/Model_2.ipynb` trên Kaggle, gắn Dataset vừa tạo.
3. Notebook tự động resume từ checkpoint gần nhất trên Google Drive.
4. Sau khi train xong, tải file checkpoint về và đặt vào thư mục `model_assets/`.

**Checkpoint Averaging** (tổng hợp 5 checkpoint tốt nhất):
```python
import glob, torch

ckpt_files = sorted(glob.glob('model_assets/transformer_ep*.pt'))[-5:]
avg_state = None
for f in ckpt_files:
    state = torch.load(f, map_location='cpu')["model_state"]
    if avg_state is None:
        avg_state = {k: v.float() for k, v in state.items()}
    else:
        for k in avg_state:
            avg_state[k] += state[k].float()
for k in avg_state:
    avg_state[k] /= len(ckpt_files)
torch.save({"model_state": avg_state}, 'model_assets/best_transformer_averaged.pt')
```

---

## 🧪 Đánh Giá & Kiểm Thử (Evaluation & Testing)

**1. Chạy Evaluation Notebook (tính BLEU Score):**
```bash
jupyter notebook notebooks/evaluate.ipynb
```
Notebook tự động so sánh **BLEU Score** của Transformer vs LSTM Baseline trên tập test.

**2. Chạy E2E Test (Playwright):**

Đảm bảo API server đang chạy, sau đó:
```bash
pytest tests/test_ui.py -v
```
Chạy ở chế độ có giao diện (headed mode):
```bash
pytest tests/test_ui.py --headed -v
```

---

## ⚙️ Kiến Trúc Hệ Thống

```
[Web UI]  ──POST /api/translate──►  [FastAPI Backend]
                                          │
                              ┌───────────┴───────────┐
                              │                       │
                         Direct Route           Self-Pivot Route
                      (EN ↔ VI/JA/ZH)       (VI/JA/ZH ↔ VI/JA/ZH)
                              │                       │
                       [Model + Beam Search]   Step1: Src→EN
                                               Step2: EN→Target
                                              (cùng 1 model)
```