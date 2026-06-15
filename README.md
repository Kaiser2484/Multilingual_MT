# Multilingual Machine Translation with Transformer

Hệ thống dịch thuật máy đa ngôn ngữ (Multilingual Machine Translation) sử dụng kiến trúc Transformer tự xây dựng từ đầu (from scratch) bằng PyTorch. Hệ thống hỗ trợ dịch thuật hai chiều (Bidirectional) giữa Tiếng Anh và 3 ngôn ngữ khác: Tiếng Việt, Tiếng Nhật, và Tiếng Trung.

Giao diện trực quan được xây dựng trên nền tảng Web App (FastAPI + HTML/CSS/JS thuần).

---

## 📌 Tính Năng Nổi Bật

- **Kiến trúc Transformer**: Xây dựng toàn bộ mạng Neural (Encoder, Decoder, Multi-Head Attention) bằng PyTorch thuần.
- **Dịch Đa Chiều (Multi-way)**: Hỗ trợ 6 chiều dịch trong cùng 1 mô hình nhờ các thẻ ngôn ngữ (Language Tags: `<2vi>`, `<2ja>`, `<2zh>`, `<2en>`).
- **Giao diện thân thiện**: Web UI trực quan, tự động hoán đổi ngôn ngữ nguồn/đích, hỗ trợ phím tắt `Ctrl + Enter`.
- **E2E Testing**: Tích hợp Playwright để kiểm thử tự động toàn bộ luồng giao diện UI.

---

## 📂 Cấu Trúc Thư Mục

```text
Multilingual_MT/
├── data/
│   ├── raw/                  # Dữ liệu gốc
│   └── processed/            # Dữ liệu đã tiền xử lý (train.txt, val.txt, test.txt)
├── model_assets/             # Nơi chứa các trọng số mô hình (.pt)
├── notebooks/                # Các file Jupyter Notebook để train/evaluate
├── src/                      # Source code chính
│   ├── models/               # Định nghĩa kiến trúc Transformer, LSTM Baseline
│   ├── static/               # Frontend UI (index.html, style, js)
│   ├── api.py                # Backend API (FastAPI)
│   ├── prepare_data.py       # Kịch bản thu thập và xử lý dữ liệu
│   └── data_utils.py         # DataLoader và các tiện ích dữ liệu
├── tests/                    # Thư mục kiểm thử (E2E với Playwright)
├── tokenizer/                # File cấu hình BPE Tokenizer (tokenizer.json)
├── requirements.txt          # Các thư viện phụ thuộc
└── README.md
```

---

## 🛠️ Cài Đặt (Installation)

**1. Clone dự án và tạo môi trường ảo (Virtual Environment):**
```bash
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

**3. Tải trình duyệt cho E2E Test (nếu cần chạy test):**
```bash
playwright install
```

---

## 🚀 Hướng Dẫn Sử Dụng (Inference)

Khởi chạy Backend (FastAPI server):

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Sau khi chạy lệnh trên, mở trình duyệt và truy cập: [http://localhost:8000](http://localhost:8000). Giao diện web sẽ hiện ra và bạn có thể bắt đầu dịch thuật!

---

## 🧠 Hướng Dẫn Huấn Luyện (Training & Fine-Tuning)

Mô hình được huấn luyện trên Notebook để dễ dàng chạy trên các nền tảng như Google Colab hoặc Kaggle (hỗ trợ GPU).

**Các bước Fine-tune mô hình (Tiếp tục học từ checkpoint cũ):**

1. Mở file `notebooks/train_transformer.ipynb`.
2. Đảm bảo bạn đã có tập dữ liệu mới (đã được cập nhật đầy đủ các chiều thông qua `src/prepare_data.py`).
3. Đảm bảo biến `PRETRAINED_MODEL` trỏ tới file checkpoint `.pt` cũ của bạn.
4. Chạy toàn bộ các Cell trong Notebook. Mô hình sẽ tự động load kiến thức cũ và học tiếp dữ liệu mới ở Learning Rate nhỏ nhằm bảo vệ phân phối đã học.
5. File mô hình tốt nhất (`best_transformer_model.pt`) sẽ tự động được lưu vào thư mục `model_assets/`.

---

## 🧪 Đánh Giá & Kiểm Thử (Evaluation & Testing)

**1. Chạy Evaluation Notebook:**
Để tính các thang điểm chất lượng dịch thuật như **BLEU Score** trên tập test, bạn hãy chạy file:
```bash
jupyter notebook notebooks/evaluate.ipynb
```
*(Trong file này sẽ tự động sinh dữ liệu test, đưa qua mô hình, và so sánh kết quả dịch với ground-truth).*

**2. Chạy E2E Test (Kiểm thử Giao Diện):**
Hệ thống đi kèm bộ End-to-End Test (E2E) tự động thao tác trên trình duyệt để kiểm tra toàn bộ luồng chọn ngôn ngữ, swap ngôn ngữ và dịch câu.

Đảm bảo API server đang chạy (`uvicorn src.api:app...`), sau đó mở một Terminal khác và chạy:
```bash
pytest tests/test_ui.py -v
```
Nếu bạn muốn xem trình duyệt mở lên chạy test (chế độ headed):
```bash
pytest tests/test_ui.py --headed -v
```