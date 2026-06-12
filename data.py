import os
import sys

# Fix UnicodeEncodeError trên Windows (console CP1252 không in được emoji)
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from datasets import load_dataset



# ---------------------------------------------------------------------------
# FIX: Helsinki-NLP/tatoeba_mt dùng custom dataset script (.py) không còn
#      được hỗ trợ trên datasets >= 2.x mới nhất.
#      → Thay bằng Helsinki-NLP/opus-100 (lưu dạng Parquet chuẩn, hoạt động
#        với mọi phiên bản datasets hiện tại).
#      Config name: "en-ja", "en-zh" — translation dict giữ nguyên cấu trúc.
# ---------------------------------------------------------------------------

def download_and_save_dataset(lang_pair, src_lang, tgt_lang, max_lines=100000):
    """
    Tải dữ liệu song ngữ EN-{tgt_lang} từ Helsinki-NLP/opus-100 trên HuggingFace
    và lưu thành 2 file plain-text song song vào data/raw/.

    Args:
        lang_pair : Chuỗi cấu hình, ví dụ "en-ja" hoặc "en-zh"
        src_lang  : Mã ngôn ngữ nguồn ("en")
        tgt_lang  : Mã ngôn ngữ đích  ("ja" hoặc "zh")
        max_lines : Số dòng tối đa cần lấy (mặc định 100 000)
    """
    print(f"🔄 Đang tải dữ liệu cặp en-{tgt_lang} từ Helsinki-NLP/opus-100 ...")

    # opus-100 lưu dạng Parquet → không cần trust_remote_code, không có script
    # Thử split "train" trước; một số config nhỏ chỉ có "test"
    dataset = None
    for split in ("train", "test"):
        try:
            dataset = load_dataset(
                "Helsinki-NLP/opus-100",
                lang_pair,          # config name: "en-ja" / "en-zh"
                split=split,
            )
            print(f"   ✓ Dùng split='{split}' ({len(dataset):,} mẫu tổng)")
            break
        except Exception as err:
            print(f"   ⚠ split='{split}' không khả dụng: {err}")

    if dataset is None:
        print(f"❌ Không thể tải dữ liệu cho cặp {lang_pair}. Bỏ qua.")
        return

    src_lines = []
    tgt_lines = []

    count = 0
    for item in dataset:
        if count >= max_lines:
            break

        # Cấu trúc: item["translation"] = {"en": "...", "ja": "..."} 
        translation_dict = item["translation"]
        src_text = translation_dict.get(src_lang, "").strip()
        tgt_text = translation_dict.get(tgt_lang, "").strip()

        # Bỏ qua dòng rỗng
        if not src_text or not tgt_text:
            continue

        src_lines.append(src_text + "\n")
        tgt_lines.append(tgt_text + "\n")
        count += 1

        # Log tiến độ mỗi 10 000 dòng
        if count % 10_000 == 0:
            print(f"   ... đã xử lý {count:,} dòng")

    # Tạo thư mục nếu chưa có
    os.makedirs("data/raw", exist_ok=True)

    # Đường dẫn file ra
    src_file_path = f"data/raw/train.en-{tgt_lang}.en"
    tgt_file_path = f"data/raw/train.en-{tgt_lang}.{tgt_lang}"

    # Ghi file nguồn (Tiếng Anh)
    with open(src_file_path, "w", encoding="utf-8") as f_src:
        f_src.writelines(src_lines)

    # Ghi file đích (Tiếng Nhật / Tiếng Trung)
    with open(tgt_file_path, "w", encoding="utf-8") as f_tgt:
        f_tgt.writelines(tgt_lines)

    print(f"💾 Đã lưu thành công {count:,} dòng vào:")
    print(f"   -> {src_file_path}")
    print(f"   -> {tgt_file_path}\n")


if __name__ == "__main__":
    # Tải dữ liệu Anh - Nhật
    download_and_save_dataset("en-ja", src_lang="en", tgt_lang="ja", max_lines=100_000)

    # Tải dữ liệu Anh - Trung
    download_and_save_dataset("en-zh", src_lang="en", tgt_lang="zh", max_lines=100_000)

    print("🎉 Hoàn thành tải toàn bộ dữ liệu thô cho Nhật và Trung!")