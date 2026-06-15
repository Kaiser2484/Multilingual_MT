"""
tests/test_ui.py  –  Kiểm thử E2E giao diện dịch thuật bằng Playwright
========================================================================
Yêu cầu:
    pip install pytest playwright pytest-playwright
    playwright install chromium

Chạy:
    pytest tests/test_ui.py -v

Điều kiện tiên quyết:
    - Backend API phải đang chạy tại http://localhost:8000
    - Frontend index.html phục vụ qua http://localhost:8000 (static mount)
"""

import pytest
from playwright.sync_api import Page, expect


# ===========================================================================
# PAGE OBJECT MODEL (POM)
# ===========================================================================
class TranslatorPage:
    """
    Đại diện cho trang giao diện dịch thuật.
    Đóng gói toàn bộ selector và thao tác tương tác với trang.
    """

    URL = "http://localhost:8000"

    def __init__(self, page: Page) -> None:
        self._page = page

        # ── Locators ──────────────────────────────────────────────────────
        self.source_textarea = page.locator("#source-text")
        self.target_lang_select = page.locator("#target-lang")
        self.translate_button = page.locator("#btn-translate")
        self.output_div = page.locator("#output-text")
        self.clear_button = page.locator("#btn-clear")
        self.copy_button = page.locator("#btn-copy")
        self.char_count = page.locator("#char-count")

    def navigate(self) -> None:
        """Điều hướng đến trang dịch thuật."""
        self._page.goto(self.URL)
        self._page.wait_for_load_state("networkidle")

    def enter_source_text(self, text: str) -> None:
        """Nhập văn bản nguồn vào textarea."""
        self.source_textarea.fill(text)

    def select_target_language(self, language: str) -> None:
        """
        Chọn ngôn ngữ đích trong dropdown.

        Args:
            language: Một trong "Tiếng Việt", "Tiếng Nhật", "Tiếng Trung", "Tiếng Anh"
                      (khớp với thuộc tính value của <option>, không kể emoji hiển thị)
        """
        self.target_lang_select.select_option(value=language)

    def click_translate(self) -> None:
        """Bấm nút Dịch và chờ kết quả xuất hiện."""
        self.translate_button.click()
        # Chờ output không còn trạng thái loading
        self._page.wait_for_function(
            "!document.getElementById('output-text').classList.contains('loading')",
            timeout=30_000,   # Tối đa 30 giây chờ inference
        )

    def get_output_text(self) -> str:
        """Lấy nội dung kết quả dịch."""
        return self.output_div.inner_text()

    def get_output_state(self) -> str:
        """Lấy class CSS hiện tại của ô output (placeholder | loading | error | '')."""
        return self.output_div.get_attribute("class") or ""

    def click_clear(self) -> None:
        """Bấm nút Xoá để reset giao diện."""
        self.clear_button.click()

    def translate(self, text: str, language: str = "Tiếng Việt") -> str:
        """
        Thực hiện đầy đủ luồng dịch: nhập văn bản → chọn ngôn ngữ → bấm Dịch.

        Args:
            text     : Văn bản tiếng Anh cần dịch.
            language : Ngôn ngữ đích.

        Returns:
            Chuỗi kết quả dịch.
        """
        self.enter_source_text(text)
        self.select_target_language(language)
        self.click_translate()
        return self.get_output_text()


# ===========================================================================
# FIXTURES
# ===========================================================================
@pytest.fixture
def translator(page: Page) -> TranslatorPage:
    """Fixture trả về TranslatorPage đã điều hướng đến trang chính."""
    tp = TranslatorPage(page)
    tp.navigate()
    return tp


# ===========================================================================
# TEST CASES
# ===========================================================================
class TestTranslatorUI:
    """Bộ kiểm thử E2E cho giao diện Dịch thuật Đa ngôn ngữ."""

    # ── TC01: Tải trang thành công ─────────────────────────────────────
    def test_page_loads_successfully(self, translator: TranslatorPage) -> None:
        """Trang phải load được và hiển thị đầy đủ các thành phần chính."""
        expect(translator.source_textarea).to_be_visible()
        expect(translator.target_lang_select).to_be_visible()
        expect(translator.translate_button).to_be_visible()
        expect(translator.output_div).to_have_class("placeholder")

    # ── TC02: Dịch sang Tiếng Việt ─────────────────────────────────────
    def test_translate_english_to_vietnamese(self, translator: TranslatorPage) -> None:
        """
        Kịch bản E2E chính:
        Nhập câu tiếng Anh → chọn Tiếng Việt → bấm Dịch
        → ô kết quả phải hiện văn bản khác 'Kết quả sẽ hiện ở đây...'
        """
        result = translator.translate(
            text     = "Hello, how are you?",
            language = "Tiếng Việt",
        )
        # Kết quả không được là placeholder
        assert result not in ("Kết quả sẽ hiện ở đây...", "Đang dịch...", "")
        # Kết quả không được là thông báo lỗi
        assert not result.startswith("❌")
        # Kết quả phải có độ dài hợp lý (> 2 ký tự)
        assert len(result) > 2

    # ── TC03: Dịch sang Tiếng Nhật ─────────────────────────────────────
    def test_translate_english_to_japanese(self, translator: TranslatorPage) -> None:
        """Dịch sang tiếng Nhật phải trả về chuỗi không rỗng."""
        result = translator.translate(
            text     = "Thank you very much.",
            language = "Tiếng Nhật",
        )
        assert result and not result.startswith("❌")

    # ── TC04: Input rỗng hiện cảnh báo ──────────────────────────────────
    def test_empty_input_shows_warning(self, translator: TranslatorPage) -> None:
        """Bấm Dịch khi ô nhập rỗng phải hiện thông báo cảnh báo (class=error)."""
        translator.translate_button.click()
        expect(translator.output_div).to_have_class("error")
        output = translator.get_output_text()
        assert "⚠️" in output or "Vui lòng" in output

    # ── TC05: Nút Xoá hoạt động ─────────────────────────────────────────
    def test_clear_button_resets_ui(self, translator: TranslatorPage) -> None:
        """Sau khi bấm Xoá, textarea phải trống và output trở về placeholder."""
        translator.enter_source_text("Hello world")
        translator.click_clear()
        expect(translator.source_textarea).to_have_value("")
        expect(translator.output_div).to_have_class("placeholder")

    # ── TC06: Đếm ký tự cập nhật ─────────────────────────────────────────
    def test_char_count_updates_on_input(self, translator: TranslatorPage) -> None:
        """Bộ đếm ký tự phải cập nhật khi người dùng nhập."""
        test_text = "Hello world"
        translator.enter_source_text(test_text)
        expect(translator.char_count).to_have_text(f"{len(test_text)} / 500")

    # ── TC07: Nút Sao chép hiện sau khi dịch ────────────────────────────
    def test_copy_button_visible_after_translation(
        self, translator: TranslatorPage
    ) -> None:
        """Nút Sao chép phải hiện sau khi dịch thành công."""
        translator.translate(text="Good morning.", language="Tiếng Việt")
        if not translator.get_output_text().startswith("❌"):
            expect(translator.copy_button).to_be_visible()

    # ── TC08: Phím tắt Ctrl+Enter ────────────────────────────────────────
    def test_keyboard_shortcut_ctrl_enter(self, translator: TranslatorPage) -> None:
        """Ctrl+Enter trong textarea phải kích hoạt dịch."""
        translator.enter_source_text("Good night.")
        translator.source_textarea.press("Control+Enter")
        # Chờ output thay đổi
        translator._page.wait_for_function(
            "!document.getElementById('output-text').classList.contains('loading')",
            timeout=30_000,
        )
        result = translator.get_output_text()
        assert result != "Kết quả sẽ hiện ở đây..."
