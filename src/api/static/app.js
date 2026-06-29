document.addEventListener('DOMContentLoaded', () => {
    const sourceText = document.getElementById('source-text');
    const targetText = document.getElementById('target-text');
    const sourceLang = document.getElementById('source-lang');
    const targetLang = document.getElementById('target-lang');
    const translateBtn = document.getElementById('translate-btn');
    const swapBtn = document.getElementById('swap-btn');
    const clearBtn = document.getElementById('clear-btn');
    const copyBtn = document.getElementById('copy-btn');
    const charCount = document.getElementById('char-count');
    const spinner = document.getElementById('loading-spinner');
    const methodBadge = document.getElementById('method-badge');

    // Char count
    sourceText.addEventListener('input', () => {
        charCount.textContent = `${sourceText.value.length} / 5000`;
        if (sourceText.value.length > 5000) {
            sourceText.value = sourceText.value.substring(0, 5000);
        }
    });

    // Clear Text
    clearBtn.addEventListener('click', () => {
        sourceText.value = '';
        targetText.value = '';
        charCount.textContent = '0 / 5000';
    });

    // Copy Text
    copyBtn.addEventListener('click', () => {
        if (targetText.value) {
            navigator.clipboard.writeText(targetText.value);
            const icon = copyBtn.querySelector('i');
            icon.className = 'fa-solid fa-check';
            icon.style.color = '#10b981';
            setTimeout(() => {
                icon.className = 'fa-regular fa-copy';
                icon.style.color = '';
            }, 2000);
        }
    });

    // Swap Languages
    swapBtn.addEventListener('click', () => {
        const tempLang = sourceLang.value;
        sourceLang.value = targetLang.value;
        targetLang.value = tempLang;

        const tempText = sourceText.value;
        sourceText.value = targetText.value;
        targetText.value = tempText;
    });

    // Translate API Call
    const handleTranslate = async () => {
        const text = sourceText.value.trim();
        if (!text) return;

        spinner.classList.remove('hidden');
        targetText.value = '';

        try {
            const response = await fetch('/api/translate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    text: text,
                    source_lang: sourceLang.value,
                    target_lang: targetLang.value
                })
            });

            if (!response.ok) {
                throw new Error('API Error');
            }

            const data = await response.json();
            targetText.value = data.translated_text;
            methodBadge.textContent = data.method;
            
        } catch (error) {
            console.error('Translation error:', error);
            targetText.value = 'Lỗi kết nối tới máy chủ AI. Vui lòng đảm bảo bạn đang chạy FastAPI.';
        } finally {
            spinner.classList.add('hidden');
        }
    };

    translateBtn.addEventListener('click', handleTranslate);
    
    // Auto translate on Enter (Ctrl + Enter)
    sourceText.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            handleTranslate();
        }
    });
});
