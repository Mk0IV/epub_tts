import os
import sys
import re
import tempfile
import shutil
import wave
from pathlib import Path

from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup

from silero_tts.silero_tts import SileroTTS
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


SAMPLE_RATE = 48000
MAX_CHUNK_LEN = 150  # длина одного куска текста в символах

# какие главы озвучивать:
# "all"        -> все
# "1"          -> только первую
# "3-7"        -> с 3 по 7 включительно
CHAPTERS = "all"

# желаемые голоса по языкам: сначала пробуем первый, если его нет — следующий
VOICE_PREFERENCES = {
    "ru": ["xenia", "baya", "eugene"],
    "en": ["en_0"],
}


# ---------- T5 Text Normalization for Russian ----------

_T5_NORM_MODEL = None
_T5_NORM_TOKENIZER = None
_T5_NORM_DEVICE = None

def init_t5_normalizer():
    """Инициализировать T5-нормализатор для русского."""
    global _T5_NORM_MODEL, _T5_NORM_TOKENIZER, _T5_NORM_DEVICE
    if _T5_NORM_MODEL is not None:
        return
    
    print("Загружаю T5 нормализатор для русского текста...")
    checkpoint = "CrabInHoney/t5-ru-text-normalization-v1"
    _T5_NORM_TOKENIZER = T5Tokenizer.from_pretrained(checkpoint)
    _T5_NORM_MODEL = T5ForConditionalGeneration.from_pretrained(checkpoint)
    _T5_NORM_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _T5_NORM_MODEL.to(_T5_NORM_DEVICE)
    print(f"T5 нормализатор загружен на {_T5_NORM_DEVICE}")


def safe_t5_normalize(text: str, max_input_len: int = 300) -> str:
    """
    T5-нормализация с ограничением длины входа/выхода.
    Разбивает на предложения, если текст слишком длинный.
    """
    init_t5_normalizer()
    
    # Пропускаем T5 для текстов > 5000 символов (экономим время)
    if len(text) > 5000:
        print(f"  Текст слишком большой ({len(text)} символов), пропускаю T5")
        return text
    
    # Если текст короче лимита — обрабатываем целиком
    if len(text) <= max_input_len:
        inputs = _T5_NORM_TOKENIZER(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(_T5_NORM_DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = _T5_NORM_MODEL.generate(
                inputs["input_ids"],
                max_length=512,
                num_beams=4,
                early_stopping=True,
            )
        
        result = _T5_NORM_TOKENIZER.decode(outputs[0], skip_special_tokens=True)
        return result
    
    # Если длинный — разбиваем по предложениям
    sentences = re.split(r'([.!?]\s+)', text)
    normalized_parts = []
    current_batch = ""
    
    for sent in sentences:
        if len(current_batch) + len(sent) <= max_input_len:
            current_batch += sent
        else:
            if current_batch:
                normalized_parts.append(safe_t5_normalize(current_batch, max_input_len))
            current_batch = sent
    
    if current_batch:
        normalized_parts.append(safe_t5_normalize(current_batch, max_input_len))
    
    result = " ".join(normalized_parts)
    return result


# ---------- нормализация текста перед TTS ----------

def normalize_spaces(text: str) -> str:
    text = text.replace("\r", "")
    text = text.replace("\u00a0", " ").replace("\t", " ")
    text = re.sub(r" {2,}", " ", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def normalize_lists_and_glossary(text: str) -> str:
    """Лёгкое выравнивание глоссариев, без изменения терминов."""
    lines = text.split("\n")
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            new_lines.append(line)
            continue

        m = re.match(r"^([A-Za-zА-Яа-яЁё0-9\-]+)\s*[–-]\s*(.+)$", stripped)
        if m:
            term, rest = m.groups()
            term = term.strip()
            rest = rest.strip()
            if not rest.endswith((".", "!", "?", "…")):
                rest = rest + "."
            new_lines.append(f"{term} — {rest}")
            continue

        new_lines.append(line)
    return "\n".join(new_lines)


def expand_numbers_km(text: str) -> str:
    """'10 км' -> '10 километров' (простая эвристика)."""
    return re.sub(r"(\d+)\s*км\b", r"\1 километров", text, flags=re.IGNORECASE)


def normalize_dialogs(text: str) -> str:
    """Разнести диалоги по репликам и добавить паузы/интонацию."""
    lines = text.split("\n")
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            new_lines.append(line)
            continue

        if stripped.count("– ") > 1:
            parts = re.split(r"(?=– )", stripped)
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                if not p.endswith(("!", "?", ".", "…")):
                    p = p + "..."
                new_lines.append(p)
        else:
            if stripped.startswith("–") and not stripped.endswith(("!", "?", ".", "…")):
                stripped = stripped + "..."
            new_lines.append(stripped)

    return "\n".join(new_lines)


# заголовки: Глава N..., Глоссарий, Пролог, Эпилог и т.п.
CHAPTER_HEADING_RE = re.compile(
    r"""^(
        Глава\s+\d+\.?\s+.+        |
        Глоссарий                  |
        Пролог                     |
        Эпилог                     |
        Послесловие                |
        Заключение
    )$""",
    flags=re.IGNORECASE | re.VERBOSE
)

def add_pause_after_headings(text: str) -> str:
    """Для заголовков добавляет пустую строку после, чтобы TTS сделал паузу."""
    lines = text.split("\n")
    new_lines = []
    for line in lines:
        new_lines.append(line)
        if CHAPTER_HEADING_RE.match(line.strip()):
            new_lines.append("")
    return "\n".join(new_lines)


# ---------- транслит латиницы в «русский звук» ----------

TRANSLIT_MAP = {
    "a": "а",  "b": "б",  "c": "к",  "d": "д",
    "e": "е",  "f": "ф",  "g": "г",  "h": "х",
    "i": "и",  "j": "дж", "k": "к",  "l": "л",
    "m": "м",  "n": "н",  "o": "о",  "p": "п",
    "q": "к",  "r": "р",  "s": "с",  "t": "т",
    "u": "у",  "v": "в",  "w": "в",  "x": "кс",
    "y": "и",  "z": "з",
}

def translit_latin_word(word: str) -> str:
    res = []
    for ch in word:
        low = ch.lower()
        if low in TRANSLIT_MAP:
            tr = TRANSLIT_MAP[low]
            if ch.isupper():
                if len(tr) == 1:
                    tr = tr.upper()
                else:
                    tr = tr[0].upper() + tr[1:]
            res.append(tr)
        else:
            res.append(ch)
    return "".join(res)

def translit_latin_in_text(text: str) -> str:
    def repl(match: re.Match) -> str:
        word = match.group(0)
        return translit_latin_word(word)

    return re.sub(r"\b\w*?[A-Za-z]\w*\b", repl, text)


# ---------- названия букв и аббревиатуры ----------

LETTER_NAMES = {
    "А": "а",   "Б": "бэ",  "В": "вэ",  "Г": "гэ",
    "Д": "дэ",  "Е": "е",   "Ё": "ё",   "Ж": "жэ",
    "З": "зэ",  "И": "и",   "Й": "и краткое",
    "К": "ка",  "Л": "эль", "М": "эм", "Н": "эн",
    "О": "о",   "П": "пэ",  "Р": "эр", "С": "эс",
    "Т": "тэ",  "У": "у",   "Ф": "эф", "Х": "ха",
    "Ц": "цэ",  "Ч": "че",  "Ш": "ша", "Щ": "ща",
    "Ъ": "твёрдый знак",
    "Ы": "ы",
    "Ь": "мягкий знак",
    "Э": "э",
    "Ю": "ю",
    "Я": "я",
}

def spell_upper_acronym(word: str) -> str:
    """
    Аббревиатура, если:
    1) все буквы заглавные;
    2) длина 2–4;
    3) нет гласных.
    """
    if len(word) < 2 or len(word) > 4:
        return word
    if not word.isupper():
        return word
    if not all("А" <= ch <= "Я" or ch == "Ё" for ch in word):
        return word

    vowels = set("АЕЁИОУЫЭЮЯ")
    if any(ch in vowels for ch in word):
        return word

    parts = []
    for ch in word:
        name = LETTER_NAMES.get(ch, ch.lower())
        parts.append(name.capitalize())
    return "-".join(parts)


def spell_acronyms_in_text(text: str) -> str:
    def repl(match: re.Match) -> str:
        word = match.group(0)
        return spell_upper_acronym(word)

    return re.sub(r"\b[А-ЯЁ]{2,}\b", repl, text)


# ---------- генерация русских годов ----------

UNITS_LOC = {
    1: "первом", 2: "втором", 3: "третьем", 4: "четвёртом",
    5: "пятом", 6: "шестом", 7: "седьмом", 8: "восьмом", 9: "девятом",
}
TEENS_LOC = {
    10: "десятом", 11: "одиннадцатом", 12: "двенадцатом", 13: "тринадцатом",
    14: "четырнадцатом", 15: "пятнадцатом", 16: "шестнадцатом",
    17: "семнадцатом", 18: "восемнадцатом", 19: "девятнадцатом",
}
TENS_LOC = {
    20: "двадцатом", 30: "тридцатом", 40: "сороковом", 50: "пятидесятом",
    60: "шестидесятом", 70: "семидесятом", 80: "восьмидесятом", 90: "девяностом",
}
HUNDREDS_CARD = {
    100: "сто", 200: "двести", 300: "триста", 400: "четыреста",
    500: "пятьсот", 600: "шестьсот", 700: "семьсот",
    800: "восемьсот", 900: "девятьсот",
}

def year_to_locative_ru(year: int) -> str:
    """Грубая генерация 'тысяча девятьсот восемьдесят восьмом' для 1000–2099."""
    if year < 1000 or year > 2099:
        return str(year)

    parts = []

    if 1000 <= year <= 1999:
        parts.append("тысяча")
        rest = year - 1000
    elif 2000 <= year <= 2099:
        parts.append("две тысячи")
        rest = year - 2000
    else:
        rest = year

    h = (rest // 100) * 100
    if h:
        parts.append(HUNDREDS_CARD[h])
        rest = rest % 100

    if rest == 0:
        return " ".join(parts)

    if 10 <= rest <= 19:
        parts.append(TEENS_LOC[rest])
    else:
        t = (rest // 10) * 10
        u = rest % 10
        if t:
            if u == 0:
                parts.append(TENS_LOC[t])
            else:
                TENS_CARD = {
                    20: "двадцать", 30: "тридцать", 40: "сорок",
                    50: "пятьдесят", 60: "шестьдесят",
                    70: "семьдесят", 80: "восемьдесят", 90: "девяносто",
                }
                parts.append(TENS_CARD[t])
        if u:
            parts.append(UNITS_LOC[u])

    return " ".join(parts)


def year_to_cardinal_ru(year: int) -> str:
    """Преобразовать год в кардинальное числительное (именительный падеж)."""
    if year < 1000 or year > 2099:
        return str(year)

    parts = []

    if 1000 <= year <= 1999:
        parts.append("тысяча")
        rest = year - 1000
    elif 2000 <= year <= 2099:
        parts.append("две тысячи")
        rest = year - 2000
    else:
        rest = year

    h = (rest // 100) * 100
    if h:
        parts.append(HUNDREDS_CARD[h])
        rest = rest % 100

    if rest == 0:
        return " ".join(parts)

    UNITS = {
        1: "один", 2: "два", 3: "три", 4: "четыре", 5: "пять",
        6: "шесть", 7: "семь", 8: "восемь", 9: "девять",
    }
    TEENS = {
        10: "десять", 11: "одиннадцать", 12: "двенадцать", 13: "тринадцать",
        14: "четырнадцать", 15: "пятнадцать", 16: "шестнадцать",
        17: "семнадцать", 18: "восемнадцать", 19: "девятнадцать",
    }
    TENS = {
        20: "двадцать", 30: "тридцать", 40: "сорок", 50: "пятьдесят",
        60: "шестьдесят", 70: "семьдесят", 80: "восемьдесят", 90: "девяносто",
    }

    if 10 <= rest <= 19:
        parts.append(TEENS[rest])
    else:
        t = (rest // 10) * 10
        u = rest % 10
        if t:
            parts.append(TENS[t])
        if u:
            parts.append(UNITS[u])

    return " ".join(parts)


def normalize_years_simple(text: str) -> str:
    """
    Преобразует годы в словесную форму.
    '2025' -> 'две тысячи двадцать пять' (для озвучки)
    'в 1925 году' -> 'в тысяча девятьсот двадцать пятом году'
    """
    def repl_year(match: re.Match) -> str:
        year_str = match.group(1)
        year = int(year_str)
        
        # Проверяем контекст: есть ли после года слово "году" или "года"
        full_match = match.group(0)
        
        # Если есть слово "году", "года", "г." — используем локатив (предложный падеж)
        if re.search(r'(году|года|г\.)', full_match):
            return f"{year_str} {year_to_locative_ru(year)}"
        else:
            # Иначе просто читаем как число словами
            return year_to_cardinal_ru(year)
    
    # Паттерн: год от 1000 до 2099, опционально с контекстом "в/году"
    pattern = r'(\d{4})(?:\s+(году|года|г\.))?'
    return re.sub(pattern, repl_year, text)


# ---------- фильтрация неподдерживаемых символов для Silero ----------

def clean_for_silero(text: str) -> str:
    """
    Заменяет символы, которых нет в алфавите модели (в т.ч. '^'),
    на пробел, чтобы избежать KeyError и не сливать слова.
    """
    allowed = (
        "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789"
        " .,!?:;\"'()-%\n"
    )

    res_chars = []
    for ch in text:
        if ch.lower() in allowed:
            res_chars.append(ch)
        else:
            res_chars.append(" ")

    cleaned = "".join(res_chars)
    # Убираем повторяющиеся пробелы
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return cleaned


def preprocess_text_for_tts(text: str, lang_code: str = "ru") -> str:
    """Основной пайплайн препроцессинга с защитой от переполнения."""
    # T5-нормализация только для русского
    if lang_code == "ru":
        try:
            text = safe_t5_normalize(text, max_input_len=300)
        except Exception as e:
            print(f"  Ошибка T5-нормализации: {e}. Продолжаю без неё.")
    
    text = normalize_spaces(text)
    text = normalize_lists_and_glossary(text)
    text = expand_numbers_km(text)
    text = normalize_dialogs(text)
    text = add_pause_after_headings(text)
    text = translit_latin_in_text(text)
    text = spell_acronyms_in_text(text)
    text = normalize_years_simple(text)
    text = clean_for_silero(text)
    
    return text


# ---------- Silero: выбор модели и голоса ----------

def select_model_and_speaker(lang_code: str) -> dict:
    available_langs = SileroTTS.get_available_languages()
    if lang_code not in available_langs:
        raise ValueError(f"Язык {lang_code!r} не поддерживается SileroTTS. Доступные: {available_langs}")

    model_id = SileroTTS.get_latest_model(lang_code)
    if model_id is None:
        raise ValueError(f"Для языка {lang_code!r} модель не найдена")

    tmp_tts = SileroTTS(language=lang_code, model_id=model_id, device="cpu")
    speakers = tmp_tts.get_available_speakers()
    print(f"Доступные голоса для {lang_code}/{model_id}: {speakers}")

    preferred = VOICE_PREFERENCES.get(lang_code, [])
    speaker = None
    for cand in preferred:
        if cand in speakers:
            speaker = cand
            break
    if speaker is None:
        if not speakers:
            raise ValueError(f"Для модели {model_id} языка {lang_code} нет спикеров")
        speaker = speakers[0]

    sample_rates = SileroTTS.get_available_sample_rates_static(lang_code, model_id)
    if SAMPLE_RATE in sample_rates:
        sr = SAMPLE_RATE
    else:
        sr = sample_rates[0]
        print(f"Выбранный SAMPLE_RATE={SAMPLE_RATE} недоступен для {lang_code}/{model_id}, берём {sr}")

    return {"lang": lang_code, "model_id": model_id, "speaker": speaker, "sample_rate": sr}


# ---------- служебные функции EPUB / главы ----------

def detect_epub_language(epub_path: str) -> str:
    """Определить язык EPUB по DC.language, вернуть 'ru'/'en'."""
    book = epub.read_epub(epub_path)
    langs = book.get_metadata("DC", "language")
    if not langs:
        return "ru"

    lang_raw = (langs[0][0] or "").strip().lower()
    if "-" in lang_raw:
        lang_code = lang_raw.split("-")[0]
    elif "_" in lang_raw:
        lang_code = lang_raw.split("_")[0]
    else:
        lang_code = lang_raw

    if lang_code in ("rus", "ru-ru"):
        lang_code = "ru"
    elif lang_code in ("eng", "en-us", "en-gb"):
        lang_code = "en"

    return lang_code


def extract_chapters(epub_path: str, out_dir: Path) -> list[Path]:
    """Извлечь главы EPUB в текстовые файлы (сырой текст)."""
    chapters_dir = out_dir / "chapters"
    chapters_dir.mkdir(parents=True, exist_ok=True)

    book = epub.read_epub(epub_path)

    chapters: list[Path] = []
    idx = 1

    for item in book.get_items():
        if item.get_type() == ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_body_content(), "html.parser")
            text = soup.get_text(separator="\n")

            lines = [line.strip() for line in text.splitlines()]
            lines = [line for line in lines if line]
            text = "\n".join(lines)

            if len(text) < 300:
                continue

            chapter_file = chapters_dir / f"chapter_{idx:03}.txt"
            chapter_file.write_text(text, encoding="utf-8")
            chapters.append(chapter_file)
            idx += 1

    return chapters


def select_chapters(chapter_files: list[Path], spec: str) -> list[Path]:
    """CHAPTERS: 'all', 'N', 'A-B'."""
    if spec == "all":
        return chapter_files

    spec = spec.strip()
    n = len(chapter_files)

    def clamp(i: int) -> int:
        return max(1, min(n, i))

    if "-" in spec:
        a, b = spec.split("-", 1)
        try:
            start = clamp(int(a))
            end = clamp(int(b))
        except ValueError:
            return chapter_files
        if start > end:
            start, end = end, start
        return chapter_files[start - 1 : end]
    else:
        try:
            idx = clamp(int(spec))
        except ValueError:
            return chapter_files
        return [chapter_files[idx - 1]]


def split_by_sentences(text: str, max_len: int) -> list[str]:
    """Разбить текст по предложениям, но не превышая max_len."""
    # Разбиваем по точкам, восклицательным и вопросительным знакам
    sentences = re.split(r'([.!?]+)', text)
    
    parts = []
    current = ""
    
    i = 0
    while i < len(sentences):
        sent = sentences[i].strip()
        punct = ""
        
        # Берём следующий пунктуационный знак, если есть
        if i + 1 < len(sentences):
            punct = sentences[i + 1]
            i += 2
        else:
            i += 1
        
        if not sent:
            continue
        
        candidate = current + (" " if current else "") + sent + punct
        
        if len(candidate) <= max_len:
            current = candidate
        else:
            if current:
                parts.append(current)
            current = sent + punct
    
    if current:
        parts.append(current)
    
    return [p for p in parts if p]


def split_text(text: str, max_len: int) -> list[str]:
    """Разбить текст на куски <= max_len, разбивая по предложениям и диалогам."""
    # Сначала разбиваем по строкам
    lines = text.split("\n")
    parts: list[str] = []

    for line in lines:
        s = line.strip()
        if not s:
            continue

        # Если строка — диалог, добавляем как есть
        if s.startswith("–"):
            if len(s) <= max_len:
                parts.append(s)
            else:
                # Разбиваем длинный диалог по предложениям
                for chunk in split_by_sentences(s, max_len):
                    parts.append(chunk)
            continue

        # Для обычного текста: разбиваем по предложениям
        if len(s) <= max_len:
            parts.append(s)
        else:
            for chunk in split_by_sentences(s, max_len):
                parts.append(chunk)

    return [p for p in parts if p]  # убираем пустые


# ---------- основной код ----------

def main():
    script_dir = Path(__file__).resolve().parent
    in_dir = script_dir / "IN"
    in_dir.mkdir(parents=True, exist_ok=True)

    # выбор файла
    if len(sys.argv) >= 2:
        book_name_arg = sys.argv[1]
        epub_path = (in_dir / book_name_arg).expanduser().resolve()
    else:
        epubs = sorted(in_dir.glob("*.epub"))
        if not epubs:
            print(f"В папке IN нет .epub файлов: {in_dir}")
            sys.exit(1)
        epub_path = epubs[0]
        print(f"Параметр не задан, используем первую книгу из IN: {epub_path.name}")

    if not epub_path.is_file():
        print(f"Файл не найден в папке IN: {epub_path}")
        sys.exit(1)

    base_dir = epub_path.parent
    book_name = epub_path.stem
    out_dir = base_dir / book_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"EPUB: {epub_path}")
    print(f"OUT:  {out_dir}")

    # язык
    lang_code = detect_epub_language(str(epub_path))
    print(f"Язык в метаданных EPUB: {lang_code!r}")

    # выбор модели / голоса
    try:
        cfg = select_model_and_speaker(lang_code)
    except ValueError as e:
        print(e)
        print("Пробуем язык 'ru' по умолчанию.")
        cfg = select_model_and_speaker("ru")

    print(f"Silero конфиг: {cfg}")

    # главы (сырой текст)
    chapter_files = extract_chapters(str(epub_path), out_dir)
    print(f"Найдено глав: {len(chapter_files)}")
    if not chapter_files:
        print("Глава не найдена, проверьте EPUB.")
        return

    # фильтрация по конфигу CHAPTERS
    chapter_files = select_chapters(chapter_files, CHAPTERS)
    print(f"Будет озвучено глав: {len(chapter_files)} (CHAPTERS={CHAPTERS!r})")

    # временная директория для препроцессинга ТЕКСТА
    text_tmp_dir = Path(tempfile.mkdtemp(prefix="silero_book_text_"))
    print("Временная папка для препроцессинга текста:", text_tmp_dir)

    # SileroTTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    tts = SileroTTS(
        language=cfg["lang"],
        model_id=cfg["model_id"],
        speaker=cfg["speaker"],
        sample_rate=cfg["sample_rate"],
        device=device,
    )

    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    try:
        # озвучка
        for chapter_idx, chapter_file in enumerate(chapter_files, start=1):
            chapter_name = chapter_file.stem
            out_wav = audio_dir / f"{chapter_name}.wav"

            print(f"\n=== Глава {chapter_name} ({chapter_idx}/{len(chapter_files)}) ===")
            print(f"RAW TXT:  {chapter_file}")
            print(f"WAV:      {out_wav}")

            raw_text = chapter_file.read_text(encoding="utf-8")
            
            # T5-нормализация (может занять время)
            print("Препроцессинг текста...")
            processed_text = preprocess_text_for_tts(raw_text, lang_code=cfg["lang"])

            # записываем препроцессинг во временный файл
            preproc_path = text_tmp_dir / f"{chapter_name}.preprocessed.txt"
            preproc_path.write_text(processed_text, encoding="utf-8")
            print(f"PREPROCESSED TXT: {preproc_path}")

            text = processed_text
            chunks = split_text(text, MAX_CHUNK_LEN)
            print(f"Текст разделён на {len(chunks)} чанков")

            if out_wav.exists():
                out_wav.unlink()

            if not chunks:
                print("Нет чанков для озвучки, пропускаем.")
                continue

            # временная папка для WAV-чанков главы
            tmp_dir = Path(tempfile.mkdtemp(prefix=f"silero_chunks_{chapter_name}_"))
            chunk_paths: list[Path] = []

            print(f"Озвучиваю чанки...")
            for i, chunk in enumerate(chunks, start=1):
                # Выводим прогресс каждые 50 чанков или в конце
                if i % 50 == 0 or i == len(chunks):
                    print(f"  [{i}/{len(chunks)}] обработано...")
                
                chunk_path = tmp_dir / f"chunk_{i:04}.wav"
                
                try:
                    tts.tts(chunk, str(chunk_path))
                    chunk_paths.append(chunk_path)
                except Exception as e:
                    print(f"  ⚠ ОШИБКА на чанке {i}: {e}")
                    # Пропускаем проблемный чанк
                    continue
            
            print(f"✓ Успешно озвучено {len(chunk_paths)} из {len(chunks)} чанков")

            if not chunk_paths:
                print("✗ Ошибка: нет озвученных чанков!")
                shutil.rmtree(tmp_dir, ignore_errors=True)
                continue

            # Склеивание WAV файлов
            print("Склеиваю аудиофайлы...")
            first = wave.open(str(chunk_paths[0]), "rb")
            params = first.getparams()
            frames = [first.readframes(first.getnframes())]
            first.close()

            for p in chunk_paths[1:]:
                try:
                    w = wave.open(str(p), "rb")
                    frames.append(w.readframes(w.getnframes()))
                    w.close()
                except Exception as e:
                    print(f"  ⚠ Ошибка при чтении {p.name}: {e}")

            with wave.open(str(out_wav), "wb") as wf:
                wf.setparams(params)
                for fr in frames:
                    wf.writeframes(fr)

            print(f"✓ Сохранено: {out_wav.name} ({len(chunk_paths)} чанков)")
            
            # Очистка временной папки
            shutil.rmtree(tmp_dir, ignore_errors=True)
            
            # Освобождение памяти
            del frames
            del chunk_paths
            if device == "cuda":
                torch.cuda.empty_cache()

        print("\n" + "="*60)
        print("✓ Готово!")
        print("="*60)
        print(f"Аудиофайлы глав (WAV) в:\n  {audio_dir.resolve()}")
        print(f"Преобразованный текст глав в:\n  {text_tmp_dir.resolve()}")
        print("Их можно удалить вручную, когда не будут нужны.")

    except KeyboardInterrupt:
        print("\n" + "="*60)
        print("✗ Прервано пользователем.")
        print("="*60)
    finally:
        # Если нужно удалять текстовые временные файлы автоматически — раскомментируй:
        # shutil.rmtree(text_tmp_dir, ignore_errors=True)
        pass


if __name__ == "__main__":
    main()
