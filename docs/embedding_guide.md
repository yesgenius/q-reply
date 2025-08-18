# Руководство по использованию Embedding модуля

## Основные концепции

### Что такое эмбеддинги?
Эмбеддинги - это числовые представления текста в виде вектора (списка чисел). Они позволяют компьютеру "понимать" смысл текста и сравнивать тексты между собой.

```python
текст = "Привет мир"
эмбеддинг = [0.123, -0.456, 0.789, ...]  # вектор из ~1024 чисел
```

## Когда использовать какую функцию?

### 1. `create_embeddings` - Основная функция

**Используйте когда:**
- У вас небольшое количество текстов (до 100)
- Вам нужен полный контроль над процессом
- Вы обрабатываете тексты в реальном времени

**Примеры:**

```python
# Простой случай - один текст
embedding = create_embeddings("Как работает Python?")

# Несколько текстов
texts = ["текст1", "текст2", "текст3"]
embeddings = create_embeddings(texts)

# С инструкцией для поиска
query_embedding = create_embeddings(
    "Что такое ИИ?",
    task_type="query"  # добавит инструкцию для поиска
)

# Без инструкции для индексации
doc_embedding = create_embeddings(
    "Документ про ИИ",
    task_type="document"  # без инструкции
)
```

### 2. `create_batch_embeddings` - Для больших объемов

**Используйте когда:**
- У вас много текстов (сотни или тысячи)
- Вы обрабатываете данные офлайн (не в реальном времени)
- Вы хотите видеть прогресс обработки

**Примеры:**

```python
# Загрузка большой базы данных
all_documents = load_documents()  # 10,000 документов
embeddings = create_batch_embeddings(
    all_documents,
    batch_size=100,  # по 100 за раз
    task_type="document"
)

# С прогресс-баром (псевдокод)
from tqdm import tqdm
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    batch_embs = create_embeddings(batch)
    progress.update(len(batch))
```

## Параметры функций

### Обязательные параметры
- `texts` - текст или список текстов для обработки

### Необязательные параметры
- `model` - какую модель использовать:
  - `"Embeddings"` - базовая модель (быстрее, без инструкций)
  - `"EmbeddingsGigaR"` - продвинутая модель (поддерживает инструкции)
- `task_type` - тип задачи (автоматически добавит нужную инструкцию):
  - `"document"` - для индексации документов (без инструкции)
  - `"query"` - для поисковых запросов
  - `"similarity"` - для поиска похожих текстов
  - `"paraphrase"` - для поиска перефразировок
- `apply_instruction` - применять ли инструкцию (True/False)
- `custom_instruction` - своя инструкция вместо автоматической

### Параметры для отладки (необязательные)
- `x_request_id` - ID запроса для отслеживания
- `x_session_id` - ID сессии для группировки запросов
- `x_client_id` - ID клиента

## Практические примеры

### Пример 1: Поисковая система

```python
# 1. Индексируем документы (без инструкции)
documents = [
    "Python - язык программирования",
    "JavaScript работает в браузере",
    "SQL используется для баз данных"
]

doc_embeddings = create_embeddings(
    documents,
    task_type="document"  # БЕЗ инструкции для документов
)

# 2. Обрабатываем поисковый запрос (с инструкцией)
query = "Как работать с базами данных?"
query_embedding = create_embeddings(
    query,
    task_type="query"  # С инструкцией для запроса
)

# 3. Ищем похожие документы (псевдокод)
similarities = compute_similarity(query_embedding, doc_embeddings)
best_doc = documents[similarities.argmax()]
```

### Пример 2: Дедупликация вопросов

```python
# Находим дубликаты вопросов
questions = [
    "Как установить Python?",
    "Как инсталлировать питон?",  # дубликат
    "Что такое список в Python?"
]

# Используем инструкцию для парафразов
embeddings = create_embeddings(
    questions,
    task_type="paraphrase"  # более "острый" поиск дубликатов
)

# Сравниваем все со всеми (псевдокод)
for i, emb1 in enumerate(embeddings):
    for j, emb2 in enumerate(embeddings[i+1:], i+1):
        similarity = cosine_similarity(emb1, emb2)
        if similarity > 0.9:
            print(f"Дубликаты: '{questions[i]}' и '{questions[j]}'")
```

### Пример 3: Обработка большого файла

```python
import pandas as pd

# Загружаем большой файл
df = pd.read_excel("questions.xlsx")  # 5000 вопросов
questions = df["question"].tolist()

# Обрабатываем батчами
embeddings = create_batch_embeddings(
    questions,
    batch_size=50,  # API лимит или оптимальный размер
    task_type="document"
)

# Сохраняем результаты
df["embedding"] = embeddings
df.to_pickle("questions_with_embeddings.pkl")
```

### Пример 4: Адаптивный выбор модели

```python
def smart_embedding(texts, is_realtime=False):
    """Умный выбор модели и метода."""
    
    # Преобразуем в список
    if isinstance(texts, str):
        texts = [texts]
    
    # Для реального времени - быстрая модель
    if is_realtime:
        return create_embeddings(
            texts,
            model="Embeddings",  # быстрая модель
            apply_instruction=False  # без инструкций для скорости
        )
    
    # Для больших объемов - батчами
    if len(texts) > 100:
        return create_batch_embeddings(
            texts,
            batch_size=100,
            model="EmbeddingsGigaR"  # точная модель
        )
    
    # Обычный случай
    return create_embeddings(
        texts,
        model="EmbeddingsGigaR"
    )

# Использование
realtime_emb = smart_embedding("Срочный запрос", is_realtime=True)
batch_emb = smart_embedding(large_dataset, is_realtime=False)
```

## Типичные ошибки и их решения

### Ошибка 1: Смешивание инструкций

```python
# НЕПРАВИЛЬНО - документы и запросы с одинаковой инструкцией
doc_emb = create_embeddings(docs, task_type="query")  # ❌
query_emb = create_embeddings(query, task_type="query")

# ПРАВИЛЬНО - разные инструкции для асимметричного поиска
doc_emb = create_embeddings(docs, task_type="document")  # ✅
query_emb = create_embeddings(query, task_type="query")
```

### Ошибка 2: Большие запросы без батчей

```python
# НЕПРАВИЛЬНО - может упасть или зависнуть
huge_list = ["текст"] * 10000
embeddings = create_embeddings(huge_list)  # ❌ Слишком много!

# ПРАВИЛЬНО - используем батчи
embeddings = create_batch_embeddings(huge_list, batch_size=100)  # ✅
```

### Ошибка 3: Неправильная модель для задачи

```python
# НЕПРАВИЛЬНО - Embeddings не поддерживает инструкции
embeddings = create_embeddings(
    text,
    model="Embeddings",  # ❌ Не поддерживает инструкции
    task_type="query"  # Инструкция будет проигнорирована
)

# ПРАВИЛЬНО - используем EmbeddingsGigaR для инструкций
embeddings = create_embeddings(
    text,
    model="EmbeddingsGigaR",  # ✅ Поддерживает инструкции
    task_type="query"
)
```

## Оптимизация производительности

### 1. Кэширование эмбеддингов

```python
import pickle
import hashlib

def get_cached_embedding(text, cache_dir="embeddings_cache"):
    # Создаем уникальный ключ для текста
    text_hash = hashlib.md5(text.encode()).hexdigest()
    cache_file = f"{cache_dir}/{text_hash}.pkl"
    
    # Проверяем кэш
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    
    # Создаем эмбеддинг
    embedding = create_embeddings(text)
    
    # Сохраняем в кэш
    with open(cache_file, "wb") as f:
        pickle.dump(embedding, f)
    
    return embedding
```

### 2. Параллельная обработка (для независимых задач)

```python
from concurrent.futures import ThreadPoolExecutor
import time

def process_category(texts, category):
    """Обработка текстов одной категории."""
    return create_embeddings(
        texts,
        task_type="document",
        x_session_id=f"category_{category}"
    )

# Параллельная обработка разных категорий
categories_data = {
    "tech": tech_texts,
    "finance": finance_texts,
    "health": health_texts
}

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = {
        executor.submit(process_category, texts, cat): cat
        for cat, texts in categories_data.items()
    }
    
    results = {}
    for future in futures:
        category = futures[future]
        results[category] = future.result()
```

## Заключение

- Используйте `create_embeddings` для обычных задач
- Используйте `create_batch_embeddings` для больших объемов
- Правильно выбирайте `task_type` для вашей задачи
- Помните про разницу между документами (без инструкции) и запросами (с инструкцией)
- Модель `EmbeddingsGigaR` нужна только если используете инструкции
