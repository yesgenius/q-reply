---
Выступай в роли эксперта по разработке инструментов для обработки данных, включая их отдельные компоненты: обёртки для API и библиотек, а также вспомогательные скрипты.

**Формат взаимодействия**
Ты — *professional solo developer*, который наставляет *beginner solo developer* в создании компонента на Python 3.12+.

**Принципы разработки**:
* **KISS (Keep It Simple, Stupid)** — решения просты, надёжны, легко поддерживаются, без лишней сложности.
* **YAGNI (You Aren't Gonna Need It)** — никакой преждевременной оптимизации.
* **DRY (Don’t Repeat Yourself)** — избегать дублирования кода и данных; всё должно иметь единственный источник правды.
* **Fail Fast** — ошибки должны проявляться как можно раньше, а не прятаться.
* **Stdlib-only**, если это не ведет к увеличению, усложнению кода. 
* Типизация и совместимость с *mypy*.
* Отсутствие требований обратной совместимости.

---
Необходимо максимально просто, но профессионально реализовать обработку массива строк с помощью модуля get_judgement_prompt.py:

1 - Внимательно проанализируй пример реализации модуля answer_generator.py: он обрабатывает массив строк с помощью модуля get_answer_prompt.py; 

2 - Необходимо строго повторить структуру answer_generator.py в модуле judge_answer.py, при этом реализовать обработку массива строк с помощью модуля get_judgement_prompt.py:

**ВХОДНЫЕ ДАННЫЕ**:
- **тестируемые данные**: INPUT_FILE_QT = Path("in/QT.xlsx") - это копия файла INPUT_FILE_Q = Path("in/Q.xlsx"), где на листе SHEET_Q = "Q" даны ответы на вопросы в колонке COL_Q_ANSWER = 2;

- **эталонные данные, построчно соответствующие тестируемым данным** INPUT_FILE_QA = Path("in/QA.xlsx") - это категоризированные и загруженные в векторную БД исторические вопросы и ответы на основе которых сформирован ответ в колонке COL_Q_ANSWER = 2;

**ВЫХОДНЫЕ ДАННЫЕ**:
- candidat_question - из QT_YYYY-MM-DD_HHMMSS, SHEET_Q = "Q", COL_Q_QUESTION = 1
- candidat_answer - из QT_YYYY-MM-DD_HHMMSS, SHEET_Q = "Q", COL_Q_ANSWER = 2
- QT_YYYY-MM-DD_HHMMSS.xlsx - это копия входного файла INPUT_FILE_QT, где на лист SHEET_Q = "Q", начиная с первой свободной колонки добавлены:
	- reference_question из INPUT_FILE_QA, SHEET_QA = "QA", COL_QA_QUESTION = 2
	- reference_answer из INPUT_FILE_QA, SHEET_QA = "QA", COL_QA_ANSWER = 3
	- score из get_judgement_prompt.run(candidat_question, reference_answer, candidate_answer).result_json["score"]
	- class из get_judgement_prompt.run(candidat_question, reference_answer, candidate_answer).result_json["class"]
	- f1 из get_judgement_prompt.run(candidat_question, reference_answer, candidate_answer).result_json["f1"]
	- precision_c_to_r из get_judgement_prompt.run(candidat_question, reference_answer, candidate_answer).result_json["precision_c_to_r"]
	- recall_r_to_c из get_judgement_prompt.run(candidat_question, reference_answer, candidate_answer).result_json["recall_r_to_c"]
	- contradiction из get_judgement_prompt.run(candidat_question, reference_answer, candidate_answer).result_json["contradiction"]
	- hallucination из get_judgement_prompt.run(candidat_question, reference_answer, candidate_answer).result_json["hallucination"]
	- justification из get_judgement_prompt.run(candidat_question, reference_answer, candidate_answer).result_json["justification"]
	- evidence из get_judgement_prompt.run(candidat_question, reference_answer, candidate_answer).result_json["evidence"]
	- penalties из get_judgement_prompt.run(candidat_question, reference_answer, candidate_answer).result_json["penalties"]

**ЛОГИРОВАНИЕ**:
Необходимо повторить логирование по аналогии answer_generator.py:
- **лист LOG_JUDGEMENT**
	- candidat_question - из QT_YYYY-MM-DD_HHMMSS, SHEET_Q = "Q", COL_Q_QUESTION = 1
	- candidat_answer - из QT_YYYY-MM-DD_HHMMSS, SHEET_Q = "Q", COL_Q_ANSWER = 2
	- reference_question из INPUT_FILE_QA, SHEET_QA = "QA", COL_QA_QUESTION = 2
	- reference_answer из INPUT_FILE_QA, SHEET_QA = "QA", COL_QA_ANSWER = 3
	- score из get_judgement_prompt.run(candidat_question, reference_answer, candidate_answer).result_json["score"]
	- class из get_judgement_prompt.run(candidat_question, reference_answer, candidate_answer).result_json["class"]
	- f1 из get_judgement_prompt.run(candidat_question, reference_answer, candidate_answer).result_json["f1"]
	- precision_c_to_r из get_judgement_prompt.run(candidat_question, reference_answer, candidate_answer).result_json["precision_c_to_r"]
	- recall_r_to_c из get_judgement_prompt.run(candidat_question, reference_answer, candidate_answer).result_json["recall_r_to_c"]
	- contradiction из get_judgement_prompt.run(candidat_question, reference_answer, candidate_answer).result_json["contradiction"]
	- hallucination из get_judgement_prompt.run(candidat_question, reference_answer, candidate_answer).result_json["hallucination"]
	- justification из get_judgement_prompt.run(candidat_question, reference_answer, candidate_answer).result_json["justification"]
	- evidence из get_judgement_prompt.run(candidat_question, reference_answer, candidate_answer).result_json["evidence"]
	- penalties из get_judgement_prompt.run(candidat_question, reference_answer, candidate_answer).result_json["penalties"]
	- messages из get_judgement_prompt.run(candidat_question, reference_answer, candidate_answer).messages_list
	- response из get_judgement_prompt.run(candidat_question, reference_answer, candidate_answer).raw_response
	- response_content из get_judgement_prompt.run(candidat_question, reference_answer, candidate_answer).raw_response.content

- **лист LOG_JUDGEMENT_PARAMS**
	- вывести все глобальные переменные по аналогии c answer_generator.py

- **Финальное сообщение лога**
	- повторить винальное сообщение по аналогии answer_generator.py
	- вывести агрегаты: `mean_score`, `median_score`, `stdev_score`, `share_good/ok/bad`, `contradiction_rate`, `hallucination_rate`



====================================================================================================


# LLM-as-Judge

---
## 0) Назначение
Дать численную оценку (0–100) и краткое объяснение того, насколько новый ответ семантически эквивалентен эталону, а также пометить два типа рисков: **противоречия** и **материальные галлюцинации**.
---

## 1) Контракты

### 1.1 Входы

* `question: str`
* `reference_answer: str`  *(далее R)*
* `candidate_answer: str`  *(далее C)*

### 1.2 Конфигурация (только необходимое)

* `determinism = {temperature: 0.0, top_p: 1.0}`
* `justification_max_words = 40`
* `evidence_max_items = 2`
* `score_scale = 100`
* `thresholds = {good: 85, ok: 70}`

### 1.3 Выход судьи (жёсткая JSON-схема)

```json
{
  "precision_c_to_r": 0.0,
  "recall_r_to_c": 0.0,
  "contradiction": false,
  "hallucination": false,
  "justification": "",
  "evidence": [
    {"source": "candidate|reference", "quote": ""},
    {"source": "candidate|reference", "quote": ""}
  ]
}
```

---

## 2) Фиксированные правила (константы спецификации)

* **Числовой допуск (фиксированный):** расхождение считается допустимым, если
  `|C − R| ≤ max(1e-6, 0.02 * |R|)`; иначе — значимое.
* **Единицы измерения:** судья **обязан** приводить простые единицы к базовым без настройки:
  мм↔см↔м↔км; мс↔с↔мин↔ч; мг↔г↔кг; ℃↔K (тривиально по разности), проценты; бит↔байт↔кБ↔МБ↔ГБ. 
  Валюта — **без конверсии курсов**, различие валюты считается значимым.
* **Материальные новые факты:** любые **проверяемые** новые числа/даты/имена/URLs/цены/политики/версии, которые влияют на вывод или решение — считать **галлюцинацией** (если не следуют из вопроса или эталона).

---

## 3) Предобработка

1. Удалить ведущие/замыкающие пробелы, нормализовать переводы строк к `\n`.
2. Ничего не лемматизировать/не менять регистр: смысл важнее формы.

---

## 4) Инструкция судье (один вызов)

**system:**
```
Ты — строгий детерминированный судья.
Оценивай смысл, а не стиль.
Верни **только валидный JSON** по схеме.
Не раскрывай рассуждения.

```

**user (шаблон):**

```
Оцени, насколько КАНДИДАТ (C) близок к ЭТАЛОНУ (R) для ВОПРОСА.

ВОПРОС:
{question}

ЭТАЛОН (R):
{reference_answer}

КАНДИДАТ (C):
{candidate_answer}

ТРЕБУЕМОЕ:
1) Дай две оценки в [0,1]:
   - precision_c_to_r (C→R): какая доля содержания C подтверждается R.
   - recall_r_to_c (R→C): какая доля содержания R покрыта C.

2) Флаги (bool):
   - contradiction: есть ли значимое логическое противоречие R (включая числовые расхождения сверх допуска).
   - hallucination: есть ли в C новые проверяемые факты, не следующие из ВОПРОСА или R, и влияющие на вывод.

ПРАВИЛА:
- Числовой допуск фиксирован: |C−R| ≤ max(1e-6, 0.02*|R|) считается эквивалентом.
- Приводи простые единицы к базовым (мм/см/м/км; мс/с/мин/ч; мг/г/кг; градусам; %, бит/байт/кБ/МБ/ГБ). Валюты не конвертируй.
- Игнорируй стилистику, вежливые фразы и формат, если не влияют на смысл.
- justification ≤ 40 слов. До 2 коротких цитат с указанием source=candidate|reference.

ФОРМАТ ВЫВОДА — только JSON:
{
  "precision_c_to_r": 0.0,
  "recall_r_to_c": 0.0,
  "contradiction": false,
  "hallucination": false,
  "justification": "",
  "evidence": [
    {"source": "candidate|reference", "quote": ""},
    {"source": "candidate|reference", "quote": ""}
  ]
}
```

**Параметры вызова:** `temperature=0.0`, `top_p=1.0`, `max_tokens` с запасом для JSON.

---

## 5) Пост-обработка и метрики

1. **Валидация JSON**: все поля есть; числовые — в `[0,1]`.
2. **Semantic F1**:

   * Если `P=0` и `R=0` ⇒ `F1=0`, иначе `F1 = 2PR/(P+R)`.
3. **Штрафы**:

   * `penalties = (0.20 if contradiction else 0.0) + (0.10 if hallucination else 0.0)` (константы вынести в глобальные переменные)
4. **Score**:

   * `raw = F1 − penalties`
   * `score = round(max(0.0, raw) * score_scale)`  *(по умолчанию ×100)*
5. **Класс**:

   * `good` если `score ≥ 85`; `ok` если `70 ≤ score ≤ 84`; иначе `bad`.(константы вынести в глобальные переменные)

---

## 6) Калиброванный гайд по шкале P и R (для единообразия)

* **1.0** — полная эквивалентность соответствующего направления.
* **0.9** — покрыты все ключевые пункты, упущены лишь несущественные детали.
* **0.8** — 1 ключевая деталь упущена/добавлена, но вывод совпадает.
* **0.6** — часть ядра упущена/добавлена; вывод частично совпадает.
* **0.4** — совпадают отдельные фрагменты, но вывод иной/неполный.
* **0.2** — совпадения эпизодичны.
* **0.0** — смысл не совпадает вовсе.

*(R→C трактуем как полноту покрытия эталона; C→R — как «чистоту» кандидата без лишнего.)*

---

## 7) Правила выставления флагов (формализованные)

### 7.1 `contradiction = true`, если выполнено **любое**:

* Инверсия ключевого утверждения (есть/нет, разрешено/запрещено, выше/ниже порога).
* Число/порог/версия расходится **сверх допусков** из §2.
* Смена сущности (иная модель/алгоритм/протокол/валюта), меняющая вывод.
* Ошибочное приведение единиц, влияющее на вывод.

### 7.2 `hallucination = true`, если выполнено **оба**:

* В C есть **новые проверяемые факты** (числа, даты, имена, URLs, цены, регламенты, версии), отсутствующие в вопросе и R.
* Эти факты **существенны** — влияют на вывод/рекомендацию/решение.

**Не считаем галлюцинацией:** перефраз, структурирование, нейтральные «служебные» фразы, обобщения без новых проверяемых фактов.

---

## 8) Краевые случаи

* **Пустой C:** `P=1.0`, `R=0.0` ⇒ `F1=0`; флаги `false`; `score=0`.
* **Пустой R:** оценка не имеет смысла (нет целевой семантики) — возвращать `P` как долю «непротиворечивости» нельзя; такой кейс помечать в логах и исключать из агрегатов.
* **Списки/многопункты:** требовать у судьи учёта пунктов как атомарных фактов (влияет на P и R).
* **Числа в разном формате:** распознавать `1 234,56` / `1,234.56` / `%`; приводить запятую/точку к единому виду; степени — `×10^n` трактовать корректно.

---

## 9) Агрегация по датасету, выводить в лог

`mean_score`, `median_score`, `stdev_score`, `share_good/ok/bad`, `contradiction_rate`, `hallucination_rate`

---

## 10) Пример расчёта

Судья вернул:

```
precision_c_to_r = 0.9
recall_r_to_c   = 0.8
contradiction   = false
hallucination   = true
```

Расчёт:

* `F1 = 2*0.9*0.8/(0.9+0.8) = 0.8471`
* `penalties = 0.10`
* `score = round(max(0, 0.8471 − 0.10) * 100) = 75` → класс **ok (70–84)**.

---
Ответь в виде полностью работоспособного, профессионально оформленного кода с чистыми и лаконичными комментариями. 
Используй **Google-style docstrings** на английском языке, обеспечивая совместимость с автоматическими системами документирования.
---
