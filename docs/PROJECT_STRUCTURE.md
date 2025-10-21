# Project Structure

```
q-reply/
├── .env
├── .gitignore
├── answer_generator.py
│     * INPUT_FILE_Q = Path('in/Q.xlsx')
│     * INPUT_FILE_QA = Path('in/QA.xlsx')
│     * DATABASE_FILE = Path('storages/qa.duckdb')
│     * OUTPUT_DIR = Path('out')
│     * SHEET_Q = 'Q'
│     * SHEET_T = 'T'
│     * SHEET_CATEGORY = 'CATEGORY'
│     * SHEET_LOG_PREFIX = 'LOG_ANSWER'
│     * SHEET_LOG_PARAMS = 'LOG_ANSWER_PARAMS'
│     * COL_Q_QUESTION = 1
│     * COL_Q_ANSWER = 2
│     * COL_Q_CONFIDENCE = 3
│     * COL_Q_SOURCES = 4
│     * COL_Q_SOURCES_REASONING = 5
│     * COL_Q_S1 = 6
│     * COL_Q_Q1 = 7
│     * COL_Q_A1 = 8
│     * COL_CATEGORY_NAME = 1
│     * COL_CATEGORY_DESC = 2
│     * TOP_K_SIMILAR = 3
│     * SIMILARITY_THRESHOLD = 0.8
│     * USE_CATEGORIES = False
│     * USE_CHAT_HISTORY = False
│     * EMBEDDING_MODEL = 'EmbeddingsGigaR'
│     * EMBEDDING_DIMENSION = 2560
│     * MAX_RETRY_ATTEMPTS_CATEGORY = 3
│     * MAX_RETRY_ATTEMPTS_ANSWER = 3
│     * RETRY_DELAY = 2
│     * LOG_ANSWER = True
│     * LOG_TO_FILE = True
│     * LOG_LEVEL = logging.INFO
│     * START_ROW = 2
│     * SAVE_FREQUENCY = 5
│     * RESUME_FILE = Path('.answer_generator_resume.json')
│     * logger = get_logger(__name__)
│     * def format_json_for_excel(data: Any) -> str
│     * class AnswerGenerator
│     *   def extract_response_content(self, raw_response: Any) -> str
│     *   def validate_input_files(self) -> bool
│     *   def load_conference_topic(self, workbook: Workbook) -> str | None
│     *   def load_categories_from_qa(self) -> dict[str, str]
│     *   def create_log_params_sheet(self, wb: Workbook) -> None
│     *   def create_output_file(self) -> tuple[Workbook, Path]
│     *   def save_resume_state(self, output_file: Path, last_row: int) -> None
│     *   def load_resume_state(self) -> dict[Any, Any] | None
│     *   def clear_resume_state(self) -> None
│     *   def categorize_question_with_retry(self, question: str) -> dict[str, Any] | None
│     *   def search_similar_questions(self, question: str, category: str | None = None) -> list[dict[str, Any]]
│     *   def generate_answer_with_retry(self, question: str, qa_pairs: list[dict[str, Any]]) -> dict[str, Any] | None
│     *   def process_row(self, sheet_q: Worksheet, sheet_log: Worksheet | None, row_idx: int) -> bool
│     *   def run(self) -> bool
│     * def main()
├── category_filler.py
│     * INPUT_FILE = Path('in/QA.xlsx')
│     * OUTPUT_DIR = Path('out')
│     * SHEET_QA = 'QA'
│     * SHEET_CATEGORY = 'CATEGORY'
│     * SHEET_LOG_PREFIX = 'LOG_CATEGORY'
│     * SHEET_LOG_PARAMS = 'LOG_CATEGORY_PARAMS'
│     * COL_CATEGORY_NAME = 1
│     * COL_CATEGORY_DESC = 2
│     * COL_QA_CATEGORY = 1
│     * COL_QA_QUESTION = 2
│     * COL_QA_ANSWER = 3
│     * USE_ANSWER_FOR_CATEGORIZATION = True
│     * MAX_RETRY_ATTEMPTS = 5
│     * RETRY_DELAY = 2
│     * LOG_CATEGORY = True
│     * LOG_TO_FILE = True
│     * LOG_LEVEL = logging.INFO
│     * START_ROW = 2
│     * SAVE_FREQUENCY = 5
│     * RESUME_FILE = Path('.category_filler_resume.json')
│     * logger = get_logger(__name__)
│     * def format_json_for_excel(data: Any) -> str
│     * class CategoryFiller
│     *   def extract_response_content(self, raw_response: Any) -> str
│     *   def validate_input_file(self) -> bool
│     *   def load_categories(self, sheet: Worksheet) -> dict[str, str]
│     *   def validate_qa_data(self, sheet: Worksheet) -> list[int]
│     *   def create_log_params_sheet(self, wb: Workbook) -> None
│     *   def create_output_file(self) -> tuple[Workbook, Path]
│     *   def save_resume_state(self, output_file: Path, last_row: int) -> None
│     *   def load_resume_state(self) -> dict | None
│     *   def clear_resume_state(self) -> None
│     *   def categorize_question_with_retry(self, question: str, answer: str | None = None) -> dict[str, Any]
│     *   def process_row(self, sheet_qa: Worksheet, sheet_log: Worksheet | None, row_idx: int) -> bool
│     *   def run(self) -> bool
│     * def main()
├── certs/
├── db_export.py
│     * OUTPUT_DIR = Path('out')
│     * DB_PATH = 'storages/qa.duckdb'
│     * SHEET_QA = 'QA'
│     * SHEET_EXPORT_INFO = 'EXPORT_INFO'
│     * COL_CATEGORY = 1
│     * COL_QUESTION = 2
│     * COL_ANSWER = 3
│     * LOG_TO_FILE = True
│     * LOG_LEVEL = logging.INFO
│     * START_ROW = 2
│     * INCLUDE_METADATA = True
│     * logger = get_logger(__name__)
│     * class DatabaseExporter
│     *   def validate_database(self) -> bool
│     *   def open_database(self) -> duckdb_qa_store.QADatabaseStore
│     *   def create_output_file(self) -> tuple[Workbook, Path]
│     *   def create_metadata_sheet(self, wb: Workbook, record_count: int) -> None
│     *   def export_records(self, qa_sheet: Worksheet) -> int
│     *   def run(self) -> bool
│     * def main()
├── db_update.py
│     * INPUT_FILE = Path('in/QA.xlsx')
│     * OUTPUT_DIR = Path('out')
│     * DB_PATH = 'storages/qa.duckdb'
│     * SHEET_QA = 'QA'
│     * SHEET_LOG_PREFIX = 'LOG_DB'
│     * SHEET_LOG_PARAMS = 'LOG_DB_PARAMS'
│     * COL_CATEGORY = 1
│     * COL_QUESTION = 2
│     * COL_ANSWER = 3
│     * EMBEDDING_SIZE = 2560
│     * LOG_DB = True
│     * LOG_TO_FILE = True
│     * LOG_LEVEL = logging.INFO
│     * START_ROW = 2
│     * SAVE_FREQUENCY = 5
│     * RESUME_FILE = Path('.update_db_resume.json')
│     * logger = get_logger(__name__)
│     * class DatabaseUpdater
│     *   def validate_input_file(self) -> bool
│     *   def validate_qa_data(self, sheet: Worksheet) -> bool
│     *   def initialize_database(self) -> duckdb_qa_store.QADatabaseStore
│     *   def create_log_params_sheet(self, wb: Workbook) -> None
│     *   def create_output_file(self) -> tuple[Workbook, Path]
│     *   def save_resume_state(self, output_file: Path, last_row: int) -> None
│     *   def load_resume_state(self) -> dict | None
│     *   def clear_resume_state(self) -> None
│     *   def process_row(self, sheet_qa: Worksheet, sheet_log: Worksheet | None, row_idx: int) -> bool
│     *   def delete_obsolete_records(self, sheet: Worksheet) -> int
│     *   def run(self) -> bool
│     * def main()
├── docs/
│   ├── PROJECT_STRUCTURE.md
│   └── Q-REPLY_REQ.md
├── embeddings/
│   ├── __init__.py
│   └── base_embedding.py
│         * logger = get_logger(__name__)
│         * llm = GigaChatClient()
│         * DEFAULT_MODEL = 'EmbeddingsGigaR'
│         * DEFAULT_MAX_RETRIES = 3
│         * DEFAULT_RETRY_DELAY = 1.0
│         * def get_instruction(task_type: str | None = None, **kwargs: Any) -> str
│         * def prepare_texts(texts: str | list[str], instruction: str = '', **kwargs: Any) -> list[str]
│         * def create_embeddings(texts: str | list[str], model: str = DEFAULT_MODEL, task_type: str | None = None, apply_instruction: bool = True, custom_instruction: str | None = None, max_retries: int = DEFAULT_MAX_RETRIES, retry_delay: float = DEFAULT_RETRY_DELAY, **kwargs: Any) -> tuple[list[list[float]], list[str]]
│         * def create_batch_embeddings(texts: list[str], batch_size: int = 100, model: str = DEFAULT_MODEL, task_type: str | None = None, **kwargs: Any) -> tuple[list[list[float]], list[str]]
├── gigachat/
│   ├── __init__.py
│   ├── client.py
│   │     * logger = get_logger(__name__)
│   │     * class GigaChatClient
│   │     *   def get_access_token(self, force_refresh: bool = False) -> str
│   │     *   def get_models(self, **kwargs: Any) -> list[dict[str, Any]]
│   │     *   def chat_completion(self, messages: list[dict[str, Any]], model: str = 'GigaChat', temperature: float | None = None, top_p: float | None = None, stream: bool = False, max_tokens: int | None = None, repetition_penalty: float | None = None, **kwargs: Any) -> dict[str, Any] | Generator[dict[str, Any], None, None]
│   │     *   def create_embeddings(self, input_texts: str | list[str], model: str = 'Embeddings', **kwargs: Any) -> dict[str, Any]
│   │     *   def close(self) -> None
│   └── config.py
│         * class GigaChatConfig
│         *   def is_basic_auth(self) -> bool
│         *   def is_cert_auth(self) -> bool
│         *   def get_cert_paths(self) -> tuple[str, str] | None
│         *   def get_verify_path(self) -> str | None
│         * def load_config(env_file: str | None = '.env') -> GigaChatConfig
├── in/
├── judge_answer.py
│     * INPUT_FILE_QT = Path('in/QT.xlsx')
│     * INPUT_FILE_QA = Path('in/QA.xlsx')
│     * OUTPUT_DIR = Path('out')
│     * SHEET_Q = 'Q'
│     * SHEET_QA = 'QA'
│     * SHEET_LOG_PREFIX = 'LOG_JUDGEMENT'
│     * SHEET_LOG_PARAMS = 'LOG_JUDGEMENT_PARAMS'
│     * COL_Q_QUESTION = 1
│     * COL_Q_ANSWER = 2
│     * COL_QA_QUESTION = 2
│     * COL_QA_ANSWER = 3
│     * MAX_RETRY_ATTEMPTS_JUDGEMENT = 3
│     * RETRY_DELAY = 2
│     * LOG_JUDGEMENT = True
│     * LOG_TO_FILE = True
│     * LOG_LEVEL = logging.INFO
│     * START_ROW = 2
│     * SAVE_FREQUENCY = 3
│     * RESUME_FILE = Path('.judge_answer_resume.json')
│     * logger = get_logger(__name__)
│     * def format_json_for_excel(data: Any) -> str
│     * class JudgeAnswer
│     *   def extract_response_content(self, raw_response: Any) -> str
│     *   def validate_input_files(self) -> bool
│     *   def load_reference_data(self) -> bool
│     *   def create_log_params_sheet(self, wb: Workbook) -> None
│     *   def create_output_file(self) -> tuple[Workbook, Path]
│     *   def save_resume_state(self, output_file: Path, last_row: int) -> None
│     *   def load_resume_state(self) -> dict[Any, Any] | None
│     *   def clear_resume_state(self) -> None
│     *   def judge_answer_with_retry(self, question: str, reference_answer: str, candidate_answer: str) -> dict[str, Any]
│     *   def process_row(self, sheet_q: Worksheet, sheet_log: Worksheet | None, row_idx: int) -> bool
│     *   def calculate_aggregates(self) -> dict[str, Any]
│     *   def run(self) -> bool
│     * def main()
├── out/
├── prompts/
│   ├── __init__.py
│   ├── base_prompt.py
│   │     * logger = get_logger(__name__)
│   │     * llm = GigaChatClient()
│   │     * def update_system_prompt(**kwargs: Any) -> str
│   │     * def update_chat_history(**kwargs: Any) -> list[dict[str, str]]
│   │     * def add_to_chat_history(message: dict[str, str]) -> None
│   │     * def clear_cache() -> None
│   │     * def reset_to_defaults() -> None
│   │     * def get_messages(user_question: str) -> list[dict[str, str]]
│   │     * def run(user_question: str, custom_params: dict[str, Any] | None = None) -> str | Generator[dict[str, Any], None, None]
│   ├── get_answer_prompt.py
│   │     * logger = get_logger(__name__)
│   │     * llm = GigaChatClient()
│   │     * knowledge_base = (Path(__file__).stem + '_kbase.txt',)
│   │     * class QAPair(TypedDict)
│   │     * def update_system_prompt(**kwargs: Any) -> str
│   │     * def update_chat_history(use_history: bool) -> None
│   │     * def get_messages(user_question: str, qa_pairs: list[QAPair]) -> list[dict[str, str]]
│   │     * def run(user_question: str, qa_pairs: list[QAPair], custom_params: dict[str, Any] | None = None) -> tuple[str, list[dict[str, str]], dict[str, Any]]
│   ├── get_answer_prompt_test.py
│   │     * current_dir = Path(__file__).parent
│   │     * parent_dir = current_dir.parent
│   │     * def save_messages(messages: list[dict[str, str]], filename: str, test_dir: Path) -> None
│   │     * def run_tests() -> None
│   ├── get_category_prompt.py
│   │     * logger = get_logger(__name__)
│   │     * llm = GigaChatClient()
│   │     * def update_system_prompt(**kwargs: Any) -> str
│   │     * def update_chat_history(**kwargs: Any) -> list[dict[str, str]]
│   │     * def add_to_chat_history(message: dict[str, str]) -> None
│   │     * def get_messages(user_question: str, answer: str | None = None) -> list[dict[str, str]]
│   │     * def run(user_question: str, answer: str | None = None, custom_params: dict[str, Any] | None = None) -> tuple[str, list[dict[str, str]], dict[str, Any]]
│   ├── get_category_prompt_test.py
│   │     * current_dir = Path(__file__).parent
│   │     * parent_dir = current_dir.parent
│   │     * def save_messages(messages: list[dict[str, str]], filename: str, test_dir: Path) -> None
│   │     * def run_tests() -> None
│   ├── get_judgement_prompt.py
│   │     * logger = get_logger(__name__)
│   │     * llm = GigaChatClient()
│   │     * JUSTIFICATION_MAX_WORDS = 40
│   │     * EVIDENCE_MAX_ITEMS = 2
│   │     * SCORE_SCALE = 100
│   │     * NUMERICAL_TOLERANCE_ABSOLUTE = 1e-06
│   │     * NUMERICAL_TOLERANCE_RELATIVE = 0.02
│   │     * CONTRADICTION_PENALTY = 0.2
│   │     * HALLUCINATION_PENALTY = 0.1
│   │     * THRESHOLD_GOOD = 85
│   │     * THRESHOLD_OK = 70
│   │     * def update_system_prompt(**kwargs: Any) -> str
│   │     * def update_chat_history(**kwargs: Any) -> list[dict[str, str]]
│   │     * def get_messages(question: str, reference_answer: str, candidate_answer: str) -> list[dict[str, str]]
│   │     * def run(question: str, reference_answer: str, candidate_answer: str, custom_params: dict[str, Any] | None = None) -> tuple[str, list[dict[str, str]], dict[str, Any]]
│   └── get_judgement_prompt_test.py
│         * logger = get_logger(__name__)
│         * class TestResult
│         *   def fail(self, message: str) -> None
│         * def test_initialization() -> TestResult
│         * def test_perfect_match() -> TestResult
│         * def test_partial_match() -> TestResult
│         * def test_contradiction_detection() -> TestResult
│         * def test_hallucination_detection() -> TestResult
│         * def test_empty_candidate() -> TestResult
│         * def test_numerical_tolerance() -> TestResult
│         * def test_invalid_json_response() -> TestResult
│         * def test_incomplete_llm_response() -> TestResult
│         * def test_empty_reference() -> TestResult
│         * def test_numerical_tolerance_boundaries() -> TestResult
│         * def test_unit_conversions() -> TestResult
│         * def test_llm_api_exceptions() -> TestResult
│         * def test_input_validation() -> TestResult
│         * def test_custom_params_merge() -> TestResult
│         * def test_evidence_array_validation() -> TestResult
│         * def test_f1_metric_calculation() -> TestResult
│         * def test_score_thresholds() -> TestResult
│         * def test_combined_penalties() -> TestResult
│         * def test_multiline_answers() -> TestResult
│         * def test_long_answers() -> TestResult
│         * def test_always_returns_tuple() -> TestResult
│         * def test_evaluation_scale_calibration() -> TestResult
│         * def test_precision_with_extra_items() -> TestResult
│         * def test_contradiction_with_hallucination() -> TestResult
│         * def test_semantic_equivalence_synonyms() -> TestResult
│         * def test_partial_contradiction() -> TestResult
│         * def test_element_order_independence() -> TestResult
│         * def test_nested_information_structures() -> TestResult
│         * def run_all_tests() -> tuple[int, int]
│         * def main() -> None
├── pyproject.toml
├── requirements.txt
├── setup.py
│     * PROJECT_URL = 'https://github.com/yesgenius/q-reply/archive/refs/heads/main.zip'
│     * VENV_DIR = '.venv'
│     * TEMP_ARCHIVE = 'project_archive.zip'
│     * def print_status(message: str, level: str = 'INFO') -> None
│     * def fail_fast(message: str) -> NoReturn
│     * def download_archive(url: str, destination: str) -> None
│     * def merge_directories(src: Path, dst: Path) -> None
│     * def extract_archive(archive_path: str, target_dir: Path) -> None
│     * def setup_virtual_environment(venv_path: str) -> None
│     * def install_requirements(venv_path: str, requirements_file: str) -> None
│     * def cleanup_archive(archive_path: str) -> None
│     * def get_activation_commands() -> list[str]
│     * def main() -> None
├── storages/
│   ├── __init__.py
│   ├── duckdb_qa_store.py
│   │     * DEFAULT_EMBEDDING_SIZE = 2560
│   │     * DEFAULT_SIMILARITY_THRESHOLD = 0.5
│   │     * DEFAULT_TOP_K = 5
│   │     * logger = get_logger(__name__)
│   │     * class QADatabaseStore
│   │     *   def preprocess_text(text: str) -> str
│   │     *   def find_question(self, question: str) -> dict[str, Any] | None
│   │     *   def insert_qa(self, question: str, answer: str, category: str | None, question_embedding: Sequence[float], answer_embedding: Sequence[float]) -> bool
│   │     *   def update_qa(self, question: str, answer: str, answer_embedding: Sequence[float]) -> bool
│   │     *   def update_category(self, question: str, category: str | None) -> bool
│   │     *   def search_similar_questions(self, question_embedding: Sequence[float], category: str | None = None, top_k: int = DEFAULT_TOP_K, threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> list[dict[str, Any]]
│   │     *   def get_all_qa_records(self) -> list[dict[str, Any]]
│   │     *   def get_categories(self) -> list[str]
│   │     *   def get_distinct_categories_from_qa(self) -> list[str]
│   │     *   def get_qa_without_category(self) -> list[dict[str, Any]]
│   │     *   def delete_missing_records(self, current_questions: Sequence[str]) -> int
│   │     *   def clear_all_records(self) -> bool
│   │     *   def close(self) -> None
│   └── duckdb_qa_store_test.py
│         * logger = logging.getLogger(__name__)
│         * def generate_fake_embedding(text: str, size: int) -> list[float]
│         * def duckdb_supports_array_cosine(db: QADatabaseStore) -> bool
│         * def main() -> None
└── utils/
    ├── __init__.py
    └── logger.py
          * DEFAULT_FORMAT = '%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s'
          * DEFAULT_LEVEL = logging.INFO
          * class LoggerSetup
          *   def setup(self, log_file: Path | None = None, level: int = DEFAULT_LEVEL, format_string: str = DEFAULT_FORMAT, force_reconfigure: bool = False) -> None
          *   def close(self) -> None
          * def setup_logging(log_file: Path | None = None, level: int = DEFAULT_LEVEL, format_string: str = DEFAULT_FORMAT, force_reconfigure: bool = False) -> None
          * def get_logger(name: str) -> logging.Logger
          * def close_logging() -> None
```
