#!/usr/bin/env python3
"""Conference question answer generator script using RAG with DuckDB.

This script generates answers for conference questions using a Retrieval-Augmented
Generation (RAG) approach. It searches for similar questions in a DuckDB database,
retrieves relevant Q&A pairs as context, and generates comprehensive answers using
the GigaChat LLM.

The script processes questions from an Excel file, performs semantic search using
embeddings, optionally categorizes questions, and generates context-aware answers
with detailed logging support.

Usage:
    python answer_generator.py

Configuration:
    Adjust settings at the beginning of the script for IDE execution.
"""

import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import openpyxl
from openpyxl import Workbook
from openpyxl.worksheet.worksheet import Worksheet

# Import required modules
try:
    from db import duckdb_qa_store
    from embeddings import base_embedding
    from prompts import define_category_prompt, get_answer_prompt
except ImportError as e:
    print(f"Error: Could not import required modules: {e}")
    print("Ensure all modules are in the correct paths:")
    print("  - db/duckdb_qa_store.py")
    print("  - embeddings/base_embedding.py")
    print("  - prompts/define_category_prompt.py")
    print("  - prompts/get_answer_prompt.py")
    sys.exit(1)

# ============================================================================
# CONFIGURATION SECTION - Adjust these settings for IDE execution
# ============================================================================

# File paths
INPUT_FILE_Q = Path("in/Q.xlsx")
INPUT_FILE_QA = Path("in/QA.xlsx")
DATABASE_FILE = Path("qa.duckdb")
OUTPUT_DIR = Path("out")

# Sheet names
SHEET_Q = "Q"
SHEET_T = "T"
SHEET_CATEGORY = "CATEGORY"
SHEET_LOG_PREFIX = "LOG_ANSWER"
SHEET_LOG_PARAMS = "LOG_ANSWER_PARAMS"  # New sheet for model parameters

# Column indices for Q sheet (1-based for openpyxl)
COL_Q_QUESTION = 1  # Column A - Question
COL_Q_ANSWER = 2  # Column B - Answer
COL_Q_Q1 = 3  # Column C - First similar question
COL_Q_A1 = 4  # Column D - First similar answer
# Additional columns for Q2/A2, Q3/A3, etc. will be dynamic

# Column indices for CATEGORY sheet (1-based for openpyxl)
COL_CATEGORY_NAME = 1  # Column A
COL_CATEGORY_DESC = 2  # Column B

# Search configuration
TOP_K_SIMILAR = 5  # Number of similar questions to retrieve
SIMILARITY_THRESHOLD = 0.15  # Minimum cosine similarity

# Embedding configuration
EMBEDDING_MODEL = "EmbeddingsGigaR"
EMBEDDING_DIMENSION = 2560

# Retry configuration
MAX_RETRY_ATTEMPTS_CATEGORY = 3  # Maximum retry attempts for categorization
MAX_RETRY_ATTEMPTS_ANSWER = 3  # Maximum retry attempts for answer generation
RETRY_DELAY = 1  # Delay between retries in seconds

# Logging configuration
LOG_ANSWER = True  # Set to False to disable answer logging sheet
LOG_TO_FILE = True  # Enable logging to text file
LOG_LEVEL = logging.INFO

# Processing configuration
START_ROW = 2  # Skip header row
SAVE_FREQUENCY = 3  # Save file every N processed rows

# Resume configuration
RESUME_FILE = Path(".answer_generator_resume.json")  # Hidden file for resume state

# ============================================================================
# END OF CONFIGURATION SECTION
# ============================================================================


class DualLogger:
    """Logger that writes to both console and file."""

    def __init__(self, log_file: Optional[Path] = None):
        """Initialize dual logger.

        Args:
            log_file: Path to log file. If None, only console logging.
        """
        self.log_file = log_file
        self.file_handler = None

        # Configure main logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(LOG_LEVEL)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(LOG_LEVEL)
        formatter = logging.Formatter(
            "[%(asctime)s][%(name)s][%(levelname)s][%(filename)s:%(lineno)d][%(message)s]"
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler if requested
        if log_file and LOG_TO_FILE:
            try:
                log_file.parent.mkdir(parents=True, exist_ok=True)
                self.file_handler = logging.FileHandler(log_file, encoding="utf-8")
                self.file_handler.setLevel(LOG_LEVEL)
                self.file_handler.setFormatter(formatter)
                self.logger.addHandler(self.file_handler)
                self.logger.info(f"Logging to file: {log_file}")
            except Exception as e:
                self.logger.warning(f"Could not create log file: {e}")

    def close(self):
        """Close file handler if exists."""
        if self.file_handler:
            self.file_handler.close()
            self.logger.removeHandler(self.file_handler)


# Global logger instance (will be initialized in AnswerGenerator)
logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Main class for generating answers using RAG approach with DuckDB."""

    def __init__(self):
        """Initialize the answer generator."""
        self.workbook: Optional[Workbook] = None
        self.output_file: Optional[Path] = None
        self.db_store: Optional[duckdb_qa_store.QADatabaseStore] = None
        self.categories: Dict[str, str] = {}
        self.conference_topic: Optional[str] = None
        self.processed_count: int = 0
        self.resume_state: Dict = {}
        self.timestamp: str = ""
        self.dual_logger: Optional[DualLogger] = None

    def _update_logger(self) -> None:
        """Update global logger reference when dual logger is available."""
        global logger
        if self.dual_logger:
            logger = self.dual_logger.logger

    def validate_input_files(self) -> bool:
        """Validate that required input files exist and are accessible.

        Returns:
            True if validation passes, False otherwise.
        """
        # Check Q.xlsx
        if not INPUT_FILE_Q.exists():
            logger.error(f"Input file not found: {INPUT_FILE_Q}")
            return False

        # Check if Q.xlsx is accessible
        try:
            with open(INPUT_FILE_Q, "rb") as f:
                pass
        except PermissionError:
            logger.error(f"Cannot access file: {INPUT_FILE_Q}")
            logger.error("The file might be open in Excel or another program.")
            return False
        except IOError as e:
            logger.error(f"Cannot read file {INPUT_FILE_Q}: {e}")
            return False

        # Check Q.xlsx structure
        try:
            wb = openpyxl.load_workbook(INPUT_FILE_Q, read_only=True)
            if SHEET_Q not in wb.sheetnames:
                logger.error(f"Required sheet '{SHEET_Q}' not found in {INPUT_FILE_Q}")
                wb.close()
                return False
            wb.close()
        except Exception as e:
            logger.error(f"Error reading {INPUT_FILE_Q}: {e}")
            return False

        # Check database file
        if not DATABASE_FILE.exists():
            logger.error(f"Database file not found: {DATABASE_FILE}")
            logger.error("Please run update_db.py first to create the database")
            return False

        return True

    def load_conference_topic(self, workbook: Workbook) -> Optional[str]:
        """Load conference topic from T sheet if available.

        Args:
            workbook: The workbook to read from.

        Returns:
            Conference topic string or None if not available.
        """
        if SHEET_T not in workbook.sheetnames:
            logger.info(
                f"Sheet '{SHEET_T}' not found, proceeding without conference topic"
            )
            return None

        sheet_t = workbook[SHEET_T]
        topic = sheet_t.cell(row=2, column=1).value  # A2 cell

        if topic and str(topic).strip():
            topic_str = str(topic).strip()
            logger.info(f"Conference topic loaded: {topic_str[:100]}...")
            return topic_str
        else:
            logger.info("No conference topic found in T.A2, proceeding without it")
            return None

    def load_categories_from_qa(self) -> Dict[str, str]:
        """Load category dictionary from QA.xlsx if available.

        Returns:
            Dictionary mapping category names to descriptions, or empty dict.
        """
        if not INPUT_FILE_QA.exists():
            logger.info(
                f"File {INPUT_FILE_QA} not found, proceeding without categories"
            )
            return {}

        try:
            wb = openpyxl.load_workbook(INPUT_FILE_QA, read_only=True)

            if SHEET_CATEGORY not in wb.sheetnames:
                logger.info(
                    f"Sheet '{SHEET_CATEGORY}' not found in {INPUT_FILE_QA}, proceeding without categories"
                )
                wb.close()
                return {}

            sheet_category = wb[SHEET_CATEGORY]
            categories = {}

            for row_idx in range(START_ROW, sheet_category.max_row + 1):
                category_name = sheet_category.cell(
                    row=row_idx, column=COL_CATEGORY_NAME
                ).value
                category_desc = sheet_category.cell(
                    row=row_idx, column=COL_CATEGORY_DESC
                ).value

                # Skip empty rows
                if category_name is None and category_desc is None:
                    continue

                # Validate completeness
                if category_name is None or category_desc is None:
                    logger.error(
                        f"Incomplete category data at row {row_idx}. "
                        f"Name: {category_name}, Description: {category_desc}"
                    )
                    wb.close()
                    raise ValueError(
                        "All categories must have both name and description"
                    )

                categories[str(category_name).strip()] = str(category_desc).strip()

            wb.close()

            if categories:
                logger.info(f"Loaded {len(categories)} categories from {INPUT_FILE_QA}")
            else:
                logger.info("No categories found in CATEGORY sheet")

            return categories

        except Exception as e:
            logger.error(f"Error loading categories: {e}")
            return {}

    def create_log_params_sheet(self, wb: Workbook) -> None:
        """Create or update LOG_ANSWER_PARAMS sheet with model parameters.

        Args:
            wb: Workbook to add the sheet to.
        """
        if not LOG_ANSWER:
            return

        # Remove existing sheet if present
        if SHEET_LOG_PARAMS in wb.sheetnames:
            wb.remove(wb[SHEET_LOG_PARAMS])

        # Create new sheet
        params_sheet = wb.create_sheet(SHEET_LOG_PARAMS)

        # Add headers
        params_sheet.cell(row=1, column=1, value="Parameter")
        params_sheet.cell(row=1, column=2, value="Value")
        params_sheet.cell(row=1, column=3, value="Description")

        # Configuration parameters with constant names
        row = 2
        config_params = [
            ("Timestamp", self.timestamp, "Processing start time"),
            ("TOP_K_SIMILAR", TOP_K_SIMILAR, "Number of similar questions to retrieve"),
            ("SIMILARITY_THRESHOLD", SIMILARITY_THRESHOLD, "Minimum cosine similarity"),
            ("EMBEDDING_MODEL", EMBEDDING_MODEL, "Model for embeddings"),
            ("EMBEDDING_DIMENSION", EMBEDDING_DIMENSION, "Embedding vector size"),
            (
                "MAX_RETRY_ATTEMPTS_CATEGORY",
                MAX_RETRY_ATTEMPTS_CATEGORY,
                "Max retries for categorization",
            ),
            (
                "MAX_RETRY_ATTEMPTS_ANSWER",
                MAX_RETRY_ATTEMPTS_ANSWER,
                "Max retries for answer generation",
            ),
            ("RETRY_DELAY", RETRY_DELAY, "Delay between retries (seconds)"),
            ("SAVE_FREQUENCY", SAVE_FREQUENCY, "Save file every N rows"),
            ("LOG_ANSWER", LOG_ANSWER, "Enable detailed logging sheet"),
            ("LOG_TO_FILE", LOG_TO_FILE, "Enable logging to text file"),
            ("START_ROW", START_ROW, "First row to process (skip headers)"),
        ]

        for param_name, param_value, param_desc in config_params:
            params_sheet.cell(row=row, column=1, value=param_name)
            params_sheet.cell(row=row, column=2, value=str(param_value))
            params_sheet.cell(row=row, column=3, value=param_desc)
            row += 1

        # Add separator for category model parameters (always show if module exists)
        try:
            category_params = getattr(define_category_prompt, "params", {})
            if category_params:
                row += 1
                params_sheet.cell(
                    row=row, column=1, value="--- Category Model Parameters ---"
                )
                row += 1

                param_descriptions = {
                    "model": "LLM model name for categorization",
                    "temperature": "Randomness of responses (0-1)",
                    "top_p": "Nucleus sampling threshold",
                    "stream": "Whether to stream responses",
                    "max_tokens": "Maximum tokens in response",
                    "repetition_penalty": "Penalty for repeated tokens",
                }

                for param_name, param_value in category_params.items():
                    params_sheet.cell(row=row, column=1, value=f"category.{param_name}")
                    params_sheet.cell(row=row, column=2, value=str(param_value))
                    params_sheet.cell(
                        row=row, column=3, value=param_descriptions.get(param_name, "")
                    )
                    row += 1
        except AttributeError:
            logger.debug("Category prompt module params not available")

        # Add separator for answer model parameters
        try:
            answer_params = getattr(get_answer_prompt, "params", {})
            if answer_params:
                row += 1
                params_sheet.cell(
                    row=row, column=1, value="--- Answer Model Parameters ---"
                )
                row += 1

                param_descriptions = {
                    "model": "LLM model name for answer generation",
                    "temperature": "Randomness of responses (0-1)",
                    "top_p": "Nucleus sampling threshold",
                    "stream": "Whether to stream responses",
                    "max_tokens": "Maximum tokens in response",
                    "repetition_penalty": "Penalty for repeated tokens",
                }

                for param_name, param_value in answer_params.items():
                    params_sheet.cell(row=row, column=1, value=f"answer.{param_name}")
                    params_sheet.cell(row=row, column=2, value=str(param_value))
                    params_sheet.cell(
                        row=row, column=3, value=param_descriptions.get(param_name, "")
                    )
                    row += 1
        except AttributeError:
            logger.debug("Answer prompt module params not available")

        # Add separator for embedding model
        row += 1
        params_sheet.cell(row=row, column=1, value="--- Embedding Model ---")
        row += 1

        default_embedding_model = getattr(base_embedding, "DEFAULT_MODEL", "Unknown")
        params_sheet.cell(row=row, column=1, value="base_embedding.DEFAULT_MODEL")
        params_sheet.cell(row=row, column=2, value=default_embedding_model)
        params_sheet.cell(
            row=row, column=3, value="Default model for creating embeddings"
        )

        # Auto-adjust column widths
        for column in params_sheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if cell.value:
                        max_length = max(max_length, len(str(cell.value)))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            params_sheet.column_dimensions[column_letter].width = adjusted_width

        logger.info(f"Created '{SHEET_LOG_PARAMS}' sheet with model parameters")

    def create_output_file(self) -> Tuple[Workbook, Path]:
        """Create output file with timestamp.

        Returns:
            Tuple of (workbook, output_path).
        """
        # Ensure output directory exists
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Generate timestamp
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_file = OUTPUT_DIR / f"Q_{self.timestamp}.xlsx"

        # Initialize file logger
        if LOG_TO_FILE:
            log_file = OUTPUT_DIR / f"Q_{self.timestamp}.log"
            self.dual_logger = DualLogger(log_file)
            self._update_logger()

        logger.info(f"Copying {INPUT_FILE_Q} to {output_file}")

        # Copy file to preserve structure
        shutil.copy2(INPUT_FILE_Q, output_file)
        logger.info(f"File copied successfully")

        # Open the copied file for modifications
        wb = openpyxl.load_workbook(output_file)

        # Add headers to Q sheet if not present
        sheet_q = wb[SHEET_Q]

        # Check if headers exist (assuming row 1 is header row)
        if sheet_q.cell(row=1, column=COL_Q_ANSWER).value is None:
            # Add headers for Q sheet
            q_headers = ["Вопрос", "answer"]
            # Add headers for similar Q&A pairs
            for i in range(1, TOP_K_SIMILAR + 1):
                q_headers.extend([f"q_{i}", f"a_{i}"])

            for col_idx, header in enumerate(q_headers, 1):
                sheet_q.cell(row=1, column=col_idx, value=header)

            logger.info(f"Added headers to '{SHEET_Q}' sheet")

        # Handle LOG_ANSWER sheet based on configuration
        log_sheet_name = SHEET_LOG_PREFIX

        if LOG_ANSWER:
            # Remove existing LOG_ANSWER sheet if it exists
            if log_sheet_name in wb.sheetnames:
                logger.info(
                    f"Removing existing '{log_sheet_name}' sheet for fresh start"
                )
                wb.remove(wb[log_sheet_name])

            # Create new LOG_ANSWER sheet
            log_sheet = wb.create_sheet(log_sheet_name)

            # Add headers to LOG_ANSWER sheet with corrected names
            headers = [
                "Вопрос",
                "category",
                "category_confidence",
                "category_reasoning",
                "category_messages",
                "answer",
                "answer_confidence",
                "answer_sources_used",
                "answer_messages",
            ]
            for col_idx, header in enumerate(headers, 1):
                log_sheet.cell(row=1, column=col_idx, value=header)

            logger.info(f"Created '{log_sheet_name}' sheet with headers")

            # Create LOG_ANSWER_PARAMS sheet
            self.create_log_params_sheet(wb)
        else:
            # If LOG_ANSWER is False and sheet exists, remove it
            if log_sheet_name in wb.sheetnames:
                logger.info(f"Removing '{log_sheet_name}' sheet (LOG_ANSWER=False)")
                wb.remove(wb[log_sheet_name])
            if SHEET_LOG_PARAMS in wb.sheetnames:
                logger.info(f"Removing '{SHEET_LOG_PARAMS}' sheet (LOG_ANSWER=False)")
                wb.remove(wb[SHEET_LOG_PARAMS])

        # Save modifications
        wb.save(output_file)
        logger.info(f"Created output file: {output_file}")

        return wb, output_file

    def save_resume_state(self, output_file: Path, last_row: int) -> None:
        """Save resume state to file.

        Args:
            output_file: Path to the output file being processed.
            last_row: Last successfully processed row number.
        """
        state = {
            "output_file": str(output_file),
            "last_row": last_row,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            with open(RESUME_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved resume state at row {last_row}")
        except Exception as e:
            logger.warning(f"Could not save resume state: {e}")

    def load_resume_state(self) -> Optional[Dict]:
        """Load resume state if exists.

        Returns:
            Resume state dictionary or None if not found.
        """
        if not RESUME_FILE.exists():
            return None

        try:
            with open(RESUME_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)

            # Validate output file still exists
            output_path = Path(state["output_file"])
            if output_path.exists():
                logger.info(f"Found resume state from {state['timestamp']}")
                logger.info(f"Will continue from row {state['last_row'] + 1}")
                return state
            else:
                logger.warning("Resume output file not found, starting fresh")
                return None

        except Exception as e:
            logger.warning(f"Could not load resume state: {e}")
            return None

    def clear_resume_state(self) -> None:
        """Clear resume state file."""
        if RESUME_FILE.exists():
            try:
                RESUME_FILE.unlink()
                logger.debug("Cleared resume state")
            except Exception as e:
                logger.warning(f"Could not clear resume state: {e}")

    def categorize_question_with_retry(self, question: str) -> Optional[Dict[str, Any]]:
        """Categorize a question with retry logic on errors.

        Args:
            question: The question to categorize.

        Returns:
            Categorization result dictionary or None if no categories.
        """
        if not self.categories:
            return None

        last_error = None

        for attempt in range(1, MAX_RETRY_ATTEMPTS_CATEGORY + 1):
            try:
                result_json, messages_list = define_category_prompt.run(question)
                result = json.loads(result_json)

                # Add messages for logging
                messages_json = json.dumps(messages_list, ensure_ascii=False, indent=2)
                result["messages"] = messages_json.replace("\\n", "\n")

                if attempt > 1:
                    logger.info(f"Categorization succeeded on attempt {attempt}")

                return result

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Categorization attempt {attempt}/{MAX_RETRY_ATTEMPTS_CATEGORY} failed: {e}"
                )

                if attempt < MAX_RETRY_ATTEMPTS_CATEGORY:
                    if RETRY_DELAY > 0:
                        time.sleep(RETRY_DELAY)
                    continue

        # All attempts failed
        logger.error(
            f"All {MAX_RETRY_ATTEMPTS_CATEGORY} categorization attempts failed"
        )
        return {
            "category": "Error",
            "confidence": 0.0,
            "reasoning": f"All retry attempts failed. Last error: {str(last_error)}",
            "messages": "[]",
        }

    def search_similar_questions(
        self, question: str, category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar questions in the database.

        Args:
            question: The question to search for.
            category: Optional category filter.

        Returns:
            List of similar Q&A pairs.
        """
        if not self.db_store:
            logger.error("Database store not initialized")
            return []

        try:
            # Get question embedding
            embeddings, _ = base_embedding.create_embeddings(
                question, model=EMBEDDING_MODEL, task_type="query"
            )

            if not embeddings or not embeddings[0]:
                logger.error("Failed to create embedding for question")
                return []

            # Search in database
            results = self.db_store.search_similar_questions(
                question_embedding=embeddings[0],
                category=category,
                top_k=TOP_K_SIMILAR,
                threshold=SIMILARITY_THRESHOLD,
            )

            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def generate_answer_with_retry(
        self, question: str, qa_pairs: List[Dict[str, str]]
    ) -> Optional[Dict[str, Any]]:
        """Generate answer with retry logic on errors.

        Args:
            question: The question to answer.
            qa_pairs: Context Q&A pairs.

        Returns:
            Answer result dictionary or None on failure.
        """
        last_error = None

        for attempt in range(1, MAX_RETRY_ATTEMPTS_ANSWER + 1):
            try:
                result_json, messages_list = get_answer_prompt.run(
                    user_question=question, qa_pairs=qa_pairs
                )
                result = json.loads(result_json)

                # Add messages for logging
                messages_json = json.dumps(messages_list, ensure_ascii=False, indent=2)
                result["messages"] = messages_json.replace("\\n", "\n")

                if attempt > 1:
                    logger.info(f"Answer generation succeeded on attempt {attempt}")

                return result

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Answer generation attempt {attempt}/{MAX_RETRY_ATTEMPTS_ANSWER} failed: {e}"
                )

                if attempt < MAX_RETRY_ATTEMPTS_ANSWER:
                    if RETRY_DELAY > 0:
                        time.sleep(RETRY_DELAY)
                    continue

        # All attempts failed
        logger.error(
            f"All {MAX_RETRY_ATTEMPTS_ANSWER} answer generation attempts failed"
        )
        return {
            "answer": f"Failed to generate answer after {MAX_RETRY_ATTEMPTS_ANSWER} attempts. Last error: {str(last_error)}",
            "confidence": 0.0,
            "sources_used": [],
            "messages": "[]",
        }

    def process_row(
        self, sheet_q: Worksheet, sheet_log: Optional[Worksheet], row_idx: int
    ) -> bool:
        """Process a single row from Q sheet.

        Args:
            sheet_q: The Q worksheet.
            sheet_log: The LOG_ANSWER worksheet (if enabled).
            row_idx: Row index to process.

        Returns:
            True if processing was successful, False otherwise.
        """
        try:
            # Get question from Q sheet
            question = sheet_q.cell(row=row_idx, column=COL_Q_QUESTION).value

            if question is None:
                logger.warning(f"Empty question at row {row_idx}, skipping")
                return True

            question_str = str(question).strip()
            question_preview = (
                question_str[:80] + "..." if len(question_str) > 80 else question_str
            )
            logger.info(f"Processing row {row_idx}: {question_preview}")

            # Step 1: Categorize question (optional) with retry
            category_result = None
            selected_category = None

            if self.categories:
                category_result = self.categorize_question_with_retry(question_str)
                if category_result:
                    selected_category = category_result.get("category")
                    if selected_category != "Error":
                        logger.info(
                            f"Categorized as: {selected_category} "
                            f"(confidence: {category_result.get('confidence', 0):.2f})"
                        )

            # Step 2: Search similar questions
            similar_questions = self.search_similar_questions(
                question_str, selected_category
            )

            if not similar_questions:
                logger.warning(f"No similar questions found for row {row_idx}")
                # Write empty result
                sheet_q.cell(
                    row=row_idx, column=COL_Q_ANSWER, value="No similar questions found"
                )
                return True

            logger.info(f"Found {len(similar_questions)} similar questions")

            # Step 3: Prepare context for answer generation
            qa_pairs = [
                {"question": result["question"], "answer": result["answer"]}
                for result in similar_questions
            ]

            # Step 4: Generate answer with retry
            answer_result = self.generate_answer_with_retry(question_str, qa_pairs)

            if not answer_result:
                logger.error(f"Failed to generate answer for row {row_idx}")
                sheet_q.cell(
                    row=row_idx, column=COL_Q_ANSWER, value="Failed to generate answer"
                )
                return False

            # Step 5: Write results to Q sheet
            # Write answer
            sheet_q.cell(
                row=row_idx, column=COL_Q_ANSWER, value=answer_result.get("answer", "")
            )

            # Write similar Q&A pairs
            for i, result in enumerate(similar_questions[:TOP_K_SIMILAR]):
                q_col = COL_Q_Q1 + (i * 2)  # Q1, Q2, Q3, etc.
                a_col = COL_Q_A1 + (i * 2)  # A1, A2, A3, etc.
                sheet_q.cell(row=row_idx, column=q_col, value=result["question"])
                sheet_q.cell(row=row_idx, column=a_col, value=result["answer"])

            # Step 6: Write to LOG_ANSWER sheet if enabled
            if LOG_ANSWER and sheet_log:
                sheet_log.cell(row=row_idx, column=1, value=question_str)

                if category_result:
                    sheet_log.cell(
                        row=row_idx, column=2, value=category_result.get("category", "")
                    )
                    sheet_log.cell(
                        row=row_idx,
                        column=3,
                        value=category_result.get("confidence", ""),
                    )
                    sheet_log.cell(
                        row=row_idx,
                        column=4,
                        value=category_result.get("reasoning", ""),
                    )
                    sheet_log.cell(
                        row=row_idx, column=5, value=category_result.get("messages", "")
                    )

                sheet_log.cell(
                    row=row_idx, column=6, value=answer_result.get("answer", "")
                )
                sheet_log.cell(
                    row=row_idx, column=7, value=answer_result.get("confidence", "")
                )
                sheet_log.cell(
                    row=row_idx,
                    column=8,
                    value=str(answer_result.get("sources_used", [])),
                )
                sheet_log.cell(
                    row=row_idx, column=9, value=answer_result.get("messages", "")
                )

            logger.info(f"Row {row_idx} processed successfully")
            self.processed_count += 1
            return True

        except Exception as e:
            logger.error(f"Error processing row {row_idx}: {e}")
            return False

    def run(self) -> bool:
        """Main execution method.

        Returns:
            True if execution completed successfully, False otherwise.
        """
        try:
            logger.info("=" * 70)
            logger.info("Starting Answer Generator Script")
            logger.info(f"Log Answer: {'ENABLED' if LOG_ANSWER else 'DISABLED'}")
            logger.info(f"Max Retry Attempts (Category): {MAX_RETRY_ATTEMPTS_CATEGORY}")
            logger.info(f"Max Retry Attempts (Answer): {MAX_RETRY_ATTEMPTS_ANSWER}")
            logger.info("=" * 70)

            # Check for resume state
            resume_state = self.load_resume_state()

            if resume_state:
                # Resume from previous run
                logger.info("Resuming from previous run...")
                self.output_file = Path(resume_state["output_file"])

                # Extract timestamp from filename for log file
                filename = self.output_file.stem  # Q_YYYY-MM-DD_HHMMSS
                if filename.startswith("Q_"):
                    self.timestamp = filename[2:]  # Remove "Q_" prefix

                # Initialize file logger for resumed session
                if LOG_TO_FILE:
                    log_file = OUTPUT_DIR / f"Q_{self.timestamp}.log"
                    self.dual_logger = DualLogger(log_file)
                    self._update_logger()
                    logger.info("=" * 70)
                    logger.info("RESUMED SESSION")
                    logger.info("=" * 70)

                self.workbook = openpyxl.load_workbook(self.output_file)
                start_from_row = resume_state["last_row"] + 1
            else:
                # Fresh start
                logger.info("Starting fresh processing...")

                # Step 1: Validate input files
                logger.info("Step 1: Validating input files...")
                if not self.validate_input_files():
                    logger.error("=" * 70)
                    logger.error("FAILED: Cannot proceed with processing")
                    logger.error("=" * 70)
                    return False

                # Step 2: Create output file
                logger.info("Step 2: Creating output file...")
                self.workbook, self.output_file = self.create_output_file()

                start_from_row = START_ROW

            # Step 3: Load conference topic
            logger.info("Step 3: Loading conference topic...")
            self.conference_topic = self.load_conference_topic(self.workbook)

            # Step 4: Load categories
            logger.info("Step 4: Loading categories...")
            self.categories = self.load_categories_from_qa()

            if self.categories:
                logger.info(f"Categories loaded: {', '.join(self.categories.keys())}")
                # Initialize category prompt
                define_category_prompt.update_system_prompt(categories=self.categories)
                logger.info("Category system initialized")

            # Step 5: Initialize database connection
            logger.info("Step 5: Connecting to database...")
            self.db_store = duckdb_qa_store.QADatabaseStore(
                str(DATABASE_FILE), embedding_size=EMBEDDING_DIMENSION
            )
            logger.info("Database connected successfully")

            # Step 6: Initialize answer generation system
            logger.info("Step 6: Initializing answer generation system...")
            get_answer_prompt.update_system_prompt(topic=self.conference_topic)
            logger.info("Answer generation system initialized")

            # Step 7: Process questions
            logger.info("Step 7: Processing questions...")

            sheet_q = self.workbook[SHEET_Q]

            # Get LOG_ANSWER sheet if enabled
            sheet_log = None
            if LOG_ANSWER:
                if SHEET_LOG_PREFIX in self.workbook.sheetnames:
                    sheet_log = self.workbook[SHEET_LOG_PREFIX]
                else:
                    logger.warning(
                        f"LOG_ANSWER is enabled but '{SHEET_LOG_PREFIX}' sheet not found"
                    )

            # Determine rows to process
            rows_to_process = []
            for row_idx in range(start_from_row, sheet_q.max_row + 1):
                question = sheet_q.cell(row=row_idx, column=COL_Q_QUESTION).value
                if question and str(question).strip():
                    rows_to_process.append(row_idx)

            if not rows_to_process:
                logger.info("No questions to process")
                self.clear_resume_state()
                return True

            logger.info(f"Found {len(rows_to_process)} questions to process")
            logger.info("-" * 70)

            # Process each row
            successfully_processed: List[int] = []

            for idx, row_idx in enumerate(rows_to_process, 1):
                # Process row
                success = self.process_row(sheet_q, sheet_log, row_idx)

                if not success:
                    logger.error(f"Failed to process row {row_idx}")
                    # Save progress on error
                    if successfully_processed:
                        try:
                            self.workbook.save(self.output_file)
                            last_successful = successfully_processed[-1]
                            self.save_resume_state(self.output_file, last_successful)
                            logger.info(
                                f"Emergency save after error: saved up to row {last_successful}"
                            )
                        except Exception as save_error:
                            logger.error(
                                f"Could not perform emergency save: {save_error}"
                            )
                    continue

                # Track successful processing
                successfully_processed.append(row_idx)

                # Periodic save
                if idx % SAVE_FREQUENCY == 0:
                    try:
                        self.workbook.save(self.output_file)
                        self.save_resume_state(self.output_file, row_idx)
                        logger.info(
                            f"Progress saved: {idx}/{len(rows_to_process)} rows processed"
                        )
                    except Exception as save_error:
                        logger.error(f"Could not save progress: {save_error}")

                # Final save
                if idx == len(rows_to_process):
                    try:
                        self.workbook.save(self.output_file)
                        self.save_resume_state(self.output_file, row_idx)
                        logger.info(
                            f"Final batch saved: {idx}/{len(rows_to_process)} rows completed"
                        )
                    except Exception as save_error:
                        logger.error(f"Could not save final batch: {save_error}")

            # Final save
            self.workbook.save(self.output_file)
            logger.info(f"Final save completed: {self.output_file}")

            # Clear resume state on successful completion
            self.clear_resume_state()

            # Close database connection
            if self.db_store:
                self.db_store.close()

            # Summary
            logger.info("-" * 70)
            logger.info("Processing completed successfully!")
            logger.info(f"Total rows processed: {self.processed_count}")
            logger.info(f"Output file: {self.output_file}")
            if LOG_ANSWER and sheet_log:
                logger.info(f"Detailed logs saved in '{SHEET_LOG_PREFIX}' sheet")
                logger.info(f"Model parameters saved in '{SHEET_LOG_PARAMS}' sheet")
            if LOG_TO_FILE:
                log_path = OUTPUT_DIR / f"Q_{self.timestamp}.log"
                logger.info(f"Text log saved to: {log_path}")
            logger.info("=" * 70)

            return True

        except KeyboardInterrupt:
            logger.warning("Processing interrupted by user (Ctrl+C)")
            if self.workbook and self.output_file:
                try:
                    self.workbook.save(self.output_file)
                    logger.info(f"Progress saved to: {self.output_file}")
                    logger.info("You can resume processing by running the script again")
                except Exception as e:
                    logger.error(f"Could not save progress: {e}")
            return False

        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)

            # Try to save current state
            if self.workbook and self.output_file:
                try:
                    self.workbook.save(self.output_file)
                    logger.info(f"Emergency save completed: {self.output_file}")
                    logger.info("You can resume processing by running the script again")
                except Exception as save_error:
                    logger.error(f"Could not save file: {save_error}")

            return False

        finally:
            # Close workbook
            if self.workbook:
                try:
                    self.workbook.close()
                except Exception:
                    pass

            # Close database
            if self.db_store:
                try:
                    self.db_store.close()
                except Exception:
                    pass

            # Close log file handler
            if self.dual_logger:
                self.dual_logger.close()


def main():
    """Main entry point for the script."""
    try:
        generator = AnswerGenerator()
        success = generator.run()

        if success:
            logger.info("Script completed successfully")
        else:
            logger.error("Script completed with errors")

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.info("\nScript interrupted by user")
        sys.exit(130)  # Standard exit code for Ctrl+C
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
