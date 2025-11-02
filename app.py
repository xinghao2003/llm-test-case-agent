# app.py
import os
import io
import json
import time
import shutil
import zipfile
import tempfile
import subprocess
import logging
import platform
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set

from dotenv import load_dotenv
import gradio as gr
import yaml
import pathspec
from pathspec import PathSpec

# ---------- Configuration ----------

MAX_TOTAL_TEXT = 250_000
ALLOWED_EXTENSIONS = {".zip", ".txt", ".md", ".pdf", ".jpg", ".jpeg", ".png"}

SYSTEM_PROMPT = """\
You are a senior QA engineer and software architect generating and improving automated test plans and code for a Python project.

## Goals
1) Read and understand all provided context (PDFs, images, flattened repo text, and conversation history).
2) Read and understand the user's current request/requirements.
3) Use the available functions to CREATE or EDIT files:
   - Python pytest test code files under appropriate `tests/` paths.
   - Helper modules under `tests/utils/` if beneficial.
   - Comprehensive Markdown test specifications (e.g., `TEST_SPEC.md` or under `docs/`).
   - Application source files that need modification to support testing.
4) Tests must be runnable with `pytest` out of the box (assume `pytest` only; no plugins unless absolutely needed).
5) Prefer pure-std-lib where possible. If external deps are strictly necessary, note them in the spec file.
6) Ensure meaningful file paths (POSIX-style), consistent naming, and idempotent generation.
7) Include minimal fixtures, parametrization, and positive/negative cases where it helps coverage.

## Function Calling
Use the `create_or_edit_file` function to create new files or edit existing ones. You can call this function multiple times in parallel for independent files.

When you need to:
- **Create a new file**: Call `create_or_edit_file` with the full file path and complete content.
- **Edit an existing file**: Call `create_or_edit_file` again with the file path and COMPLETE new content (full replacement).
- **Create multiple files**: Call `create_or_edit_file` multiple times in a single response for parallel execution.

## Content Requirements
- For `markdown` files, produce full content (headings, scenarios, cases, traceability).
- For Python test files:
  - Use pytest (`test_*.py`) naming.
  - Use clear test functions, possibly fixtures.
  - Add docstrings explaining intent.
  - Where app APIs are unclear, stub interfaces using reasonable assumptions, with TODOs clearly marked.
  - Avoid environment-dependent operations unless explicitly described in context.
- For edited files:
  - Provide the COMPLETE file content, not partial updates.
  - Maintain existing imports and structure unless changes are specifically requested.
  - Preserve functionality not mentioned in the improvement request.

### TEST_SPEC.md Template Guidance
When generating or editing `TEST_SPEC.md` (or similar Markdown specs), use the structure below. If specific details (for example, execution evidence or assigned tester) are unknown, populate the field with `TBD` rather than fabricating data.

# **Test Case Specification Template**

**Module Name:**
**Test Case ID:**

---

### **Test Case Description**

> *Describe the purpose of this test case.*

---

### **Prerequisites**

1.
2.

---

### **Environmental Information**

| Item                   | Details          |
| ---------------------- | ---------------- |
| **Operating System**   | TBD              |
| **System Type**        | Laptop / Desktop |
| **Browser / Platform** | TBD              |

---

### **Test Scenario**

> *Describe what is being tested and why.*

---

## **Test Case Details**

| **Test Case ID** | **Test Steps**   | **Test Input** | **Expected Results** | **Verification Notes** | **Comments** |
| ---------------- | ---------------- | -------------- | -------------------- | ---------------------- | ------------ |
| 1                | 1. <br>2. <br>3. | TBD            | TBD                  | Not executed yet       | TBD          |

---

### **Notes**

* Mark unknown or future-state information as `TBD`.
* Attach screenshots, logs, or evidence externally when available.
* Keep *Expected Results* clear and measurable.

---

## **Example**

**Module Name:** Login
**Test Case ID:** gfg_01

---

### **Test Case Description**

To verify login functionality of the GeeksforGeeks website.

---

### **Prerequisites**

1. Stable internet connection
2. Browser (Chrome, Firefox, Internet Explorer, etc.)

---

### **Environmental Information**

| Item                   | Details                              |
| ---------------------- | ------------------------------------ |
| **Operating System**   | Windows / Linux / Mac                |
| **System Type**        | Laptop / Desktop                     |
| **Browser / Platform** | Chrome / Firefox / Internet Explorer |

---

### **Test Scenario**

Checking that after entering the correct username and password, the user can successfully log in.

---

## **Test Case Details**

| **Test Case ID** | **Test Steps**                                              | **Test Input**                                   | **Expected Results**                        | **Verification Notes** | **Comments**    |
| ---------------- | ----------------------------------------------------------- | ------------------------------------------------ | ------------------------------------------- | ---------------------- | --------------- |
| 1                | 1. Enter Username<br>2. Enter Password<br>3. Click on Login | Username: `geeksforgeeks`<br>Password: `geek123` | "Welcome to GeeksforGeeks!" message appears | Execution pending      | No issues found |

## Process
1) Review conversation history to understand what has been generated previously.
2) Summarize the repo: key modules, public APIs (if detectable), and how they relate to the current request.
3) For initial requests: Generate comprehensive test scenarios and cases.
4) For improvement requests: Identify specific files to modify and generate updated complete content.
5) Use function calls to create/edit files.
6) Provide a summary of what was done in your final response.

## Constraints
- Call the `create_or_edit_file` function to persist files.
- Always provide the complete file content (not diffs or patches).
- Use parallel function calls when creating multiple independent files.
- Reference previous iterations when making improvements.
"""

# ---------- Logger Setup ----------

def setup_logger() -> logging.Logger:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logger = logging.getLogger("test_generator")
    logger.setLevel(logging.DEBUG)

    log_file = log_dir / f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

logger = setup_logger()

DEFAULT_TEMPERATURE = 0.4
DEFAULT_MODEL_NAME = "gemini-flash-latest"


def get_model_temperature() -> float:
    raw_value = os.getenv("GEMINI_TEMPERATURE")
    if raw_value is None:
        return DEFAULT_TEMPERATURE
    try:
        return float(raw_value)
    except ValueError:
        logger.warning(
            "Invalid GEMINI_TEMPERATURE=%s; falling back to %.2f",
            raw_value,
            DEFAULT_TEMPERATURE,
        )
        return DEFAULT_TEMPERATURE


def get_model_name() -> str:
    return os.getenv("GEMINI_MODEL", DEFAULT_MODEL_NAME)

# ---------- Gemini Client ----------

try:
    from google import genai
    from google.genai import types
except ImportError:
    raise SystemExit(
        "google-genai not installed. Run: pip install google-genai\n"
        "Docs: https://pypi.org/project/google-genai/"
    )

def create_gemini_client() -> genai.Client:
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY environment variable is not set.")
    return genai.Client(api_key=api_key)

# ---------- Tool: create_or_edit_file ----------

def get_file_creation_tool_definition() -> Dict[str, Any]:
    return {
        "name": "create_or_edit_file",
        "description": "Create a new file or edit an existing file by replacing its entire content.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "POSIX path to create/edit (under temp work dir)"},
                "content": {"type": "string", "description": "Complete file content"},
                "description": {"type": "string", "description": "Short purpose/notes"}
            },
            "required": ["file_path", "content", "description"]
        }
    }

def execute_file_creation(file_path: str, content: str, description: str, work_dir: Path) -> Dict[str, Any]:
    try:
        normalized_path = file_path.replace("\\", "/").lstrip("/")
        target_path = work_dir / normalized_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(content, encoding='utf-8')
        logger.info(f"File created/edited: {target_path}")
        return {
            "status": "success",
            "file_path": str(target_path),
            "message": f"File '{file_path}' created/edited ({len(content)} bytes)",
            "description": description
        }
    except Exception as e:
        logger.error(f"Error creating/editing file {file_path}: {e}", exc_info=True)
        return {"status": "error", "file_path": file_path, "message": str(e)}

# ---------- Repo-to-Text (single, de-duplicated implementation) ----------

def check_tree_command() -> bool:
    return shutil.which('tree') is not None

def is_ignored_path(file_path: str) -> bool:
    return ('.git' in file_path) or os.path.basename(file_path).startswith('repo-to-text_')

def run_tree_command(path: str) -> str:
    if platform.system() == "Windows":
        cmd = ["cmd", "/c", "tree", "/a", "/f", path]
    else:
        cmd = ["tree", "-a", "-f", "--noreport", path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', check=True)
    return result.stdout

def extract_full_path(line: str, path: str) -> Optional[str]:
    idx = line.find('./')
    if idx == -1:
        idx = line.find(path)
    return line[idx:].strip() if idx != -1 else None

def _mark_non_empty_dirs(relative_path: str, non_empty_dirs: Set[str]) -> None:
    dir_path = os.path.dirname(relative_path)
    while dir_path:
        non_empty_dirs.add(dir_path)
        dir_path = os.path.dirname(dir_path)

def _should_ignore_file(
    file_path: str,
    relative_path: str,
    gitignore_spec: Optional[PathSpec],
    content_ignore_spec: Optional[PathSpec],
    tree_and_content_ignore_spec: Optional[PathSpec]
) -> bool:
    relative_path = relative_path.replace(os.sep, '/')
    if relative_path.startswith('./'):
        relative_path = relative_path[2:]
    if os.path.isdir(file_path):
        relative_path += '/'

    return (
        is_ignored_path(file_path) or
        bool(gitignore_spec and gitignore_spec.match_file(relative_path)) or
        bool(content_ignore_spec and content_ignore_spec.match_file(relative_path)) or
        bool(tree_and_content_ignore_spec and tree_and_content_ignore_spec.match_file(relative_path))
    )

def _process_line(
    line: str,
    path: str,
    gitignore_spec: Optional[PathSpec],
    tree_and_content_ignore_spec: Optional[PathSpec],
    non_empty_dirs: Set[str]
) -> Optional[str]:
    full_path = extract_full_path(line, path)
    if not full_path or full_path == '.':
        return None

    try:
        relative_path = os.path.relpath(full_path, path).replace(os.sep, '/')
    except (ValueError, OSError):
        relative_path = os.path.basename(full_path)

    if _should_ignore_file(full_path, relative_path, gitignore_spec, None, tree_and_content_ignore_spec):
        return None

    if not os.path.isdir(full_path):
        _mark_non_empty_dirs(relative_path, non_empty_dirs)

    if not os.path.isdir(full_path) or os.path.dirname(relative_path) in non_empty_dirs:
        return line.replace('./', '', 1)
    return None

def _filter_tree_output(
    tree_output: str,
    path: str,
    gitignore_spec: Optional[PathSpec],
    tree_and_content_ignore_spec: Optional[PathSpec]
) -> str:
    lines = tree_output.splitlines()
    non_empty_dirs: Set[str] = set()
    filtered = [_process_line(line, path, gitignore_spec, tree_and_content_ignore_spec, non_empty_dirs) for line in lines]
    return '\n'.join(filter(None, filtered))

def get_tree_structure(
    path: str = '.',
    gitignore_spec: Optional[PathSpec] = None,
    tree_and_content_ignore_spec: Optional[PathSpec] = None
) -> str:
    if not check_tree_command():
        return ""
    try:
        tree_output = run_tree_command(path)
    except Exception:
        return ""
    if not gitignore_spec and not tree_and_content_ignore_spec:
        return tree_output
    return _filter_tree_output(tree_output, path, gitignore_spec, tree_and_content_ignore_spec)

def load_ignore_specs(path: str = '.', cli_ignore_patterns: Optional[List[str]] = None) -> Tuple[Optional[PathSpec], Optional[PathSpec], PathSpec]:
    gitignore_spec = None
    content_ignore_spec = None
    tree_and_content_ignore_list: List[str] = []
    use_gitignore = True

    repo_settings_path = os.path.join(path, '.repo-to-text-settings.yaml')
    if os.path.exists(repo_settings_path):
        try:
            with open(repo_settings_path, 'r', encoding='utf-8') as f:
                settings: Dict[str, Any] = yaml.safe_load(f) or {}
                use_gitignore = settings.get('gitignore-import-and-ignore', True)
                if 'ignore-content' in settings:
                    content_ignore_spec = pathspec.PathSpec.from_lines('gitwildmatch', settings['ignore-content'])
                if 'ignore-tree-and-content' in settings:
                    tree_and_content_ignore_list.extend(settings.get('ignore-tree-and-content', []))
        except Exception:
            pass

    if cli_ignore_patterns:
        tree_and_content_ignore_list.extend(cli_ignore_patterns)

    if use_gitignore:
        gitignore_path = os.path.join(path, '.gitignore')
        if os.path.exists(gitignore_path):
            try:
                with open(gitignore_path, 'r', encoding='utf-8') as f:
                    gitignore_spec = pathspec.PathSpec.from_lines('gitwildmatch', f)
            except Exception:
                pass

    tree_and_content_ignore_spec = pathspec.PathSpec.from_lines('gitwildmatch', tree_and_content_ignore_list)
    return gitignore_spec, content_ignore_spec, tree_and_content_ignore_spec

def _read_file_content(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file_path, 'rb') as f_bin:
            return f_bin.read().decode('latin1')
    except FileNotFoundError:
        if os.path.islink(file_path) and not os.path.exists(file_path):
            try:
                target = os.readlink(file_path)
            except OSError:
                target = ''
            return f"[symlink] -> {target}"
        raise

def _load_additional_specs(path: str = '.') -> Dict[str, Any]:
    kv: Dict[str, Any] = {'maximum_word_count_per_file': None}
    repo_settings_path = os.path.join(path, '.repo-to-text-settings.yaml')
    if os.path.exists(repo_settings_path):
        try:
            with open(repo_settings_path, 'r', encoding='utf-8') as f:
                settings = yaml.safe_load(f) or {}
                max_words = settings.get('maximum_word_count_per_file')
                if isinstance(max_words, int) and max_words > 0:
                    kv['maximum_word_count_per_file'] = max_words
        except Exception:
            pass
    return kv

def _generate_output_content(
    path: str,
    tree_structure: str,
    gitignore_spec: Optional[PathSpec],
    content_ignore_spec: Optional[PathSpec],
    tree_and_content_ignore_spec: Optional[PathSpec],
    maximum_word_count_per_file: Optional[int] = None
) -> List[str]:
    output_segments: List[str] = []
    current: List[str] = []
    current_wc = 0
    project_name = os.path.basename(os.path.abspath(path))

    def count_words(text: str) -> int:
        return len(text.split())

    def finalize():
        nonlocal current_wc
        if current:
            output_segments.append("".join(current))
            current.clear()
            current_wc = 0

    def add(chunk: str):
        nonlocal current_wc
        wc = count_words(chunk)
        if maximum_word_count_per_file and current and current_wc + wc > maximum_word_count_per_file:
            finalize()
        current.append(chunk)
        current_wc += wc

    add('<repo-to-text>\n')
    add(f'Directory: {project_name}\n\n')
    add('Directory Structure:\n')
    add('<directory_structure>\n.\n')
    if os.path.exists(os.path.join(path, '.gitignore')):
        add('├── .gitignore\n')
    add(tree_structure + '\n' + '</directory_structure>\n')

    for root, _, files in os.walk(path):
        for filename in files:
            file_path = os.path.join(root, filename)
            relative_path = os.path.relpath(file_path, path)
            if _should_ignore_file(file_path, relative_path, gitignore_spec, content_ignore_spec, tree_and_content_ignore_spec):
                continue
            cleaned_relative_path = relative_path.replace('./', '', 1)
            add(f'\n<content full_path="{cleaned_relative_path}">\n')
            add(_read_file_content(file_path))
            add('\n</content>\n')

    add('\n</repo-to-text>\n')
    finalize()
    return output_segments or ["<repo-to-text>\n</repo-to-text>\n"]

def save_repo_to_text(
        path: str = '.',
        output_dir: Optional[str] = None,
        to_stdout: bool = False,
        cli_ignore_patterns: Optional[List[str]] = None
) -> str:
    gitignore_spec, content_ignore_spec, tree_and_content_ignore_spec = load_ignore_specs(path, cli_ignore_patterns)
    additional_specs = _load_additional_specs(path)
    tree_structure = get_tree_structure(path, gitignore_spec, tree_and_content_ignore_spec)
    segments = _generate_output_content(
        path,
        tree_structure,
        gitignore_spec,
        content_ignore_spec,
        tree_and_content_ignore_spec,
        additional_specs.get('maximum_word_count_per_file')
    )

    if to_stdout:
        for s in segments:
            print(s, end='')
        return "".join(segments)

    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d-%H-%M-%S-UTC')
    base_stem = f'repo-to-text_{timestamp}'
    output_paths: List[str] = []

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if len(segments) == 1:
        filename = f"{base_stem}.txt"
        full = os.path.join(output_dir, filename) if output_dir else filename
        Path(full).write_text(segments[0], encoding='utf-8')
        print(f'[SUCCESS] Repository structure and contents saved to "{os.path.basename(full)}"')
        output_paths.append(full)
    else:
        for i, segment in enumerate(segments, 1):
            filename = f"{base_stem}_part_{i}.txt"
            full = os.path.join(output_dir, filename) if output_dir else filename
            Path(full).write_text(segment, encoding='utf-8')
            output_paths.append(full)
        print(f"[SUCCESS] Saved to {len(output_paths)} files:")
        for fp in output_paths:
            print(f'  - "{os.path.basename(fp)}"')

    return output_paths[0] if output_paths else ""

# ---------- Helpers: file ingest & Google Files ----------

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def copy_file_to_work_dir(src: Path, work_dir: Path) -> Path:
    dst = work_dir / (src.stem + ".txt" if src.suffix.lower() == ".md" else src.name)
    shutil.copy2(src, dst)
    return dst

def categorize_upload(file_path: Path) -> Optional[str]:
    ext = file_path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        logger.warning(f"Unsupported file extension: {file_path.name} ({ext})")
        return None
    return "zip" if ext == ".zip" else "context"

def get_mime_type(file_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if mime_type:
        return mime_type
    return {
        '.md': 'text/markdown',
        '.txt': 'text/plain',
        '.pdf': 'application/pdf',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
    }.get(file_path.suffix.lower(), 'application/octet-stream')

def upload_file_to_gemini(client: genai.Client, file_path: Path) -> Dict[str, str]:
    return upload_path_to_gemini(client, file_path, None)

def upload_path_to_gemini(client: genai.Client, path: Path, mime_hint: Optional[str] = None) -> Optional[Dict[str, str]]:
    mime_type = mime_hint or get_mime_type(path)
    try:
        config = types.UploadFileConfig(mime_type=mime_type)
    except (ImportError, AttributeError):
        config = None

    uploaded_file = None
    last_error: Optional[Exception] = None

    attempt_id = 0

    def _log_failure(exc: Exception, label: str) -> None:
        logger.warning(f"Upload attempt {label} failed for {path}: {exc}")

    attempts: List[Tuple[str, Any]] = []

    if config:
        attempts.append(("file+config", lambda: client.files.upload(file=str(path), config=config)))

    attempts.append(("file+mime", lambda: client.files.upload(file=str(path), mime_type=mime_type)))

    def attempt_file_handle():
        with open(path, 'rb') as fh:
            return client.files.upload(file=fh, mime_type=mime_type)

    attempts.append(("fh+mime", attempt_file_handle))

    if hasattr(client.files, "upload_from_path"):
        attempts.append(("upload_from_path", lambda: client.files.upload_from_path(path=str(path), mime_type=mime_type)))

    for label, func in attempts:
        attempt_id += 1
        try:
            uploaded_file = func()
            if uploaded_file:
                break
        except KeyError as exc:
            last_error = exc
            _log_failure(exc, label)
        except Exception as exc:  # pylint: disable=broad-except
            last_error = exc
            _log_failure(exc, label)

    if not uploaded_file:
        logger.error(f"Giving up uploading {path}: {last_error}")
        return None

    logger.info(f"Uploaded to Google Files: {path.name}")
    file_uri = getattr(uploaded_file, "uri", None)
    if not file_uri:
        logger.warning(f"Uploaded file object missing URI for {path}; response: {uploaded_file}")
        return None
    return {"file_uri": file_uri, "mime_type": mime_type, "file_path": str(path)}


def prepare_file_for_upload(source: Path) -> Tuple[Path, Optional[Path]]:
    """Return path to upload and optional temp artifact to clean up."""
    try:
        if source.suffix.lower() == ".md":
            with tempfile.NamedTemporaryFile(
                mode='w',
                encoding='utf-8',
                suffix='.txt',
                prefix=source.stem + '_',
                dir=str(source.parent),
                delete=False
            ) as tmp_fh:
                tmp_fh.write(source.read_text(encoding='utf-8'))
                temp_path = Path(tmp_fh.name)
            logger.info(f"Converted markdown to text for upload: {source.name} -> {temp_path.name}")
            return temp_path, temp_path
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(f"Failed converting markdown {source} to text: {exc}; uploading original file.")
    return source, None

def extract_and_flatten_repo(zip_path: Path) -> Tuple[str, str]:
    temp_dir = Path(tempfile.mkdtemp(prefix="codebase_"))
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(temp_dir)
    context_dir = ensure_dir(Path("context"))
    output_file_path = save_repo_to_text(path=str(temp_dir), output_dir=str(context_dir), to_stdout=False)
    output_file = Path(output_file_path)
    content = output_file.read_text(encoding="utf-8", errors="replace")
    if len(content) > MAX_TOTAL_TEXT:
        content = content[:MAX_TOTAL_TEXT]
        logger.warning(f"Truncated {zip_path.stem} from {len(content)} to {MAX_TOTAL_TEXT} chars")
    return output_file.as_posix(), content

def save_json_file(data: Dict[str, Any], directory: str, prefix: str) -> Path:
    dir_path = ensure_dir(Path(directory))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_path = dir_path / f"{prefix}_{timestamp}.json"
    file_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    logger.info(f"Saved {prefix} to: {file_path.name} ({file_path.stat().st_size} bytes)")
    return file_path

def save_llm_context(user_story: str, repo_files: List[Tuple[str, str]], uploaded_files: List[Path]) -> Path:
    context_data = {
        "timestamp": datetime.now().isoformat(),
        "user_story": user_story,
        "code_files": [{"path": path, "size_chars": len(content)} for path, content in repo_files],
        "uploaded_files": [{"name": f.name, "size_bytes": f.stat().st_size} for f in uploaded_files],
        "file_content_sample": {path: content[:500] for path, content in repo_files[:5]}
    }
    return save_json_file(context_data, "context_logs", "llm_context")

def write_outputs_to_zip_from_workdir(result: Dict[str, Any], work_dir: Path, original_codebase_dir: Optional[Path] = None) -> Path:
    output_dir = ensure_dir(Path("results"))
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    zip_filename = output_dir / f"generated_files_{timestamp}.zip"

    original_dir = Path(original_codebase_dir) if original_codebase_dir and Path(original_codebase_dir).exists() else None
    original_root_dir = None
    root_folder_name = "generated_output"

    if original_dir:
        try:
            candidates = [p for p in original_dir.iterdir() if not p.name.startswith('__MACOSX')]
        except FileNotFoundError:
            candidates = []

        if len(candidates) == 1 and candidates[0].is_dir():
            original_root_dir = candidates[0]
            root_folder_name = candidates[0].name
        else:
            original_root_dir = original_dir
            root_folder_name = original_dir.name

    created_files: Dict[Path, Path] = {}
    for entry in result.get("created_files", []):
        file_path = Path(entry.get("file_path", ""))
        if not file_path.exists():
            continue
        try:
            rel_path = file_path.relative_to(work_dir)
        except ValueError:
            logger.debug(f"Skipping generated file outside work dir: {file_path}")
            continue
        created_files[rel_path] = file_path

    try:
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zf:
            if original_root_dir:
                for path in original_root_dir.rglob('*'):
                    if not path.is_file():
                        continue
                    rel = path.relative_to(original_root_dir)
                    # Skip original file if a generated file overrides the same relative path
                    if rel in created_files:
                        continue
                    arcname = Path(root_folder_name) / rel
                    zf.write(path, arcname)

            for rel_path, abs_path in created_files.items():
                arcname = Path(root_folder_name) / rel_path
                zf.write(abs_path, arcname)

        logger.info(f"ZIP file created: {zip_filename}")
        return zip_filename
    except Exception as e:
        logger.error(f"Failed to create ZIP: {e}")
        raise

def collect_and_upload_created_files(client: genai.Client, work_dir: Path, since_ts: float) -> List[Dict[str, str]]:
    """Upload files created/edited since `since_ts` under work_dir to Google Files."""
    uploaded: List[Dict[str, str]] = []
    for root, _, files in os.walk(work_dir):
        for fn in files:
            fp = Path(root) / fn
            try:
                if fp.stat().st_mtime >= since_ts:
                    upload_target, cleanup_path = prepare_file_for_upload(fp)
                    upload_info = upload_path_to_gemini(client, upload_target)
                    if upload_info:
                        uploaded.append(upload_info)
                    if cleanup_path and cleanup_path != fp:
                        try:
                            cleanup_path.unlink(missing_ok=True)
                        except Exception as exc:  # pylint: disable=broad-except
                            logger.warning(f"Failed to clean temporary upload artifact {cleanup_path}: {exc}")
            except FileNotFoundError:
                continue
            except Exception as exc:  # pylint: disable=broad-except
                logger.error(f"Failed to upload generated file {fp}: {exc}")
    return uploaded

# ---------- Pipeline ----------

def process_uploads(uploads, work_dir: Path) -> Tuple[List[Path], List[Path]]:
    context_files, zip_files = [], []
    if not uploads:
        return context_files, zip_files
    for upload in uploads:
        src_path = Path(upload.name if hasattr(upload, "name") else upload)
        category = categorize_upload(src_path)
        if not category:
            continue
        dst_path = copy_file_to_work_dir(src_path, work_dir)
        (zip_files if category == "zip" else context_files).append(dst_path)
    return context_files, zip_files

def process_repositories(zip_files: List[Path]) -> List[Tuple[str, str]]:
    repo_contents = []
    for zip_file in zip_files:
        try:
            logger.info(f"Processing ZIP: {zip_file.name}")
            file_path, content = extract_and_flatten_repo(zip_file)
            repo_contents.append((file_path, content))
        except Exception as e:
            logger.error(f"Failed to process {zip_file.name}: {e}")
    return repo_contents

def build_llm_request(
    user_story: str,
    repo_contents: List[Tuple[str, str]],
    uploaded_refs: List[Any],
    conversation_history: List[Dict[str, str]] = None
) -> List[Any]:
    
    user_parts: List[types.Part] = []

    if conversation_history:
        buf = io.StringIO()
        buf.write("### CONVERSATION HISTORY\n")
        for i, turn in enumerate(conversation_history, 1):
            buf.write(f"\n--- Turn {i} ---\nUser: {turn['user']}\n")
            if 'assistant_summary' in turn:
                buf.write(f"Assistant: {turn['assistant_summary']}\n")
        user_parts.append(types.Part.from_text(text=buf.getvalue()))

    user_parts.append(
        types.Part.from_text(text="### CURRENT USER REQUEST\n" + user_story.strip())
    )

    for ref in uploaded_refs or []:
        if not isinstance(ref, dict):
            continue
        file_uri = ref.get('file_uri')
        if not file_uri:
            continue
        mime_type = ref.get('mime_type') or 'application/octet-stream'
        user_parts.append(types.Part.from_uri(file_uri=file_uri, mime_type=mime_type))

    for file_path, content in repo_contents:
        chunk = content[:MAX_TOTAL_TEXT]
        user_parts.append(
            types.Part.from_text(
                text=f"### FLATTENED CODEBASE: {file_path}\n{chunk}"
            )
        )

    return [types.Content(role='user', parts=user_parts)]

def call_gemini_model(client: genai.Client, contents: List[Any], work_dir: Path) -> Dict[str, Any]:
    file_tool = get_file_creation_tool_definition()

    def _extract_response_parts(resp):
        if not resp or not getattr(resp, "candidates", None):
            return None, None, []
        candidate = resp.candidates[0]
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        return candidate, content, parts

    config = types.GenerateContentConfig(
        temperature=get_model_temperature(),
        thinking_config=types.ThinkingConfig(
            thinking_budget=-1,
        ),
        tools=[types.Tool(function_declarations=[file_tool])],
        system_instruction=SYSTEM_PROMPT
    )

    response = client.models.generate_content(
        model=get_model_name(),
        contents=contents,
        config=config
    )

    created_files: List[Dict[str, Any]] = []
    last_candidate_parts: List[Any] = []
    last_text_candidate = None
    while response:
        candidate, assistant_content, assistant_parts = _extract_response_parts(response)
        last_candidate_parts = assistant_parts
        last_text_candidate = candidate
        if candidate is None:
            logger.warning("Gemini response had no candidates; stopping loop.")
            break

        new_tool_results: List[Dict[str, Any]] = []
        for part in assistant_parts:
            if getattr(part, "function_call", None):
                fc = part.function_call
                logger.info(f"Function call: {fc.name}")
                if fc.name == "create_or_edit_file":
                    result = execute_file_creation(
                        file_path=fc.args.get("file_path", ""),
                        content=fc.args.get("content", ""),
                        description=fc.args.get("description", ""),
                        work_dir=work_dir
                    )
                    created_files.append(result)
                    new_tool_results.append(result)
                else:
                    logger.warning(f"Unsupported function call received: {fc.name}")

        if new_tool_results:
            # Send tool results back
            tool_parts = [
                types.Part.from_function_response(name="create_or_edit_file", response=r)
                for r in new_tool_results
            ]
            if assistant_content is not None:
                contents.append(assistant_content)
            else:
                contents.append(types.Content(role="assistant", parts=[]))
            contents.append(types.Content(role="tool", parts=tool_parts))
            response = client.models.generate_content(
                model=get_model_name(),
                contents=contents,
                config=config
            )
        else:
            break

    final_text = ""
    for part in last_candidate_parts:
        if getattr(part, "text", None):
            final_text = part.text
            break
    if not final_text and last_text_candidate is not None:
        finish_info = getattr(last_text_candidate, "finish_reason", "")
        if finish_info:
            final_text = f"[finish_reason: {finish_info}]"

    return {
        "status": "success",
        "created_files": created_files,
        "summary": final_text,
        "total_files_created": sum(1 for f in created_files if f.get("status") == "success")
    }

def infer(uploads, user_story: str, create_zip: bool, conversation_history: List[Dict[str, str]], session_state: Dict[str, Any]):
    session_state = dict(session_state or {})
    session_id = session_state.get('session_id') or datetime.now().strftime('%Y%m%d_%H%M%S')
    logger.info(f"=== Processing request in session: {session_id} ===")

    progress_lines: List[str] = []

    def emit_progress(line: Optional[str] = None, replace_last: bool = False) -> Dict[str, Any]:
        if line is not None:
            if replace_last and progress_lines:
                progress_lines[-1] = line
            else:
                progress_lines.append(line)
        return {
            "type": "progress",
            "message": "\n".join(progress_lines),
            "conversation_history": conversation_history,
            "session_state": session_state
        }

    if not user_story or not user_story.strip():
        yield {
            "type": "final",
            "message": "⚠️ Please provide a requirement or request.",
            "zip_path": None,
            "conversation_history": conversation_history,
            "session_state": session_state
        }
        return

    try:
        yield emit_progress("🔌 Initializing Gemini client…")
        client = create_gemini_client()
        yield emit_progress("🔌 Gemini client ready ✅", replace_last=True)

        repo_contents = session_state.get('repo_contents', [])
        uploaded_refs = session_state.get('uploaded_refs', [])
        work_dir_path = session_state.get('work_dir')
        work_dir = Path(work_dir_path) if work_dir_path else None
        original_codebase_dir = session_state.get('original_codebase_dir')
        original_codebase_path = Path(original_codebase_dir) if original_codebase_dir else None
        context_files: List[Path] = []
        zip_files: List[Path] = []

        if not session_state or work_dir is None:
            work_dir = Path(tempfile.mkdtemp(prefix="poc_ctx_"))
            yield emit_progress("📥 Ingesting uploads…")
            context_files, zip_files = process_uploads(uploads, work_dir)
            yield emit_progress("📥 Ingesting uploads ✅", replace_last=True)

            yield emit_progress("🧩 Flattening codebase (zip → text)…")
            repo_contents = process_repositories(zip_files)
            yield emit_progress("🧩 Flattening codebase (zip → text) ✅", replace_last=True)

            original_codebase_path = None
            if zip_files:
                original_codebase_path = Path(tempfile.mkdtemp(prefix="codebase_"))
                with zipfile.ZipFile(zip_files[0], 'r') as zf:
                    zf.extractall(original_codebase_path)

            yield emit_progress("☁️ Uploading context to Google Files…")
            uploaded_refs = [upload_file_to_gemini(client, f) for f in context_files] if context_files else []
            for file_path, _ in repo_contents:
                flattened_ref = upload_path_to_gemini(client, Path(file_path), "text/plain")
                uploaded_refs.append(flattened_ref)
            yield emit_progress("☁️ Uploading context to Google Files ✅", replace_last=True)

            session_state = {
                'session_id': session_id,
                'work_dir': str(work_dir),
                'repo_contents': repo_contents,
                'uploaded_refs': uploaded_refs,
                'context_files': [str(f) for f in context_files],
                'original_codebase_dir': str(original_codebase_path) if original_codebase_path else None
            }
            save_llm_context(user_story, repo_contents, context_files)
        else:
            work_dir = Path(work_dir)
            original_codebase_path = Path(original_codebase_path) if original_codebase_path else None

        yield emit_progress("🤖 Calling LLM (with create_file tool)…")
        turn_start = time.time()

        contents = build_llm_request(user_story, repo_contents, uploaded_refs, conversation_history)
        result = call_gemini_model(client, contents, work_dir)
        yield emit_progress("🤖 LLM response received ✅", replace_last=True)

        yield emit_progress("☁️ Uploading newly created files to Google Files…")
        new_refs = collect_and_upload_created_files(client, work_dir, since_ts=turn_start)
        if new_refs:
            session_state.setdefault("uploaded_refs", []).extend(new_refs)
        yield emit_progress("☁️ Uploading newly created files to Google Files ✅", replace_last=True)

        save_json_file(result, "results", "function_call_result")

        zip_path: Optional[Path] = None
        if create_zip:
            yield emit_progress("📦 Bundling ZIP (original + generated files)…")
            original_dir = session_state.get('original_codebase_dir')
            original_dir_path = Path(original_dir) if original_dir else None
            zip_path = write_outputs_to_zip_from_workdir(result, work_dir, original_dir_path)
            yield emit_progress("📦 ZIP bundle ready ✅", replace_last=True)

        summary = result.get('summary', '')
        success_count = result.get('total_files_created', 0)
        progress_lines.append(f"🗂️ {success_count} file(s) created/edited")
        progress_lines.append("✅ Completed. See summary below." if summary else "✅ Completed.")

        updated_history = conversation_history + [{
            'user': user_story,
            'assistant_summary': (summary[:500] if summary else f"Created {success_count} files")
        }]

        assistant_msg = "\n".join(progress_lines)
        if summary:
            assistant_msg += "\n\n" + summary

        yield {
            "type": "final",
            "message": assistant_msg,
            "zip_path": str(zip_path) if zip_path else None,
            "conversation_history": updated_history,
            "session_state": session_state
        }

    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        progress_lines.append(f"❌ Error: {str(e)}")
        yield {
            "type": "final",
            "message": "\n".join(progress_lines) if progress_lines else f"❌ Error: {str(e)}",
            "zip_path": None,
            "conversation_history": conversation_history,
            "session_state": session_state
        }

# ---------- Gradio UI (progress inside chat box) ----------

with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown(
        "# 🧪 User Story Test Generator\n"
        "Upload PDFs/Images/ZIP (codebase) → Generate test assets → Iterate to refine results\n\n"
    )

    conversation_state = gr.State([])
    session_state = gr.State({})

    with gr.Column():
        with gr.Group():
            gr.Markdown("### Upload Context")
            uploads = gr.File(
                label="📤 Upload Files (First Message Only)",
                file_count="multiple",
                file_types=list(ALLOWED_EXTENSIONS)
            )
            gr.Markdown(
                "**Tip:** Upload your codebase ZIP and any PDFs/images once. Then continue the conversation to refine results."
            )
            create_zip = gr.Checkbox(
                label="📦 Create ZIP with original codebase + generated files",
                value=True
            )

        with gr.Group():
            gr.Markdown("### Conversation")
            chatbot = gr.Chatbot(label="💬 Conversation", height=480, type="messages")
            user_input = gr.Textbox(
                label="Your Request (User Story)",
                lines=3,
                placeholder="Describe what to test… e.g. 'Generate tests for login and error handling.'"
            )
            with gr.Row():
                send_btn = gr.Button("Send", variant="primary", scale=3)
                clear_btn = gr.Button("Clear Chat", scale=1)

        with gr.Group():
            gr.Markdown("### Downloads")
            out_zip = gr.File(label="📥 Download Latest ZIP", interactive=False)

    def send_message(uploads, user_msg, create_zip, conv_history, conv_state, sess_state):
        conv_history = list(conv_history or [])
        conv_state = list(conv_state or [])
        sess_state = dict(sess_state or {})

        if not user_msg or not user_msg.strip():
            conv_history.append({"role": "assistant", "content": "⚠️ Please enter a message."})
            yield conv_history, "", None, conv_state, sess_state
            return

        conv_history.append({"role": "user", "content": user_msg})
        conv_history.append({"role": "assistant", "content": "⏳ Preparing request…"})

        yield conv_history, "", None, conv_state, sess_state

        try:
            inference = infer(uploads, user_msg, create_zip, conv_state, sess_state)
            current_zip = None
            updated_history = conv_state
            updated_session_state = sess_state

            for update in inference:
                update_type = update.get("type")
                if update_type == "progress":
                    conv_history[-1]["content"] = update.get("message", "")
                    updated_session_state = update.get("session_state", updated_session_state)
                    yield conv_history, "", current_zip, updated_history, updated_session_state
                elif update_type == "final":
                    conv_history[-1]["content"] = update.get("message", "")
                    current_zip = update.get("zip_path")
                    updated_history = update.get("conversation_history", updated_history)
                    updated_session_state = update.get("session_state", updated_session_state)
                    yield conv_history, "", current_zip, updated_history, updated_session_state
                    return

            conv_history[-1]["content"] = "❌ Inference interrupted unexpectedly."
            yield conv_history, "", current_zip, updated_history, updated_session_state

        except Exception as exc:
            logger.error(f"send_message failed: {exc}", exc_info=True)
            conv_history[-1]["content"] = f"❌ Error: {exc}"
            yield conv_history, "", None, conv_state, sess_state

    def clear_chat():
        return [], "", None, [], {}

    send_btn.click(
        send_message,
        inputs=[uploads, user_input, create_zip, chatbot, conversation_state, session_state],
        outputs=[chatbot, user_input, out_zip, conversation_state, session_state]
    )

    user_input.submit(
        send_message,
        inputs=[uploads, user_input, create_zip, chatbot, conversation_state, session_state],
        outputs=[chatbot, user_input, out_zip, conversation_state, session_state]
    )

    clear_btn.click(
        clear_chat,
        outputs=[chatbot, user_input, out_zip, conversation_state, session_state]
    )

if __name__ == "__main__":
    demo.launch()
