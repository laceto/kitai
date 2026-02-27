"""
kitai/batch.py
Low-level helpers for the OpenAI Batch API.

Architecture — two independent layers:

  1. Generic Batch API primitives
     Works for any endpoint (/v1/embeddings, /v1/chat/completions, etc.).
     Callers are responsible for building task dicts and interpreting results.

       submit_batch_job()        — serialise tasks → upload → create job → job_id
       check_batch_job()         — retrieve status dict for one job
       download_batch_results()  — download + parse output JSONL for one job
       BatchJobNotCompleteError  — raised when download is attempted too early

  2. Embedding workflow helpers
     Wraps the generic primitives for the common case of embedding Documents
     via the /v1/embeddings endpoint (derived from icd_batch_embedding.ipynb).

       build_embedding_tasks()   — Documents → Batch API task dicts
       poll_until_complete()     — block/poll until all jobs reach a terminal state
       parse_embedding_results() — raw result dicts → (custom_id, embedding) pairs

Invariants:
  - No module-level side effects: no client initialisation, no load_dotenv().
  - All network failures propagate to callers; per-item errors are logged, not raised.
  - Every doc passed to build_embedding_tasks must carry doc.metadata["id"].

Debugging:
  - Set logging level to DEBUG to see individual upload/job IDs and poll ticks.
  - check_batch_job() logs counts at INFO on every call — safe to call frequently.
  - parse_embedding_results() logs a success/skip summary at INFO.
"""

import json
import logging
import os
import tempfile
import time
from typing import List, Tuple

from langchain_core.documents import Document
from openai import OpenAI

logger = logging.getLogger(__name__)

# ── Generic Batch API primitives ──────────────────────────────────────────────

# Job statuses that will never change again.
_TERMINAL_STATES: frozenset[str] = frozenset(
    {"completed", "failed", "expired", "cancelled"}
)


class BatchJobNotCompleteError(Exception):
    """Raised when download_batch_results is called on an in-flight batch job.

    Attributes:
        batch_id (str): The job ID that was polled.
        status (str): The current (non-complete) status string.

    Invariant: callers can inspect .status to decide whether to retry
    (``"in_progress"`` / ``"finalizing"``) or abort (``"failed"`` /
    ``"expired"`` / ``"cancelled"``).
    """

    def __init__(self, batch_id: str, status: str) -> None:
        self.batch_id = batch_id
        self.status = status
        super().__init__(
            f"Batch job '{batch_id}' is not complete (status: '{status}'). "
            "Poll with check_batch_job() before calling download_batch_results()."
        )


def submit_batch_job(
    client: OpenAI,
    tasks: list[dict],
    endpoint: str = "/v1/embeddings",
    completion_window: str = "24h",
    metadata: dict | None = None,
) -> str:
    """Upload a list of task dicts to the OpenAI Batch API and create a job.

    Serialises ``tasks`` to a temporary JSONL file, streams it to the Files
    API, then creates the batch job.  The temp file is deleted immediately
    after upload regardless of success or failure.

    Each task must conform to the OpenAI Batch API request schema::

        {
            "custom_id": "<unique string per task>",
            "method":    "POST",
            "url":       "/v1/embeddings",   # match the endpoint arg
            "body":      { ... }             # endpoint-specific payload
        }

    Args:
        client (OpenAI): Initialised OpenAI client.
        tasks (list[dict]): Non-empty list of batch task objects.
        endpoint (str): OpenAI API endpoint for all tasks in this job.
        completion_window (str): Max duration OpenAI will process the batch.
            Only ``"24h"`` is accepted by the API at this time.
        metadata (dict | None): Arbitrary key-value pairs attached to the job
            (visible in the OpenAI dashboard).  Pass
            ``{"description": "my_job"}`` for easy identification.

    Returns:
        str: The new batch job ID (e.g. ``"batch_abc123"``).

    Raises:
        ValueError: If ``tasks`` is empty.
        openai.APIError: Auth errors, quota errors, or schema rejections
            propagate directly to the caller.
    """
    if not tasks:
        raise ValueError("tasks must be a non-empty list.")

    logger.info(
        "Submitting batch job: %d task(s), endpoint=%s", len(tasks), endpoint
    )

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    try:
        for task in tasks:
            tmp.write(json.dumps(task) + "\n")
        tmp.flush()
        tmp.close()

        with open(tmp.name, "rb") as f:
            uploaded = client.files.create(file=f, purpose="batch")
        logger.debug("Uploaded batch file: %s", uploaded.id)
    finally:
        os.unlink(tmp.name)

    job = client.batches.create(
        input_file_id=uploaded.id,
        endpoint=endpoint,
        completion_window=completion_window,
        metadata=metadata or {},
    )
    logger.info("Batch job created: %s (status: %s)", job.id, job.status)
    return job.id


def check_batch_job(client: OpenAI, batch_id: str) -> dict:
    """Retrieve the current status of a batch job.

    Logs status and request counts at INFO on every call — safe to poll
    frequently.

    Args:
        client (OpenAI): Initialised OpenAI client.
        batch_id (str): Job ID returned by :func:`submit_batch_job`.

    Returns:
        dict with keys:

        - ``batch_id`` (str)
        - ``status`` (str): e.g. ``"in_progress"``, ``"completed"``, …
        - ``is_terminal`` (bool): ``True`` once the job will no longer change.
        - ``is_complete`` (bool): ``True`` only for ``"completed"`` status.
        - ``counts`` (dict): ``{"total": int, "completed": int, "failed": int}``
        - ``output_file_id`` (str | None): Set when ``is_complete`` is True.
        - ``error_file_id`` (str | None): Set when the job has failed items.

    Raises:
        openai.APIError: Propagated from the client.
    """
    batch = client.batches.retrieve(batch_id)
    counts = batch.request_counts

    status_info = {
        "batch_id": batch_id,
        "status": batch.status,
        "is_terminal": batch.status in _TERMINAL_STATES,
        "is_complete": batch.status == "completed",
        "counts": {
            "total": counts.total,
            "completed": counts.completed,
            "failed": counts.failed,
        },
        "output_file_id": batch.output_file_id,
        "error_file_id": batch.error_file_id,
    }

    logger.info(
        "Batch %s — status: %s | total: %d | completed: %d | failed: %d",
        batch_id,
        batch.status,
        counts.total,
        counts.completed,
        counts.failed,
    )
    return status_info


def download_batch_results(client: OpenAI, batch_id: str) -> list[dict]:
    """Download and parse the output of a completed batch job.

    Each element of the returned list is one raw result object::

        {
            "id":        "...",
            "custom_id": "<id you set in the task>",
            "response": {
                "status_code": 200,
                "body": { ... }     # endpoint-specific
            },
            "error": null           # or an error object for failed items
        }

    Per-item failures do **not** raise — callers must check
    ``item["error"]`` and ``item["response"]["status_code"]`` themselves.

    Args:
        client (OpenAI): Initialised OpenAI client.
        batch_id (str): Job ID returned by :func:`submit_batch_job`.

    Returns:
        list[dict]: One result dict per submitted task.

    Raises:
        BatchJobNotCompleteError: If the job is not yet ``"completed"``.
            Inspect ``.status`` to distinguish in-flight vs terminal failure.
        openai.APIError: Propagated from the client on download failure.
    """
    status = check_batch_job(client, batch_id)

    if not status["is_complete"]:
        raise BatchJobNotCompleteError(batch_id, status["status"])

    output_file_id = status["output_file_id"]
    logger.info("Downloading results from file %s ...", output_file_id)

    raw_text = client.files.content(output_file_id).text
    results = [
        json.loads(line)
        for line in raw_text.splitlines()
        if line.strip()
    ]

    item_errors = sum(1 for r in results if r.get("error"))
    logger.info(
        "Downloaded %d result(s) from batch %s. Item-level errors: %d",
        len(results),
        batch_id,
        item_errors,
    )
    return results


# ── Embedding workflow helpers ────────────────────────────────────────────────
#
# These three functions implement the embedding batch pipeline from
# icd_batch_embedding.ipynb:
#
#   docs  →  build_embedding_tasks()
#          →  submit_batch_job()          (generic, above)
#          →  poll_until_complete()
#          →  download_batch_results()    (generic, above)
#          →  parse_embedding_results()
#          →  list[(custom_id, embedding)]
#
# ─────────────────────────────────────────────────────────────────────────────

#: Default embedding model used by :func:`build_embedding_tasks`.
DEFAULT_EMBEDDING_MODEL: str = "text-embedding-3-small"
#: Default embedding dimensions used by :func:`build_embedding_tasks`.
DEFAULT_EMBEDDING_DIMENSIONS: int = 1536


def build_embedding_tasks(
    docs: list[Document],
    model: str = DEFAULT_EMBEDDING_MODEL,
    dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
) -> list[dict]:
    """Build OpenAI Batch API task dicts for embedding a list of Documents.

    The ``custom_id`` for each task is ``"custom_id_{doc.metadata['id']}"``
    which matches the convention used by :func:`parse_embedding_results` and
    the downstream CSV / vectorstore helpers in ``kitai/index.py``.

    Args:
        docs (list[Document]): Non-empty list of LangChain Document objects.
            Every doc must have ``doc.metadata["id"]`` set to a unique value.
        model (str): OpenAI embedding model name.
        dimensions (int): Output embedding dimensionality.

    Returns:
        list[dict]: One Batch API task dict per document, ready to pass to
        :func:`submit_batch_job`.

    Raises:
        ValueError: If ``docs`` is empty.
        KeyError: If any doc is missing ``metadata["id"]``.
    """
    if not docs:
        raise ValueError("docs must be a non-empty list.")

    tasks = [
        {
            "custom_id": f"custom_id_{doc.metadata['id']}",
            "method": "POST",
            "url": "/v1/embeddings",
            "body": {
                "input": doc.page_content,
                "model": model,
                "encoding_format": "float",
                "dimensions": dimensions,
            },
        }
        for doc in docs
    ]
    logger.debug(
        "Built %d embedding tasks (model=%s, dimensions=%d).",
        len(tasks),
        model,
        dimensions,
    )
    return tasks


def poll_until_complete(
    client: OpenAI,
    batch_ids: list[str],
    poll_interval: float = 10.0,
) -> list[str]:
    """Block until all batch jobs reach a terminal state.

    Polls every ``poll_interval`` seconds and logs progress at INFO.  Stops
    as soon as every job is terminal (completed, failed, expired, or
    cancelled).

    Args:
        client (OpenAI): Initialised OpenAI client.
        batch_ids (list[str]): Job IDs to monitor.
        poll_interval (float): Seconds to wait between poll rounds.
            Use a small value (e.g. ``1.0``) in tests; the default ``10.0``
            is appropriate for production use.

    Returns:
        list[str]: The subset of ``batch_ids`` that finished with status
        ``"completed"``.  Jobs that failed, expired, or were cancelled are
        logged at ERROR but not raised — the caller decides how to handle
        partial success.

    Raises:
        ValueError: If ``batch_ids`` is empty.
    """
    if not batch_ids:
        raise ValueError("batch_ids must be a non-empty list.")

    pending = set(batch_ids)
    completed: list[str] = []

    while pending:
        still_pending = set()
        for batch_id in pending:
            status = check_batch_job(client, batch_id)
            if not status["is_terminal"]:
                still_pending.add(batch_id)
            elif status["is_complete"]:
                logger.info("Batch %s completed successfully.", batch_id)
                completed.append(batch_id)
            else:
                logger.error(
                    "Batch %s ended with non-successful status: %s",
                    batch_id,
                    status["status"],
                )
        pending = still_pending
        if pending:
            logger.debug(
                "Waiting for %d job(s) to finish. Next poll in %.1fs.",
                len(pending),
                poll_interval,
            )
            time.sleep(poll_interval)

    logger.info(
        "All batch jobs resolved. Completed: %d / %d.",
        len(completed),
        len(batch_ids),
    )
    return completed


def parse_embedding_results(
    results: list[dict],
) -> List[Tuple[str, List[float]]]:
    """Extract (custom_id, embedding) pairs from raw Batch API result dicts.

    Skips items that have an ``"error"`` field or an unexpected response
    structure and logs each skip at ERROR.  The caller receives only the
    successfully parsed pairs.

    Args:
        results (list[dict]): Output of :func:`download_batch_results` — one
            result dict per submitted task.

    Returns:
        List[Tuple[str, List[float]]]: One ``(custom_id, embedding)`` pair per
        successfully parsed item.  ``custom_id`` preserves the original value
        set in :func:`build_embedding_tasks` (e.g. ``"custom_id_42"``).

    Raises:
        ValueError: If ``results`` is empty.
    """
    if not results:
        raise ValueError("results must be a non-empty list.")

    parsed: List[Tuple[str, List[float]]] = []

    for item in results:
        custom_id = item.get("custom_id", "<unknown>")

        if item.get("error"):
            logger.error(
                "Skipping item custom_id=%s — batch item error: %s",
                custom_id,
                item["error"],
            )
            continue

        try:
            embedding: List[float] = item["response"]["body"]["data"][0]["embedding"]
            parsed.append((custom_id, embedding))
        except (KeyError, IndexError) as exc:
            logger.error(
                "Skipping item custom_id=%s — unexpected response structure: %s",
                custom_id,
                exc,
            )

    logger.info(
        "Parsed %d/%d embeddings successfully. Skipped: %d.",
        len(parsed),
        len(results),
        len(results) - len(parsed),
    )
    return parsed
