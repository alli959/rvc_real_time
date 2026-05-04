"""Client for reporting progress and completion back to Laravel."""

import logging
import time
import requests

logger = logging.getLogger(__name__)

INTERNAL_TOKEN = None  # Set at startup from env


def init(token: str):
    """Initialize the webhook client with the internal token."""
    global INTERNAL_TOKEN
    INTERNAL_TOKEN = token


def _headers():
    return {"X-Internal-Token": INTERNAL_TOKEN, "Content-Type": "application/json"}


def report_progress(progress_url: str, progress: int, message: str, step: str, step_number: int, total_steps: int):
    """Report progress to Laravel. Non-fatal on failure."""
    try:
        requests.post(progress_url, json={
            "progress": progress,
            "message": message,
            "step": step,
            "step_number": step_number,
            "total_steps": total_steps,
        }, headers=_headers(), timeout=5)
    except Exception as e:
        logger.warning(f"Failed to report progress: {e}")


def report_completion(complete_url: str, output_path: str, sample_rate: int = None,
                      duration: float = None, output_paths: list = None):
    """Report successful completion to Laravel with retry."""
    payload = {"status": "completed", "output_path": output_path}
    if sample_rate is not None:
        payload["sample_rate"] = sample_rate
    if duration is not None:
        payload["duration"] = duration
    if output_paths:
        payload["output_paths"] = output_paths

    _post_with_retry(complete_url, payload, retries=3)


def report_failure(complete_url: str, error: str):
    """Report failure to Laravel with retry."""
    _post_with_retry(complete_url, {"status": "failed", "error": error}, retries=3)


def check_cancelled(status_url: str) -> bool:
    """Check if job has been cancelled. Returns False on network error (safe default)."""
    try:
        resp = requests.get(status_url, headers=_headers(), timeout=5)
        if resp.ok:
            return resp.json().get("status") == "cancelled"
    except Exception as e:
        logger.warning(f"Failed to check cancellation status: {e}")
    return False


def _post_with_retry(url: str, payload: dict, retries: int = 3):
    """POST with exponential backoff retry."""
    for attempt in range(retries):
        try:
            resp = requests.post(url, json=payload, headers=_headers(), timeout=10)
            if resp.ok:
                return
            logger.warning(f"Webhook POST to {url} returned {resp.status_code}, attempt {attempt + 1}/{retries}")
        except Exception as e:
            logger.warning(f"Webhook POST to {url} failed: {e}, attempt {attempt + 1}/{retries}")

        if attempt < retries - 1:
            time.sleep(2 ** (attempt + 1))  # 2s, 4s

    logger.error(f"All {retries} webhook retries failed for {url}")
