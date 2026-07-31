"""scheduler_log.py -- persistent, greppable logging for the background
scheduler, so a silent freeze is visible instead of invisible.

WHY THIS EXISTS. The scheduler runs as a daemon=True thread with no
supervisor. If the Flask main thread dies, the thread dies with it and
NOTHING is written anywhere -- which is exactly how a machine hibernation
froze progress for 7h20m while the process still looked alive from outside.
The only way to notice was to compare a log file's mtime against the
process's start time by hand.

Two design rules follow from that:

  1. Every tick logs, including the boring ones. A log that only records
     interesting events cannot distinguish "healthy and idle" from "dead",
     which is the single thing this file has to answer.

  2. Liveness is recorded as a DURABLE HEARTBEAT (a timestamp written to
     disk each tick), not inferred from process state. A process can be
     alive with a dead scheduler thread; the heartbeat cannot.

The log is line-oriented and prefixed so it can be grepped without tooling:

    SCHEDULER  tick start / tick ok / tick failed
    UPDATE     scheduled-Update due-check outcomes
    RETRAIN    continuous-retraining tick outcomes

Rotation is size-based (5 MB x 3) because this is meant to run for weeks
unattended and must not fill a disk.
"""
import logging
import os
import json
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_PATH = os.path.join(LOG_DIR, "scheduler.log")
HEARTBEAT_PATH = os.path.join(LOG_DIR, "scheduler_heartbeat.json")

MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 3

_logger = None


def get_logger():
    """Module-level singleton. Handlers are attached exactly once -- calling
    this from both app.py and job_runner.py must not duplicate every line."""
    global _logger
    if _logger is not None:
        return _logger

    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger("exoplanetai.scheduler")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s UTC  %(levelname)-7s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fmt.converter = lambda *args: datetime.now(timezone.utc).timetuple()

        fh = RotatingFileHandler(LOG_PATH, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        # Also to stdout, so `python3 app.py` in a terminal shows the same
        # thing the file gets -- no need to tail two places while debugging.
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    _logger = logger
    return logger


def write_heartbeat(**fields):
    """Durable proof-of-life, rewritten every tick.

    Deliberately a separate tiny file rather than "read the last line of the
    log": the log rotates, may be mid-write, and needs parsing. This is one
    small atomic-ish JSON write that /health can read cheaply and that
    survives restarts. If it stops advancing, the scheduler is dead --
    regardless of whether the process is still up.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    payload = {"last_tick_at": datetime.now(timezone.utc).isoformat(), **fields}
    tmp = HEARTBEAT_PATH + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, HEARTBEAT_PATH)  # atomic on POSIX: no torn reads
    except Exception:
        # Never let heartbeat bookkeeping take down the scheduler thread.
        get_logger().exception("SCHEDULER  heartbeat write failed")


def read_heartbeat():
    """Returns (payload_dict, seconds_since_last_tick) or (None, None)."""
    try:
        with open(HEARTBEAT_PATH) as f:
            payload = json.load(f)
        last = datetime.fromisoformat(payload["last_tick_at"])
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - last).total_seconds()
        return payload, age
    except Exception:
        return None, None
