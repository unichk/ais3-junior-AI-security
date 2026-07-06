import os
import time
import uuid
import logging
import tempfile
import threading
import itertools
from logging.handlers import RotatingFileHandler
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import flask

from guard import load_data
from train import train

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
FLAG = os.environ.get("FLAG", "AIS3{f4k3_fl4g}")
ACC_THRESHOLD = 0.75                 # poisoning succeeds if valid acc < this
NUM_WORKERS = 8                      # fixed pool doing guard + training
MAX_CONTENT_LENGTH = 64 * 1024 * 1024  # cap total upload size (bytes)

# ---------------------------------------------------------------------------
# On-disk logging of every submission's outcome
# ---------------------------------------------------------------------------
LOG_DIR = os.path.join(BASE, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("submissions")
logger.setLevel(logging.INFO)
_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "submissions.log"),
    maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8",
)
_handler.setFormatter(logging.Formatter("%(asctime)s\t%(message)s"))
logger.addHandler(_handler)
logger.propagate = False

# ---------------------------------------------------------------------------
# Job queue / worker pool
# ---------------------------------------------------------------------------
# A job is: {"seq", "state", "result"}
#   state  : "queued" -> "running" -> "done"
#   result : None until done, then a dict describing the outcome
jobs = OrderedDict()
jobs_lock = threading.Lock()
_seq = itertools.count()

executor = ThreadPoolExecutor(max_workers=NUM_WORKERS)
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "poison_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _run_job(job_id, img_path, lbl_path):
    with jobs_lock:
        jobs[job_id]["state"] = "running"
    started = time.monotonic()
    try:
        valid, data = load_data(img_path, lbl_path)
        if not valid:
            result = {"success": False, "message": str(data)}
        else:
            X, y = data
            acc = float(train(X, y))
            if acc < ACC_THRESHOLD:
                result = {"success": True, "acc": acc, "flag": FLAG}
            else:
                result = {
                    "success": False,
                    "acc": acc,
                    "message": (
                        f"Validation accuracy is {acc * 100:.2f}% "
                        f"(need < {ACC_THRESHOLD * 100:.0f}% to win). "
                        "The model shrugged off your poison. Try harder."
                    ),
                }
    except Exception as e:  # never let a worker die silently
        result = {"success": False, "message": f"Internal error during evaluation: {e}"}
    finally:
        for p in (img_path, lbl_path):
            try:
                os.remove(p)
            except OSError:
                pass

    elapsed = time.monotonic() - started
    _log_result(job_id, result, elapsed)

    with jobs_lock:
        jobs[job_id]["state"] = "done"
        jobs[job_id]["result"] = result


def _log_result(job_id, result, elapsed):
    """Append one tab-separated line per finished job to the on-disk log.

    Columns after the timestamp: job id, outcome, detail (acc or error), seconds.
    """
    if result.get("success"):
        outcome = "WIN"
        detail = f"acc={result['acc']:.4f}"
    elif "acc" in result:
        outcome = "FAIL"
        detail = f"acc={result['acc']:.4f}"
    else:
        outcome = "REJECT"
        # keep the message on one line so each record is a single row
        detail = str(result.get("message", "")).replace("\n", " ").replace("\t", " ")
    logger.info("%s\t%s\t%s\t%.2fs", job_id, outcome, detail, elapsed)


def _queue_status(job_id):
    """Return (state, ahead, running) for a job.

    ahead   = number of not-yet-finished jobs submitted before this one
    running = number of jobs currently being processed by the pool
    """
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            return None, 0, 0
        my_seq = job["seq"]
        ahead = sum(
            1 for j in jobs.values()
            if j["state"] != "done" and j["seq"] < my_seq
        )
        running = sum(1 for j in jobs.values() if j["state"] == "running")
        return job["state"], ahead, running


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = flask.Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


@app.route("/")
def index():
    return flask.render_template(
        "index.html",
        thresh=f"{ACC_THRESHOLD * 100:.0f}",
        workers=NUM_WORKERS,
    )


@app.route("/submit", methods=["POST"])
def submit():
    img = flask.request.files.get("images")
    lbl = flask.request.files.get("labels")
    if img is None or lbl is None or img.filename == "" or lbl.filename == "":
        return flask.jsonify(error="Please provide both images.npy and labels.npy."), 400

    job_id = uuid.uuid4().hex
    img_path = os.path.join(UPLOAD_DIR, job_id + "_images.npy")
    lbl_path = os.path.join(UPLOAD_DIR, job_id + "_labels.npy")
    img.save(img_path)
    lbl.save(lbl_path)

    with jobs_lock:
        jobs[job_id] = {"seq": next(_seq), "state": "queued", "result": None}

    executor.submit(_run_job, job_id, img_path, lbl_path)
    return flask.jsonify(job_id=job_id)


@app.route("/status/<job_id>")
def status(job_id):
    state, ahead, running = _queue_status(job_id)
    if state is None:
        return flask.jsonify(error="Unknown job id."), 404
    resp = {"state": state, "ahead": ahead, "running": running, "workers": NUM_WORKERS}
    if state == "done":
        with jobs_lock:
            resp["result"] = jobs[job_id]["result"]
    return flask.jsonify(resp)


@app.errorhandler(413)
def too_large(e):
    return flask.jsonify(error="Upload too large."), 413


if __name__ == "__main__":
    # threaded=True so the request-handling loop keeps serving /status polls
    # while the 8 worker threads chew through the queue.
    app.run(host="0.0.0.0", port=5000, threaded=True)
