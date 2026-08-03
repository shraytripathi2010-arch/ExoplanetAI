"""
app.py -- Flask app for the exoplanet candidate web site.

Local-only: run with `python3 app.py`, open http://127.0.0.1:5000 in a
browser. Not intended for deployment (binds to 127.0.0.1, uses Flask's
built-in dev server, no auth -- fine for a single user on their own machine).

Reuses ALL pipeline logic from src/06_download_unknown.py,
src/07_search_unknown.py, src/08_characterize_candidates.py unmodified --
this file only adds persistence (db.py/sync.py), background job
orchestration (job_runner.py), and the web UI itself.
"""
import os
import sys

from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))

import db
import sync
import job_runner
import scheduler_log
import exofop_vetting

app = Flask(__name__)

DISCLAIMER = ("This is an unconfirmed candidate ranked by an automated pipeline for human "
              "review. It is NOT a confirmed or discovered exoplanet.")


@app.context_processor
def inject_globals():
    return {"disclaimer": DISCLAIMER, "running_run": db.get_running_run()}


@app.route("/")
def dashboard():
    counts = db.summary_counts()
    recent_runs = db.list_runs(limit=20)
    for r in recent_runs:
        r["batch_counts"] = db.get_run_candidate_counts(r["run_id"])
    scheduler_cfg = db.get_scheduler_config()
    return render_template("dashboard.html", counts=counts, recent_runs=recent_runs,
                            scheduler_cfg=scheduler_cfg)


@app.route("/models")
def model_history():
    """Transparency page for the continuous-retraining pipeline (Item 2,
    Part B): which model version is live, every version ever trained by it,
    and every retrain attempt including ones that did NOT get promoted --
    this project has always preferred showing negative results over hiding
    them, and "the scheduler tried and didn't promote" is exactly that."""
    versions = db.list_model_versions(limit=50)
    attempts = db.list_retrain_attempts(limit=50)
    live_version = db.get_live_model_version()
    architecture_baseline = db.get_architecture_baseline()
    cnn_flag_active = any(a.get("cnn_reevaluation_flag") for a in attempts)
    production_meta = None
    production_meta_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "models", "best_model_metadata.json"
    )
    if os.path.exists(production_meta_path):
        import json
        with open(production_meta_path) as f:
            production_meta = json.load(f)
    return render_template("model_history.html", versions=versions, attempts=attempts,
                            live_version=live_version, production_meta=production_meta,
                            architecture_baseline=architecture_baseline,
                            cnn_flag_active=cnn_flag_active)


@app.route("/scheduler/settings", methods=["POST"])
def scheduler_settings():
    enabled = request.form.get("enabled") == "1"
    interval_days = float(request.form.get("interval_days", 7))
    sample_size = int(request.form.get("sample_size", 300))
    next_run_at = None
    if enabled:
        # (Re)start the countdown from now whenever settings are saved with
        # scheduling turned on -- simplest, least-surprising behavior when
        # toggling the interval.
        from datetime import datetime, timedelta, timezone
        next_run_at = (datetime.now(timezone.utc) + timedelta(days=interval_days)).strftime(
            "%Y-%m-%d %H:%M:%S UTC")
    db.update_scheduler_config(enabled, interval_days, sample_size, next_run_at=next_run_at)
    return redirect(url_for("dashboard"))


@app.route("/runs/<int:run_id>/candidates")
def run_candidates_json(run_id):
    """Powers the dashboard's expandable per-run batch: the candidates
    newly found in that specific Update run (see sync.py -- only NEW
    candidates get linked to a run, so this is genuinely "what this run
    found," not a repeat of the full candidate list every time)."""
    candidates = db.get_candidates_for_run(run_id)
    return jsonify([_candidate_summary_json(c) for c in candidates])


@app.route("/runs/<int:run_id>/delete", methods=["POST"])
def delete_run(run_id):
    ok, error = db.delete_run(run_id)
    if not ok:
        return error, 409
    return redirect(url_for("dashboard"))


@app.route("/candidates")
def candidate_list():
    sort_by = request.args.get("sort", "predicted_probability")
    ascending = request.args.get("dir", "desc") == "asc"
    tier_filter = request.args.get("tier") or None
    combined_only = request.args.get("combined_only") == "1"

    candidates = db.list_candidates(sort_by=sort_by, ascending=ascending,
                                     tier_filter=tier_filter, combined_only=combined_only)
    counts = db.summary_counts()
    recent_runs = db.list_runs(limit=1)
    last_run = recent_runs[0] if recent_runs else None
    return render_template("candidate_list.html", candidates=candidates,
                            sort_by=sort_by, ascending=ascending,
                            tier_filter=tier_filter, combined_only=combined_only,
                            counts=counts, last_run=last_run)


@app.route("/candidates/<int:tic_id>")
def candidate_detail(tic_id):
    candidate = db.get_candidate(tic_id)
    if candidate is None:
        return "Candidate not found", 404
    history = db.get_status_history(tic_id)
    events = db.get_verification_events(tic_id)
    char = candidate["characterization"]

    exofop_view_url = f"https://exofop.ipac.caltech.edu/tess/target.php?id={tic_id}"
    exofop_account_url = "https://exofop.ipac.caltech.edu/tess/index.php"

    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "plots",
                              f"{candidate['host']}_folded.png")
    has_plot = os.path.exists(plot_path)
    multi_sector = db.get_multi_sector_evidence(tic_id)
    centroid = db.get_centroid_evidence(tic_id)
    reverify = db.get_reverify_status(tic_id)
    exofop_refresh = db.get_exofop_refresh(tic_id)
    transit_anim = _transit_animation_params(char)
    evidence_items = _evidence_items(char, centroid, candidate)
    ctoi_summary = _build_ctoi_summary(candidate, char, multi_sector, centroid, exofop_refresh)
    # External TFOP vetting cross-check. Read-only lookup against a cached
    # ExoFOP export -- no network call on page render, so a slow or down
    # ExoFOP cannot make candidate pages hang.
    tfop = exofop_vetting.lookup(tic_id)

    return render_template("candidate_detail.html", c=candidate, char=char, history=history,
                            events=events, exofop_view_url=exofop_view_url,
                            exofop_account_url=exofop_account_url, has_plot=has_plot,
                            multi_sector=multi_sector, centroid=centroid, reverify=reverify,
                            exofop_refresh=exofop_refresh, ctoi_summary=ctoi_summary,
                            transit_anim=transit_anim, evidence_items=evidence_items,
                            tfop=tfop)


def _fmt(value, spec, fallback="?"):
    return format(value, spec) if value is not None else fallback


def _build_ctoi_summary(candidate, char, multi_sector, centroid, exofop_refresh):
    """Builds the copyable CTOI submission summary.

    Lives here rather than inline in the template so the page render and the
    Refresh button produce byte-identical text from one implementation --
    duplicating this formatting in JS would guarantee the two drift apart,
    and the whole point of the refresh is that the copied text matches what
    the app currently knows.

    Every value is read from what the pipeline has already stored. Nothing is
    recomputed and no external service is queried here (the Re-verify and
    ExoFOP buttons do that); this assembles the current state, including
    results from checks that finished after the page was first opened."""
    lines = [
        f"TIC ID: {candidate['tic_id']}",
        f"Host: {candidate['host']}",
        f"RA / Dec: {char.get('ra')}, {char.get('dec')}",
        f"Orbital period: {_fmt(char.get('period_days'), '.6f')} days",
        f"Epoch (T0): {_fmt(char.get('epoch_bjd'), '.6f')} BJD",
        f"Transit depth: {_fmt(char.get('transit_depth_ppm'), '.1f')} ppm",
        f"Transit duration: {_fmt(char.get('transit_duration_hours'), '.3f')} hours",
        f"Derived planet radius: {_fmt(char.get('planet_radius_earth'), '.2f')} R_earth",
        f"Host spectral type (rough): {char.get('stellar_spectral_type_rough') or '?'}",
        f"Classifier probability: {_fmt(candidate.get('predicted_probability'), '.3f')}",
        f"Confidence tier: {candidate.get('confidence_tier') or '?'} "
        f"(NOT a confirmation -- automated screening only)",
        f"Plausibility verdict: {char.get('plausibility_verdict')}",
        f"Gaia blend risk: {char.get('blending_status')}",
        f"Variable-star check: {char.get('vsx_status')}",
    ]

    # These three checks are on-demand, so a candidate may have picked any of
    # them up since the last full pipeline run. They were missing from the
    # copied text entirely, which meant the most recent -- and for follow-up
    # purposes most interesting -- evidence was the evidence least likely to
    # be passed along. Only completed results are included; a failed or
    # never-run check contributes nothing rather than an empty-looking line.
    if multi_sector and multi_sector.get("status") == "completed":
        lines.append(
            f"Multi-sector re-analysis (sectors {multi_sector.get('sectors_used')}): "
            f"period {_fmt(multi_sector.get('period_days'), '.6f')} days, "
            f"{int(multi_sector.get('transit_count') or 0)} transits, "
            f"SDE {_fmt(multi_sector.get('sde'), '.1f')}"
            + ("" if multi_sector.get("plausible") else " [flagged as a likely search artifact]"))
    if centroid and centroid.get("status") == "completed":
        lines.append(
            f"Pixel-level centroid check (sector {centroid.get('sector_used')}): "
            f"{centroid.get('verdict')}")
    if exofop_refresh and exofop_refresh.get("status") == "completed":
        toi = exofop_refresh.get("toi_designation")
        lines.append(
            f"ExoFOP status as of {exofop_refresh.get('computed_at')}: "
            + (f"listed as {toi}" if toi else "no TOI designation"))

    lines.extend([
        f"Reasons this could be real: {char.get('supporting_evidence')}",
        f"Reasons for doubt: {char.get('doubting_evidence')}",
    ])
    # The "not previously flagged" line is only true while the candidate is
    # still unknown. Once a check finds it in the archive/TOI lists, keeping
    # that sentence would put a flatly false claim into text meant for
    # submission, so the flagged case gets its own line instead.
    if candidate.get("current_status") == "unknown_candidate":
        lines.append("Not previously flagged by ExoFOP/TOI/archive as of: "
                     f"{char.get('last_verified_unknown_utc')}")
    else:
        lines.append(f"NOW FLAGGED in the archive/TOI lists ({candidate.get('current_status')}) "
                     f"as of {char.get('last_verified_unknown_utc')} -- this is no longer an "
                     f"unknown candidate and should not be submitted as one.")
    return "\n".join(lines)


def _evidence_items(char, centroid, candidate=None):
    """Maps each already-computed check's existing status field/code to a
    display-only pass/fail/caution/skip classification for the evidence
    grid's status icons. Purely presentational -- reads values 08 already
    computed and stored, assigns no new judgement, changes no stored data."""
    items = []

    # Model-stability band, so a wide interval is visible as evidence rather
    # than only as a small "+/-" next to the headline number. The thresholds
    # are descriptive labels for the reader, not a new gate -- nothing about
    # the candidate's tier or probability depends on them.
    if candidate is not None and candidate.get("uncertainty_std") is not None:
        std = candidate["uncertainty_std"]
        p16, p84 = candidate.get("uncertainty_p16"), candidate.get("uncertainty_p84")
        u_status = "pass" if std < 0.05 else "caution" if std < 0.12 else "fail"
        descriptor = ("tight -- the model is stable here" if std < 0.05
                      else "moderate" if std < 0.12
                      else "WIDE -- this score is unstable to how the model was trained")
        items.append({
            "label": "Prediction stability (bootstrap)",
            "status": u_status,
            "value": (f"+/- {std:.3f} ({descriptor}); 68% of "
                      f"{candidate.get('uncertainty_n_members')} refits fall in "
                      f"{p16:.3f}-{p84:.3f}"),
        })

    plausible = char.get("radius_plausible")
    items.append({
        "label": "Physical plausibility",
        "status": "pass" if plausible else "fail" if plausible is not None else "skip",
        "value": char.get("plausibility_verdict") or "--",
    })

    blend_code = char.get("blend_code") or char.get("blending_status", "")
    blend_status = char.get("blending_status") or "--"
    if "HIGH" in blend_status:
        b_status = "fail"
    elif "MODERATE" in blend_status or "AMBIGUOUS" in blend_status or "UNKNOWN" in blend_status:
        b_status = "caution"
    elif blend_status.startswith("LOW") or blend_status.startswith("No "):
        b_status = "pass"
    else:
        b_status = "skip"
    items.append({"label": "Gaia blend risk", "status": b_status, "value": blend_status})

    vsx_code = char.get("vsx_code")
    vsx_status_map = {"HIT": "caution", "NO_HIT": "pass", "ERROR": "skip", "SKIPPED": "skip"}
    vsx_value = char.get("vsx_status") or "--"
    if char.get("vsx_detail"):
        vsx_value += f" -- {char['vsx_detail']}"
    items.append({"label": "Variable-star match (VSX)", "status": vsx_status_map.get(vsx_code, "skip"), "value": vsx_value})

    flagged = char.get("newly_flagged_in_archive_or_exofop")
    archive_value = ("flagged" if flagged else "not flagged") + f" as of {char.get('last_verified_unknown_utc', '--')}"
    if char.get("exofop_note"):
        archive_value += f" -- {char['exofop_note']}"
    items.append({"label": "Archive / ExoFOP / TOI", "status": "fail" if flagged else "pass", "value": archive_value})

    if char.get("ads_code") and char.get("ads_code") != "SKIPPED":
        lit_value = f"via ADS: {char.get('ads_status', '--')}"
        if char.get("ads_links"):
            lit_value += f" -- {char['ads_links']}"
        lit_status = "caution" if char.get("ads_code") == "HIT" else "pass"
    else:
        lit_value = f"via arXiv (ADS key not set): {char.get('arxiv_status', '--')}"
        if char.get("arxiv_links"):
            lit_value += f" -- {char['arxiv_links']}"
        lit_status = "caution" if char.get("arxiv_code") == "HIT" else "pass"
    items.append({"label": "Literature search", "status": lit_status, "value": lit_value})

    if centroid and centroid.get("status") == "completed":
        verdict = centroid.get("verdict", "")
        c_status = "fail" if "contaminant" in verdict.lower() else "pass"
        items.append({"label": "Pixel-level centroid check", "status": c_status,
                      "value": f"{verdict} (shift {centroid.get('shift_pixels'):.3f} px)"})
    elif centroid and centroid.get("status") == "failed":
        # Was "Not run yet" for these too, which is simply untrue -- it WAS
        # run and it did not produce a usable answer, which is a materially
        # different thing for a reviewer to know (and, for the depth-mismatch
        # guard in particular, is itself informative).
        items.append({"label": "Pixel-level centroid check", "status": "skip",
                      "value": f"Attempted, no usable result: {centroid.get('error_message')}"})
    elif centroid and centroid.get("status") == "running":
        items.append({"label": "Pixel-level centroid check", "status": "skip",
                      "value": "Currently running -- see the centroid check section below."})
    else:
        items.append({"label": "Pixel-level centroid check", "status": "skip",
                      "value": "Not run yet -- see the centroid check section below."})

    rv_code = char.get("rv_code")
    rv_status_map = {"LARGE_VARIATION": "fail", "CONSISTENT": "pass",
                      "INSUFFICIENT_BASELINE": "skip", "NO_HIT": "skip",
                      "SKIPPED": "skip", "ERROR": "skip"}
    items.append({"label": "Public RV archive cross-check", "status": rv_status_map.get(rv_code, "skip"),
                  "value": char.get("rv_status") or "Not checked."})

    return items


def _transit_animation_params(char):
    """Derives the (small, JSON-safe) set of numbers the candidate page's
    animated transit illustration needs, straight from this candidate's own
    measured values -- no arbitrary/generic constants. Rp/R_star comes
    directly from the transit depth (depth_fraction = 1 - depth, so
    Rp/R_star = sqrt(depth_fraction)) rather than the derived planet_radius_earth
    field, since that also depends on stellar radius and can be NaN even when
    the raw TLS depth/duration are known."""
    depth_ppm = char.get("transit_depth_ppm")
    duration_hours = char.get("transit_duration_hours")
    period_days = char.get("period_days")
    if depth_ppm is None or duration_hours is None or period_days is None or period_days <= 0:
        return {"has_data": False}
    depth_fraction = max(0.0, min(1.0, depth_ppm / 1e6))
    rp_rstar = depth_fraction ** 0.5
    duration_fraction = min(0.4, (duration_hours / (period_days * 24)))
    return {
        "has_data": True,
        "rp_rstar": rp_rstar,
        "duration_fraction": duration_fraction,
        "depth_ppm": depth_ppm,
        "duration_hours": duration_hours,
        "period_days": period_days,
        "loop_seconds": TRANSIT_ANIM_LOOP_SECONDS,
    }


TRANSIT_ANIM_LOOP_SECONDS = 6


def _candidate_summary_json(c):
    """Builds the payload used by the accordion expand-in-place UI (both in
    the main candidate list and in a dashboard run batch) -- enough detail
    to show the full picture without navigating away, while the standalone
    /candidates/<id> page remains the place for the plot, status history,
    and the re-verify action."""
    char = c.get("characterization")
    if char is None:
        char = db.get_candidate(c["tic_id"])["characterization"]
    return {
        "tic_id": c["tic_id"], "host": c["host"],
        "predicted_probability": c.get("predicted_probability"),
        "confidence_tier": c.get("confidence_tier"),
        "combined_filter_pass": bool(c.get("combined_filter_pass")),
        "combined_filter_tier": c.get("combined_filter_tier"),
        "needs_manual_review": bool(c.get("needs_manual_review")),
        "current_status": c.get("current_status"),
        "first_found_date": c.get("first_found_date"),
        "last_verified_date": c.get("last_verified_date"),
        "supporting_evidence": char.get("supporting_evidence"),
        "doubting_evidence": char.get("doubting_evidence"),
        "plausibility_verdict": char.get("plausibility_verdict"),
        "period_days": char.get("period_days"),
        "transit_depth_ppm": char.get("transit_depth_ppm"),
        "transit_duration_hours": char.get("transit_duration_hours"),
        "planet_radius_earth": char.get("planet_radius_earth"),
        "equilibrium_temp_k": char.get("equilibrium_temp_k"),
        "insolation_flux_earth": char.get("insolation_flux_earth"),
        "stellar_spectral_type_rough": char.get("stellar_spectral_type_rough"),
        "blending_status": char.get("blending_status"),
        "vsx_status": char.get("vsx_status"),
        "arxiv_status": char.get("arxiv_status"),
        "ads_status": char.get("ads_status"),
        "ads_links": char.get("ads_links"),
        "ads_code": char.get("ads_code"),
        "SDE": char.get("SDE"),
        "distinct_transit_count": char.get("distinct_transit_count"),
        "detail_url": url_for("candidate_detail", tic_id=c["tic_id"]),
    }


@app.route("/candidates/<int:tic_id>/json")
def candidate_json(tic_id):
    candidate = db.get_candidate(tic_id)
    if candidate is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(_candidate_summary_json(candidate))


@app.route("/candidates/<int:tic_id>/plot")
def candidate_plot(tic_id):
    """Serves the folded light curve plot, generating it lazily on first
    request (see job_runner.generate_plot_for_candidate) rather than for
    every candidate up front -- most candidates in a 100+ batch will never
    be looked at closely."""
    candidate = db.get_candidate(tic_id)
    if candidate is None:
        return "Candidate not found", 404
    plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f"{candidate['host']}_folded.png")

    if not os.path.exists(plot_path):
        ok, err = job_runner.generate_plot_for_candidate(candidate, plot_path)
        if not ok:
            return f"Plot could not be generated: {err}", 404

    return send_file(plot_path, mimetype="image/png")


@app.route("/candidates/<int:tic_id>/reverify", methods=["POST"])
def reverify_candidate(tic_id):
    candidate = db.get_candidate(tic_id)
    if candidate is None:
        return "Candidate not found", 404
    existing = db.get_reverify_status(tic_id)
    if existing and existing["status"] == "running":
        return jsonify({"error": "already running"}), 409
    job_runner.start_reverify_check(candidate)
    return redirect(url_for("candidate_detail", tic_id=tic_id))


@app.route("/candidates/<int:tic_id>/reverify/status")
def reverify_status(tic_id):
    status = db.get_reverify_status(tic_id)
    if status is None:
        return jsonify({"status": "never_run"})
    return jsonify(status)


@app.route("/candidates/<int:tic_id>/ctoi_update", methods=["POST"])
def ctoi_update(tic_id):
    """Starts a live ExoFOP/archive re-check for this candidate on behalf of
    the CTOI section's Update button.

    Deliberately runs the SAME job as the ExoFOP button rather than only
    re-rendering text: the summary quotes a "not previously flagged as of
    <time>" timestamp, and the only honest way for that timestamp to move to
    now is for a check to actually have run to now. The button therefore
    refreshes the data first and the text second."""
    candidate = db.get_candidate(tic_id)
    if candidate is None:
        return jsonify({"error": "not found"}), 404
    existing = db.get_exofop_refresh(tic_id)
    if existing and existing["status"] == "running":
        return jsonify({"error": "already running"}), 409
    job_runner.start_exofop_refresh(candidate)
    return jsonify({"started": True})


@app.route("/candidates/<int:tic_id>/ctoi_summary")
def ctoi_summary(tic_id):
    """Re-renders the CTOI submission text from current stored state, so the
    Refresh button can update it in place without a page reload."""
    candidate = db.get_candidate(tic_id)
    if candidate is None:
        return jsonify({"error": "not found"}), 404
    text = _build_ctoi_summary(
        candidate, candidate["characterization"],
        db.get_multi_sector_evidence(tic_id), db.get_centroid_evidence(tic_id),
        db.get_exofop_refresh(tic_id))
    return jsonify({"text": text, "last_verified_date": candidate.get("last_verified_date")})


@app.route("/candidates/<int:tic_id>/exofop_refresh", methods=["POST"])
def start_exofop_refresh(tic_id):
    candidate = db.get_candidate(tic_id)
    if candidate is None:
        return "Candidate not found", 404
    existing = db.get_exofop_refresh(tic_id)
    if existing and existing["status"] == "running":
        return jsonify({"error": "already running"}), 409
    job_runner.start_exofop_refresh(candidate)
    return redirect(url_for("candidate_detail", tic_id=tic_id))


@app.route("/candidates/<int:tic_id>/exofop_refresh/status")
def exofop_refresh_status(tic_id):
    status = db.get_exofop_refresh(tic_id)
    if status is None:
        return jsonify({"status": "never_run"})
    return jsonify(status)


@app.route("/candidates/<int:tic_id>/multi_sector", methods=["POST"])
def start_multi_sector(tic_id):
    candidate = db.get_candidate(tic_id)
    if candidate is None:
        return jsonify({"error": "not found"}), 404
    existing = db.get_multi_sector_evidence(tic_id)
    if existing and existing["status"] == "running":
        return jsonify({"error": "already running"}), 409
    job_runner.start_multi_sector_check(candidate)
    return redirect(url_for("candidate_detail", tic_id=tic_id))


@app.route("/candidates/<int:tic_id>/multi_sector/status")
def multi_sector_status(tic_id):
    evidence = db.get_multi_sector_evidence(tic_id)
    if evidence is None:
        return jsonify({"status": "never_run"})
    return jsonify(evidence)


@app.route("/candidates/<int:tic_id>/centroid", methods=["POST"])
def start_centroid(tic_id):
    candidate = db.get_candidate(tic_id)
    if candidate is None:
        return jsonify({"error": "not found"}), 404
    existing = db.get_centroid_evidence(tic_id)
    if existing and existing["status"] == "running":
        return jsonify({"error": "already running"}), 409
    job_runner.start_centroid_check(candidate)
    return redirect(url_for("candidate_detail", tic_id=tic_id))


@app.route("/candidates/<int:tic_id>/centroid/status")
def centroid_status(tic_id):
    evidence = db.get_centroid_evidence(tic_id)
    if evidence is None:
        return jsonify({"status": "never_run"})
    return jsonify(evidence)


@app.route("/jobs/update", methods=["POST"])
def start_update():
    sample_size = int(request.form.get("sample_size", 300))
    if db.get_running_run():
        return jsonify({"error": "A run is already in progress"}), 409
    run_id = job_runner.start_update_job(sample_size)
    return redirect(url_for("dashboard"))


@app.route("/jobs/<int:run_id>/status")
def job_status(run_id):
    run = db.get_run(run_id)
    if run is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(run)


@app.route("/health")
def health():
    """External liveness check for long unattended runs.

    Answers the question the hibernation incident could not: is the SCHEDULER
    still ticking? Process-level "is it up" is not enough -- during that
    freeze the process was up the whole time while progress was dead.

    Returns HTTP 503 (not 200) when the scheduler looks stalled, so a plain
    `curl -f` or any uptime monitor treats it as down without parsing JSON.
    A tick is expected every 60s; 300s of silence is unambiguous.
    """
    hb, age = scheduler_log.read_heartbeat()
    thread_alive = job_runner.scheduler_is_alive()
    stalled = (age is None) or (age > 300) or (not thread_alive)

    # Two different label counts, and only one of them gates a retrain.
    # `processed_watch_labels` is all-time; the retrain trigger compares the
    # count SINCE THE LAST ATTEMPT against the threshold. Reporting only the
    # all-time number next to the threshold read as "138/50, why hasn't it
    # fired?" when the real answer was 1/50.
    try:
        n_labels = db.count_processed_watch_labels_since("2000-01-01 00:00:00 UTC")
    except Exception:
        n_labels = None
    try:
        last = db.list_retrain_attempts(limit=1)
        since = last[0]["triggered_at"] if last else "2000-01-01 00:00:00 UTC"
        n_since = db.count_processed_watch_labels_since(since)
    except Exception:
        since, n_since = None, None
    # Imported here, not at module scope: retrain_pipeline drags in sklearn and
    # pandas, and /health must stay cheap enough to poll. Falls back to the
    # documented default rather than failing the health check outright.
    try:
        import retrain_pipeline
        threshold = retrain_pipeline.RETRAIN_THRESHOLD
    except Exception:
        threshold = 50

    body = {
        "status": "stalled" if stalled else "ok",
        "scheduler_thread_alive": thread_alive,
        "last_tick_at": (hb or {}).get("last_tick_at"),
        "seconds_since_last_tick": None if age is None else round(age, 1),
        "hours_since_last_tick": None if age is None else round(age / 3600.0, 2),
        "last_tick_detail": hb,
        "processed_watch_labels": n_labels,
        "processed_since_last_retrain_attempt": n_since,
        "last_retrain_attempt_at": since,
        "retrain_threshold": threshold,
        "log_file": scheduler_log.LOG_PATH,
    }
    return jsonify(body), (503 if stalled else 200)


if __name__ == "__main__":
    db.init_db()
    job_runner.start_scheduler_thread()
    port = int(os.environ.get("PORT", 5050))
    app.run(host="127.0.0.1", port=port, debug=True, threaded=True, use_reloader=False)
