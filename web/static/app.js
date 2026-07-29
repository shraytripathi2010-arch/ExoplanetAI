// Polls the running job's status every 3s and updates the banner (and, on the
// dashboard, the pipeline step indicator) in place -- no page reload needed
// while an Update run is in progress.
(function () {
  const banner = document.querySelector(".run-banner");
  if (!banner) return;
  const runId = banner.dataset.runId;
  const textEl = document.getElementById("run-banner-text");
  const STAGE_ORDER = ["06_download_unknown", "07_search_unknown", "08_characterize_candidates", "sync", "centroid_check"];

  function updatePipelineSteps(currentStage) {
    const steps = document.querySelectorAll("#update-progress .pipeline-step");
    if (!steps.length) return;
    const curIdx = STAGE_ORDER.indexOf(currentStage);
    steps.forEach((step) => {
      const idx = STAGE_ORDER.indexOf(step.dataset.stage);
      step.classList.toggle("is-done", curIdx >= 0 && idx < curIdx);
      step.classList.toggle("is-active", idx === curIdx);
    });
  }

  function poll() {
    fetch(`/jobs/${runId}/status`)
      .then((r) => r.json())
      .then((data) => {
        if (data.status === "running") {
          textEl.textContent = `${data.current_stage || ""} -- ${data.progress_text || ""}`;
          const progressEl = document.getElementById("run-progress-text");
          if (progressEl) progressEl.textContent = data.progress_text || "";
          updatePipelineSteps(data.current_stage);
          setTimeout(poll, 3000);
        } else {
          window.location.reload();
        }
      })
      .catch(() => setTimeout(poll, 5000));
  }

  setTimeout(poll, 3000);
})();

// ---- CTOI submission-prep: copy-to-clipboard + refresh ----
(function () {
  const actions = document.getElementById("ctoi-actions");
  if (!actions) return;
  const textarea = document.getElementById("ctoi-summary");
  const status = document.getElementById("ctoi-copy-status");
  const copyBtn = document.getElementById("ctoi-copy-btn");
  const updateBtn = document.getElementById("ctoi-update-btn");
  const ticId = actions.dataset.tic;

  copyBtn.addEventListener("click", () => {
    navigator.clipboard.writeText(textarea.value).then(
      () => { status.textContent = "Copied."; setTimeout(() => (status.textContent = ""), 2000); },
      () => { textarea.select(); document.execCommand("copy"); status.textContent = "Copied."; }
    );
  });

  // Update = re-check, THEN re-render. Two steps on purpose.
  //
  // The summary quotes a "not previously flagged by ExoFOP/TOI/archive as of
  // <time>" line. Re-rendering alone could never honestly move that
  // timestamp forward -- it is a claim that a lookup happened at that moment,
  // so the lookup has to actually happen. So this kicks off the same live
  // ExoFOP/archive check the ExoFOP button runs, waits for it, and only then
  // pulls the rebuilt text (which by that point carries the new timestamp,
  // plus anything else that finished while this page was open).
  const poll = (resolve, reject, deadline) => {
    if (Date.now() > deadline) return reject(new Error("timed out waiting for the check"));
    fetch(`/candidates/${ticId}/exofop_refresh/status`)
      .then((r) => r.json())
      .then((d) => {
        if (d.status === "running") return setTimeout(() => poll(resolve, reject, deadline), 1500);
        if (d.status === "failed") return reject(new Error(d.error_message || "the check failed"));
        resolve(d);
      })
      .catch(() => setTimeout(() => poll(resolve, reject, deadline), 3000));
  };

  updateBtn.addEventListener("click", () => {
    updateBtn.disabled = true;
    const before = textarea.value;
    status.textContent = "Re-checking ExoFOP and the archive lists...";
    fetch(`/candidates/${ticId}/ctoi_update`, { method: "POST" })
      .then((r) => {
        if (r.status === 409) throw new Error("a check is already running for this candidate");
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return new Promise((res, rej) => poll(res, rej, Date.now() + 180000));
      })
      .then((checkResult) => fetch(`/candidates/${ticId}/ctoi_summary`)
        .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
        .then((data) => ({ data, checkResult })))
      .then(({ data, checkResult }) => {
        const changed = data.text !== before;
        textarea.value = data.text;
        // Say which of the two actually happened. Claiming an update when
        // nothing moved would just teach you to ignore the message.
        let msg = changed
          ? "Updated -- re-checked live, and the summary changed."
          : "Re-checked live; nothing had changed.";
        // A PARTIAL failure is the confusing case: if one of the two lookups
        // didn't come back, the timestamp correctly refuses to advance, and
        // without this you'd see "nothing had changed" and reasonably assume
        // the button was broken rather than that a service was down. The job
        // already words that caveat for the user; surface it verbatim.
        const caveat = (checkResult && checkResult.result_summary || "").match(/\[([^\]]+)\]/);
        if (caveat) msg += ` Note: ${caveat[1]}`;
        status.textContent = msg;
        setTimeout(() => (status.textContent = ""), 12000);
      })
      // Never let a failed check leave the text looking freshly verified.
      .catch((e) => {
        status.textContent = `Could not update (${e.message}) -- the text above is unchanged, `
          + `and its timestamp still refers to the last successful check.`;
      })
      .finally(() => { updateBtn.disabled = false; });
  });
})();

// ---- live elapsed-time counters for any running action ----
//
// Previously a running check showed one fixed sentence and a spinner for its
// entire duration. A job progressing normally and a job wedged on a stalled
// network call rendered identically, so the only way to tell them apart was
// to give up and reload. A ticking elapsed count, plus an explicit warning
// once a job passes well beyond its normal runtime, makes that difference
// visible without pretending to know more than we do.
(function () {
  const els = document.querySelectorAll(".elapsed[data-started]");
  if (!els.length) return;

  function parseStamp(s) {
    // Timestamps are stored as "YYYY-MM-DD HH:MM:SS UTC".
    const t = Date.parse(s.replace(" UTC", "Z").replace(" ", "T"));
    return isNaN(t) ? null : t;
  }

  function tick() {
    els.forEach((el) => {
      const started = parseStamp(el.dataset.started || "");
      if (started === null) {
        el.textContent = "";
        return;
      }
      const secs = Math.max(0, Math.round((Date.now() - started) / 1000));
      const shown = secs < 60 ? `${secs}s` : `${Math.floor(secs / 60)}m ${secs % 60}s`;
      const expected = parseInt(el.dataset.expected || "0", 10);
      // 3x the normal runtime: late enough not to cry wolf on ordinary
      // slowness, early enough to be useful. This is a hint, not a verdict --
      // the job's own watchdog is what actually ends a stalled run.
      if (expected && secs > expected * 3) {
        el.textContent = `Elapsed: ${shown} -- longer than the usual ~${expected}s. ` +
          `Still running; it will report a real error if it stalls out.`;
      } else {
        el.textContent = `Elapsed: ${shown}`;
      }
    });
  }
  tick();
  setInterval(tick, 1000);
})();

// ---- on-demand per-candidate checks: poll while running ----
//
// One shared implementation instead of four near-identical copies. Beyond
// deduplication this fixes a real gap: the multi-sector and centroid pollers
// discarded the status payload entirely and only looked at data.status, so
// the sub-stage text the backend now reports ("downloading 3 sectors",
// "re-running the transit search") never reached the page. It updates the
// visible line on every poll instead.
(function () {
  const CHECKS = [
    { section: "multi-sector-section", status: "multi-sector-status", path: "multi_sector", interval: 5000 },
    { section: "centroid-section", status: "centroid-status", path: "centroid", interval: 4000 },
    { section: "reverify-section", status: "reverify-status", path: "reverify", interval: 2000 },
    { section: "exofop-refresh-section", status: "exofop-refresh-status", path: "exofop_refresh", interval: 2000 },
  ];

  CHECKS.forEach((cfg) => {
    const section = document.getElementById(cfg.section);
    if (!section) return;
    const statusEl = document.getElementById(cfg.status);
    if (!statusEl) return; // not currently running
    const ticId = section.dataset.tic;
    const spinner = statusEl.querySelector(".spinner");

    function poll() {
      fetch(`/candidates/${ticId}/${cfg.path}/status`)
        .then((r) => r.json())
        .then((data) => {
          if (data.status === "running") {
            if (data.progress_text) {
              statusEl.textContent = data.progress_text;
              if (spinner) statusEl.prepend(spinner);
            }
            setTimeout(poll, cfg.interval);
          } else {
            window.location.reload();
          }
        })
        // Network blip while polling is not the same as the job failing --
        // back off and keep trying rather than reporting a false failure.
        .catch(() => setTimeout(poll, cfg.interval * 2));
    }
    setTimeout(poll, cfg.interval);
  });
})();

// ---- animated transit illustration: simplified 2D canvas loop driven by
// this candidate's own measured Rp/R_star, transit duration fraction, and
// transit depth -- not generic/arbitrary values. Orbit is compressed to a
// fixed loop length (same for every candidate) purely so candidates are
// comparable to look at; only relative size/timing/depth carry real meaning. ----
(function () {
  const wrap = document.querySelector(".transit-anim-wrap");
  const canvas = document.getElementById("transit-anim-canvas");
  if (!wrap || !canvas) return;
  const ctx = canvas.getContext("2d");
  const brightnessEl = document.getElementById("transit-anim-brightness");

  const rpRstar = parseFloat(wrap.dataset.rpRstar);
  const durationFraction = parseFloat(wrap.dataset.durationFraction);
  const depthPpm = parseFloat(wrap.dataset.depthPpm);
  if (!isFinite(rpRstar) || !isFinite(durationFraction) || !isFinite(depthPpm)) return;

  const LOOP_MS = 6000;
  const W = canvas.width, H = canvas.height;
  const cx = W / 2, cy = H / 2;
  const starR = 46;
  // Real Rp/R_star, clamped only so a tiny or oversized ratio stays visible/on-canvas --
  // relative differences between candidates are preserved either side of the clamp.
  const planetR = Math.max(3, Math.min(starR * 0.55, starR * rpRstar));
  const orbitA = W / 2 - planetR - 8;

  // Half-width, in orbital phase (radians), of the in-transit window -- directly
  // proportional to this candidate's real duration/period ratio.
  const halfWidthRad = Math.max(0.03, durationFraction * Math.PI);
  // Depth is real ppm, but a literally-proportional opacity change (e.g. 0.05%)
  // would be imperceptible on screen -- sqrt-scaled and clamped so relative
  // depth ordering between candidates still holds, with the real ppm value
  // always shown as text alongside.
  const depthFraction = depthPpm / 1e6;
  const visualDip = Math.max(0.12, Math.min(0.85, Math.sqrt(depthFraction) * 6));

  function draw(tMs) {
    const t = ((tMs % LOOP_MS) / LOOP_MS) * 2 * Math.PI;
    const x = cx + orbitA * Math.cos(t);
    const z = Math.sin(t); // > 0 in front of the star, < 0 behind it
    const angDist = Math.min(Math.abs(t - Math.PI / 2), Math.abs(t - Math.PI / 2 - 2 * Math.PI));
    const inTransitWindow = z > 0 && angDist <= halfWidthRad;
    const fade = inTransitWindow ? 1 - angDist / halfWidthRad : 0;
    const dipNow = visualDip * Math.max(0, fade);

    ctx.clearRect(0, 0, W, H);

    // star
    const starColor = `rgb(${255 - Math.round(70 * dipNow)}, ${210 - Math.round(90 * dipNow)}, ${120 - Math.round(60 * dipNow)})`;
    ctx.beginPath();
    ctx.arc(cx, cy, starR, 0, 2 * Math.PI);
    ctx.fillStyle = starColor;
    ctx.fill();

    const behindStar = z <= 0 && Math.abs(x - cx) < starR + planetR;
    if (!behindStar) {
      ctx.beginPath();
      ctx.arc(x, cy, planetR, 0, 2 * Math.PI);
      ctx.fillStyle = inTransitWindow ? "#05070d" : "#8fa6c9";
      ctx.fill();
    }

    if (brightnessEl) {
      // The readout shows the real physical dip (from measured depth_ppm),
      // separate from the visually-amplified dipNow used just for the canvas draw.
      const realDipPct = inTransitWindow ? depthFraction * 100 * Math.max(0, fade) : 0;
      brightnessEl.textContent = `Relative brightness: ${(100 - realDipPct).toFixed(3)}%`;
    }

    requestAnimationFrame(draw);
  }
  requestAnimationFrame(draw);
})();

// ---- shared candidate detail renderer, used by both the main candidate
// list's inline expand and a dashboard run-batch's nested candidate rows ----
function fmt(n, digits) {
  return n === null || n === undefined ? "--" : Number(n).toFixed(digits);
}

function renderCandidateDetail(d) {
  const filterLabel = d.combined_filter_pass
    ? `PASS (${d.combined_filter_tier})`
    : d.needs_manual_review
    ? "needs manual review"
    : "fail";
  return `
    <div class="inline-detail">
      <div class="inline-detail-row">
        <span class="badge tier-badge-${(d.confidence_tier || "").toLowerCase()}">${d.confidence_tier || "?"} confidence</span>
        <span class="muted">P(planet) = ${fmt(d.predicted_probability, 3)}</span>
        <span class="muted">Combined filter: ${filterLabel}</span>
        <a href="${d.detail_url}">Full page &rarr;</a>
      </div>
      <p><strong>Reasons this could be real:</strong> ${d.supporting_evidence || "--"}</p>
      <p><strong>Reasons for doubt:</strong> ${d.doubting_evidence || "--"}</p>
      <table class="plain-table">
        <tr><td>Orbital period</td><td>${fmt(d.period_days, 4)} days</td></tr>
        <tr><td>Transit depth / duration</td><td>${fmt(d.transit_depth_ppm, 0)} ppm / ${fmt(d.transit_duration_hours, 2)} hours</td></tr>
        <tr><td>Planet radius</td><td>${fmt(d.planet_radius_earth, 2)} R&#8853;</td></tr>
        <tr><td>Equilibrium temp / insolation</td><td>${fmt(d.equilibrium_temp_k, 0)} K / ${fmt(d.insolation_flux_earth, 1)}&times; Earth</td></tr>
        <tr><td>Host spectral type</td><td>${d.stellar_spectral_type_rough || "--"}</td></tr>
        <tr><td>Detection strength</td><td>SDE ${fmt(d.SDE, 1)}, ${d.distinct_transit_count != null ? Math.trunc(d.distinct_transit_count) : "?"} distinct transits</td></tr>
        <tr><td>Gaia blend risk</td><td>${d.blending_status || "--"}</td></tr>
        <tr><td>Literature search</td><td>${
          d.ads_code && d.ads_code !== "SKIPPED"
            ? `<span class="badge badge-pass">via ADS</span> ${d.ads_status || ""}${d.ads_links ? " -- " + d.ads_links : ""}`
            : `<span class="badge badge-review">via arXiv (ADS key not set)</span> ${d.arxiv_status || ""}`
        }</td></tr>
        <tr><td>Variable-star match</td><td>${d.vsx_status || "--"}</td></tr>
        <tr><td>Plausibility verdict</td><td>${d.plausibility_verdict || "--"}</td></tr>
        <tr><td>First found / last verified</td><td>${d.first_found_date} / ${d.last_verified_date}</td></tr>
      </table>
    </div>`;
}

// ---- main candidate list: expand a row in place ----
document.querySelectorAll(".candidate-row").forEach((row) => {
  const arrow = row.querySelector(".expand-arrow");
  const ticId = row.dataset.tic;
  const detailRow = document.querySelector(`.candidate-detail-row[data-tic="${ticId}"]`);
  if (!arrow || !detailRow) return;
  const container = detailRow.querySelector(".candidate-detail-inline");
  let loaded = false;

  arrow.addEventListener("click", () => {
    const expanded = arrow.getAttribute("aria-expanded") === "true";
    if (!expanded && !loaded) {
      container.textContent = "Loading...";
      fetch(`/candidates/${ticId}/json`)
        .then((r) => r.json())
        .then((d) => {
          container.innerHTML = renderCandidateDetail(d);
          loaded = true;
        });
    }
    arrow.setAttribute("aria-expanded", String(!expanded));
    arrow.innerHTML = !expanded ? "&#9662;" : "&#9656;";
    detailRow.hidden = expanded;
  });
});

// ---- dashboard: expand a run batch to show its candidates, each with its
// own nested expand arrow reusing the same detail renderer ----
document.querySelectorAll(".run-batch").forEach((batch) => {
  const runId = batch.dataset.runId;
  const head = batch.querySelector(".run-batch-head");
  const body = batch.querySelector(".run-batch-body");
  let loaded = false;

  head.addEventListener("click", () => {
    const expanded = head.getAttribute("aria-expanded") === "true";
    if (!expanded && !loaded) {
      body.innerHTML = '<p class="muted">Loading...</p>';
      fetch(`/runs/${runId}/candidates`)
        .then((r) => r.json())
        .then((list) => {
          if (list.length === 0) {
            body.innerHTML = '<p class="muted">No new candidates in this batch.</p>';
            loaded = true;
            return;
          }
          body.innerHTML = list
            .map(
              (d, i) => `
            <div class="batch-candidate" data-tic="${d.tic_id}">
              <button type="button" class="expand-arrow batch-candidate-head" aria-expanded="false">
                <span class="arrow-glyph">&#9656;</span>
                <span class="mono">${d.host}</span>
                <span class="badge tier-badge-${(d.confidence_tier || "").toLowerCase()}">${d.confidence_tier || "?"}</span>
                <span class="muted">P=${fmt(d.predicted_probability, 3)}</span>
              </button>
              <div class="batch-candidate-body" hidden></div>
            </div>`
            )
            .join("");
          list.forEach((d) => {
            const el = body.querySelector(`.batch-candidate[data-tic="${d.tic_id}"]`);
            const btn = el.querySelector(".batch-candidate-head");
            const detailBody = el.querySelector(".batch-candidate-body");
            btn.addEventListener("click", (e) => {
              e.stopPropagation();
              const exp = btn.getAttribute("aria-expanded") === "true";
              if (!exp && !detailBody.dataset.loaded) {
                detailBody.innerHTML = renderCandidateDetail(d);
                detailBody.dataset.loaded = "1";
              }
              btn.setAttribute("aria-expanded", String(!exp));
              btn.querySelector(".arrow-glyph").innerHTML = !exp ? "&#9662;" : "&#9656;";
              detailBody.hidden = exp;
            });
          });
          loaded = true;
        });
    }
    head.setAttribute("aria-expanded", String(!expanded));
    head.querySelector(".arrow-glyph").innerHTML = !expanded ? "&#9662;" : "&#9656;";
    body.hidden = expanded;
  });
});
