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

// ---- CTOI submission-prep: copy-to-clipboard ----
(function () {
  const btn = document.getElementById("ctoi-copy-btn");
  if (!btn) return;
  const textarea = document.getElementById("ctoi-summary");
  const status = document.getElementById("ctoi-copy-status");
  btn.addEventListener("click", () => {
    navigator.clipboard.writeText(textarea.value).then(
      () => { status.textContent = "Copied."; setTimeout(() => (status.textContent = ""), 2000); },
      () => { textarea.select(); document.execCommand("copy"); status.textContent = "Copied."; }
    );
  });
})();

// ---- multi-sector strengthening: poll while running ----
(function () {
  const section = document.getElementById("multi-sector-section");
  if (!section) return;
  const statusEl = document.getElementById("multi-sector-status");
  if (!statusEl) return; // not currently running
  const ticId = section.dataset.tic;

  function poll() {
    fetch(`/candidates/${ticId}/multi_sector/status`)
      .then((r) => r.json())
      .then((data) => {
        if (data.status === "running") {
          setTimeout(poll, 5000);
        } else {
          window.location.reload();
        }
      })
      .catch(() => setTimeout(poll, 8000));
  }
  setTimeout(poll, 5000);
})();

// ---- centroid check: poll while running ----
(function () {
  const section = document.getElementById("centroid-section");
  if (!section) return;
  const statusEl = document.getElementById("centroid-status");
  if (!statusEl) return;
  const ticId = section.dataset.tic;

  function poll() {
    fetch(`/candidates/${ticId}/centroid/status`)
      .then((r) => r.json())
      .then((data) => {
        if (data.status === "running") {
          setTimeout(poll, 4000);
        } else {
          window.location.reload();
        }
      })
      .catch(() => setTimeout(poll, 6000));
  }
  setTimeout(poll, 4000);
})();

// ---- reverify: poll while running ----
(function () {
  const section = document.getElementById("reverify-section");
  if (!section) return;
  const statusEl = document.getElementById("reverify-status");
  if (!statusEl) return; // not currently running
  const ticId = section.dataset.tic;

  function poll() {
    fetch(`/candidates/${ticId}/reverify/status`)
      .then((r) => r.json())
      .then((data) => {
        if (data.status === "running") {
          setTimeout(poll, 2000);
        } else {
          window.location.reload();
        }
      })
      .catch(() => setTimeout(poll, 4000));
  }
  setTimeout(poll, 2000);
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
