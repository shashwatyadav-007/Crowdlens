/* ══════════════════════════════════════════════════
   CrowdLens — app.js v2.3
   Crowd Density Management System
   ══════════════════════════════════════════════════ */

'use strict';

// ─── STATE ────────────────────────────────────────────────────────────────────
const state = {
  file:             null,
  isVideo:          false,
  b64:              { original: '', annotated: '', heatmap: '' },
  videoObjectURL:   null,
  lineChart:        null,
  timelineChart:    null,
  gaugeAnim:        null,
  currentView:      'original',
  // Real-time video analysis
  liveActive:       false,
  liveInterval:     null,
  liveTimeline:     [],    // [{time_s, count}]
  liveFrameCount:   0,
  livePeakCount:    0,
  liveAnalysisBusy: false,
};

// ─── CLOCK ────────────────────────────────────────────────────────────────────
(function startClock() {
  const el   = document.getElementById('clock');
  const tick = () => el.textContent = new Date().toLocaleTimeString('en-GB', { hour12: false });
  tick(); setInterval(tick, 1000);
})();

// ─── PIPELINE INFO — load on startup ─────────────────────────────────────────
(async function loadModelInfo() {
  try {
    const res  = await fetch('/model_info');
    const data = await res.json();
    const map  = { pipeline: 'mi_pipeline', detector: 'mi_detector',
                   occlusion: 'mi_occlusion', density: 'mi_density',
                   binning: 'mi_binning', nms: 'mi_nms' };
    for (const [key, id] of Object.entries(map)) {
      const el = document.getElementById(id);
      if (el && data[key]) el.textContent = data[key];
    }
  } catch (e) {
    // silently ignore if server not yet ready
  }
})();

// ─── FILE INPUT ───────────────────────────────────────────────────────────────
const dropzone = document.getElementById('dropzone');

document.getElementById('fileInput').addEventListener('change',       e => handleFile(e.target.files[0]));
document.getElementById('fileInputChange').addEventListener('change', e => handleFile(e.target.files[0]));

dropzone.addEventListener('dragover',  e => { e.preventDefault(); dropzone.classList.add('drag-over'); });
dropzone.addEventListener('dragleave', ()  => dropzone.classList.remove('drag-over'));
dropzone.addEventListener('drop',      e  => {
  e.preventDefault(); dropzone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0]; if (f) handleFile(f);
});

// ─── HANDLE FILE ─────────────────────────────────────────────────────────────
function handleFile(file) {
  if (!file) return;

  // Release previous Blob URL to free memory immediately
  if (state.videoObjectURL) {
    URL.revokeObjectURL(state.videoObjectURL);
    state.videoObjectURL = null;
  }

  state.file    = file;
  state.isVideo = file.type.startsWith('video/');

  // Update left-panel preview
  document.getElementById('dropzoneInner').classList.add('hidden');
  document.getElementById('filePreview').classList.remove('hidden');
  document.getElementById('fileName').textContent    = file.name;
  document.getElementById('fileSize').textContent    = formatBytes(file.size);
  document.getElementById('fileTypeBadge').textContent = state.isVideo ? 'VIDEO' : 'IMAGE';

  if (state.isVideo) {
    // ── VIDEO — Real-time live analysis ──
    document.getElementById('thumbImg').classList.add('hidden');
    document.getElementById('thumbVideo').classList.remove('hidden');

    // Stop any previous analysis before loading new file
    if (state.liveActive) stopLiveAnalysis();
    // Reset charts so they rebuild fresh for the new video
    if (state.lineChart)     { state.lineChart.destroy();     state.lineChart     = null; }
    if (state.timelineChart) { state.timelineChart.destroy(); state.timelineChart = null; }

    state.videoObjectURL = URL.createObjectURL(file);
    const vid = document.getElementById('resultVideo');
    vid.src   = state.videoObjectURL;

    // Show video player immediately — no upload wait
    document.getElementById('videoBlock').classList.remove('hidden');
    document.getElementById('imgBlock').style.display = 'none';
    document.getElementById('emptyState').classList.add('hidden');
    document.getElementById('results').classList.remove('hidden');

    // Reset live state
    state.liveTimeline   = [];
    state.liveFrameCount = 0;
    state.livePeakCount  = 0;

    // Hide post-analysis sections until first result arrives
    document.getElementById('summaryCards').style.visibility = 'hidden';
    document.getElementById('gaugeRow').style.visibility     = 'hidden';
    hidePostAnalysis();

    document.getElementById('sampleRateConfig').classList.remove('hidden');

    // Auto-play and auto-start analysis simultaneously
    vid.onloadedmetadata = () => {
      vid.play().catch(() => {});
      // Small delay so first frame is rendered before we capture it
      setTimeout(() => startLiveAnalysis(), 400);
    };

  } else {
    // ── IMAGE ──
    // Read locally for instant preview display (no upload needed yet)
    const reader = new FileReader();
    reader.onload = ev => {
      document.getElementById('thumbImg').src = ev.target.result;
      document.getElementById('thumbImg').classList.remove('hidden');
      document.getElementById('thumbVideo').classList.add('hidden');
      // Store so the image tab works immediately
      state.b64.original = ev.target.result;
      // Show image straight away in original tab
      setView('original', document.querySelector('.vtab'));
    };
    reader.readAsDataURL(file);

    document.getElementById('videoBlock').classList.add('hidden');
    document.getElementById('imgBlock').style.display = '';
    document.getElementById('sampleRateConfig').classList.add('hidden');
  }

  document.getElementById('analyzeBtn').disabled = false;
  log(`File loaded: ${file.name} (${formatBytes(file.size)})`, 'accent');
  document.getElementById('alertBanner').classList.add('hidden');
}

// ─── SLIDERS ──────────────────────────────────────────────────────────────────
['gridRows', 'gridCols', 'sampleRate'].forEach(id => {
  const el  = document.getElementById(id);
  const val = document.getElementById(id + 'Val');
  if (el && val) el.addEventListener('input', () => val.textContent = el.value);
});

// ─── RUN ANALYSIS ─────────────────────────────────────────────────────────────
async function runAnalysis() {
  if (!state.file) return;

  if (state.isVideo) {
    // Toggle live analysis on/off
    if (state.liveActive) {
      stopLiveAnalysis();
    } else {
      startLiveAnalysis();
    }
    return;
  }

  // ── IMAGE analysis ──
  setStatus('running', 'ANALYZING');
  document.getElementById('analyzeBtn').disabled = true;
  showProgress(true);
  setProgress(5, 'Preparing upload…');
  log('Starting analysis pipeline…', 'accent');

  const fd = new FormData();
  fd.append('file',      state.file);
  fd.append('grid_rows', document.getElementById('gridRows').value);
  fd.append('grid_cols', document.getElementById('gridCols').value);

  // Animate progress bar while uploading
  let progPct = 5;
  const progTimer = setInterval(() => {
    if (progPct < 80) { progPct += 2; setProgress(progPct, 'Uploading…'); }
  }, 200);

  try {
    const resp = await fetch('/analyze/image', { method: 'POST', body: fd });
    clearInterval(progTimer);
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ error: 'Unknown server error' }));
      throw new Error(err.error || `HTTP ${resp.status}`);
    }
    setProgress(90, 'Processing detection results…');
    const data = await resp.json();
    setProgress(100, 'Analysis complete');
    renderResults(data);
    setStatus('done', 'DONE');
    log('Analysis complete ✓', 'ok');
  } catch (err) {
    clearInterval(progTimer);
    setStatus('error', 'ERROR');
    log(`Error: ${err.message}`, 'err');
    alert(`Analysis failed: ${err.message}`);
  } finally {
    document.getElementById('analyzeBtn').disabled = false;
    setTimeout(() => showProgress(false), 2000);
  }
}

// ─── LIVE VIDEO ANALYSIS ──────────────────────────────────────────────────────
function startLiveAnalysis() {
  if (state.liveActive) return; // already running
  const vid        = document.getElementById('resultVideo');
  const offscreen  = document.getElementById('captureCanvas');
  const overlayCtx = document.getElementById('videoOverlayCanvas').getContext('2d');

  if (vid.paused || vid.ended) vid.play().catch(() => {});

  state.liveActive       = true;
  state.liveTimeline     = [];
  state.liveFrameCount   = 0;
  state.livePeakCount    = 0;
  state.liveAnalysisBusy = false;

  // Show summary area & LIVE badge
  document.getElementById('summaryCards').style.visibility = '';
  document.getElementById('gaugeRow').style.visibility     = '';
  document.getElementById('timelineSection').classList.remove('hidden');
  const badge = document.getElementById('liveBadge');
  if (badge) badge.classList.remove('hidden');

  setStatus('running', 'LIVE');
  log('Live analysis started — video playing + analyzing simultaneously', 'accent');

  // Update button to STOP
  const btn = document.getElementById('analyzeBtn');
  btn.querySelector('.btn-analyze-inner').innerHTML =
    `<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="1"/></svg> STOP ANALYSIS`;
  btn.disabled = false;

  const sampleEvery = parseInt(document.getElementById('sampleRate').value, 10) || 30;
  const gridRows    = parseInt(document.getElementById('gridRows').value, 10);
  const gridCols    = parseInt(document.getElementById('gridCols').value, 10);

  // Interval: aim for ~1 analysis per (sampleEvery/fps) seconds, min 200ms
  const MS_PER_SAMPLE = Math.max(200, Math.round((sampleEvery / 30) * 1000));
  let lastVideoTime = -1;

  state.liveInterval = setInterval(() => {
    if (!state.liveActive)       return;
    if (state.liveAnalysisBusy)  return;
    if (vid.paused || vid.ended) return;

    // Detect video loop — time jumped backwards means video restarted
    if (vid.currentTime < lastVideoTime - 0.5 && lastVideoTime > 0) {
      log('Video looped — resetting timeline', 'dim');
      state.liveTimeline   = [];
      state.liveFrameCount = 0;
      state.livePeakCount  = 0;
      if (state.timelineChart) { state.timelineChart.destroy(); state.timelineChart = null; }
    }

    if (vid.currentTime === lastVideoTime) return;
    lastVideoTime = vid.currentTime;


    // Capture current video frame — scale down for speed (max 640px wide)
    const vw = vid.videoWidth  || vid.clientWidth;
    const vh = vid.videoHeight || vid.clientHeight;
    if (!vw || !vh) return;

    const scale  = Math.min(1.0, 640 / Math.max(vw, vh, 1));
    const capW   = Math.round(vw * scale);
    const capH   = Math.round(vh * scale);
    offscreen.width  = capW;
    offscreen.height = capH;
    offscreen.getContext('2d').drawImage(vid, 0, 0, capW, capH);

    // Encode as JPEG blob — quality 0.75 keeps payload small
    offscreen.toBlob(async (blob) => {
      if (!blob || !state.liveActive) return;
      state.liveAnalysisBusy = true;

      const fd = new FormData();
      fd.append('file',      blob, 'frame.jpg');
      fd.append('grid_rows', gridRows);
      fd.append('grid_cols', gridCols);

      try {
        const resp = await fetch('/analyze/frame', { method: 'POST', body: fd });
        if (!resp.ok) return;
        const data = await resp.json();
        if (!state.liveActive) return;

        state.liveTimeline.push({ time_s: Math.round(vid.currentTime * 10) / 10, count: data.total_people });
        state.livePeakCount  = Math.max(state.livePeakCount, data.total_people);
        state.liveFrameCount++;

        drawLiveOverlay(overlayCtx, vid, data);
        updateLiveUI(data);

        state.b64.heatmap   = data.heatmap_b64;
        state.b64.annotated = data.annotated_b64;
      } catch (e) {
        log(`Frame error: ${e.message}`, 'warn');
      } finally {
        state.liveAnalysisBusy = false;
      }
    }, 'image/jpeg', 0.75);

  }, MS_PER_SAMPLE);
}

function stopLiveAnalysis() {
  state.liveActive = false;
  if (state.liveInterval) { clearInterval(state.liveInterval); state.liveInterval = null; }

  // Hide LIVE badge
  const badge = document.getElementById('liveBadge');
  if (badge) badge.classList.add('hidden');

  setStatus('done', 'DONE');
  log(`Live analysis stopped. Frames analyzed: ${state.liveFrameCount}`, 'ok');

  // Restore button
  const btn = document.getElementById('analyzeBtn');
  btn.querySelector('.btn-analyze-inner').innerHTML =
    `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg> RUN ANALYSIS`;

  if (state.liveTimeline.length > 0) renderTimeline(state.liveTimeline, state.livePeakCount);
}

// ─── LETTERBOX HELPER ────────────────────────────────────────────────────────
// Returns {x, y, w, h} — the actual rendered image rect inside the video
// element, accounting for object-fit:contain letterboxing/pillarboxing.
function getVideoRenderRect(vid) {
  const elemW  = vid.clientWidth;
  const elemH  = vid.clientHeight;
  const vidW   = vid.videoWidth  || elemW;
  const vidH   = vid.videoHeight || elemH;

  const elemAR = elemW / elemH;
  const vidAR  = vidW  / vidH;

  let renderW, renderH, offsetX, offsetY;

  if (vidAR > elemAR) {
    // Letterboxed top/bottom (pillarbox on sides of content is width-constrained)
    renderW = elemW;
    renderH = elemW / vidAR;
    offsetX = 0;
    offsetY = (elemH - renderH) / 2;
  } else {
    // Pillarboxed left/right
    renderH = elemH;
    renderW = elemH * vidAR;
    offsetX = (elemW - renderW) / 2;
    offsetY = 0;
  }

  return { x: offsetX, y: offsetY, w: renderW, h: renderH };
}

// Draw detection results on the overlay canvas that sits over the video
function drawLiveOverlay(ctx, vid, data) {
  const el    = document.getElementById('videoOverlayCanvas');
  const elemW = vid.clientWidth;
  const elemH = vid.clientHeight;

  // Set canvas pixel dimensions to match its CSS size (avoids blurry scaling)
  el.width  = elemW;
  el.height = elemH;

  ctx.clearRect(0, 0, elemW, elemH);

  // Compute the actual rendered video rect (excludes letterbox bars)
  const rect  = getVideoRenderRect(vid);
  const vidW  = vid.videoWidth  || elemW;
  const vidH  = vid.videoHeight || elemH;

  // Scale factors: from video-pixel coords → canvas pixel coords
  const scaleX = rect.w / vidW;
  const scaleY = rect.h / vidH;

  // Helper: convert video-pixel point to canvas coords
  const toCanvasX = (vx) => rect.x + vx * scaleX;
  const toCanvasY = (vy) => rect.y + vy * scaleY;

  // ── Grid cells (only inside the actual video frame) ──
  if (data.cell_counts && data.grid_rows && data.grid_cols) {
    const rows   = data.grid_rows, cols = data.grid_cols;
    const cellW  = rect.w / cols;
    const cellH  = rect.h / rows;
    const maxVal = Math.max(...data.cell_counts.flat(), 1);

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const cnt   = data.cell_counts[r][c] || 0;
        const ratio = cnt / maxVal;
        const cx    = rect.x + c * cellW;
        const cy    = rect.y + r * cellH;

        // Heat fill — only when occupied (low opacity so video is still visible)
        if (ratio > 0.05) {
          ctx.fillStyle = `rgba(${lerpColor(ratio)},${Math.round(ratio * 0.18 * 255)})`;
          ctx.fillRect(cx, cy, cellW, cellH);
        }

        // Grid line — very subtle
        ctx.strokeStyle = 'rgba(120,160,200,0.15)';
        ctx.lineWidth   = 0.5;
        ctx.strokeRect(cx, cy, cellW, cellH);

        // Count label — only show when cell has people
        if (cnt > 0) {
          ctx.fillStyle = ratio > 0.38 ? 'rgba(255,255,255,0.95)' : 'rgba(255,255,255,0.70)';
          const fsize   = Math.max(9, Math.min(13, Math.round(cellW * 0.28)));
          ctx.font      = `bold ${fsize}px "IBM Plex Mono", monospace`;
          ctx.fillText(String(cnt), cx + 4, cy + cellH * 0.58);
        }
      }
    }
  }

  // ── Bounding boxes (clipped to video rect) ──
  ctx.save();
  ctx.beginPath();
  ctx.rect(rect.x, rect.y, rect.w, rect.h);
  ctx.clip();

  if (data.boxes) {
    for (const b of data.boxes) {
      const bx = toCanvasX(b.x1);
      const by = toCanvasY(b.y1);
      const bw = (b.x2 - b.x1) * scaleX;
      const bh = (b.y2 - b.y1) * scaleY;

      ctx.strokeStyle = b.src === 'yolo' ? '#00dc5a' : '#b4ff3c';
      ctx.lineWidth   = 1.5;
      ctx.strokeRect(bx, by, bw, bh);
      // No conf text overlay — keeps video clean
    }
  }
  ctx.restore();

  // ── Status badge — compact, semi-transparent, top-left of video frame ──
  const levelColor = data.level_color || '#22d96a';
  const badgeX     = rect.x + 8;
  const badgeY     = rect.y + 8;
  const badgeText  = `● ${(data.level||'—').toUpperCase()}  ${data.total_people} people  ${data.density_pct}%`;
  const badgeW     = 210;
  const badgeH     = 28;
  ctx.fillStyle    = 'rgba(0,0,0,0.60)';
  ctx.fillRect(badgeX, badgeY, badgeW, badgeH);
  ctx.strokeStyle  = levelColor;
  ctx.lineWidth    = 1;
  ctx.strokeRect(badgeX, badgeY, badgeW, badgeH);
  ctx.fillStyle    = levelColor;
  ctx.font         = 'bold 11px "IBM Plex Mono", monospace';
  ctx.fillText(badgeText, badgeX + 8, badgeY + 18);
}

function lerpColor(ratio) {
  // Returns "r,g,b" string for heatmap tint
  const stops = [[0,[0,60,120]],[0.25,[0,120,60]],[0.5,[180,140,0]],[0.75,[220,80,0]],[1,[200,30,20]]];
  for (let i = 1; i < stops.length; i++) {
    if (ratio <= stops[i][0]) {
      const t = (ratio - stops[i-1][0]) / (stops[i][0] - stops[i-1][0]);
      const a = stops[i-1][1], b = stops[i][1];
      return `${Math.round(a[0]+(b[0]-a[0])*t)},${Math.round(a[1]+(b[1]-a[1])*t)},${Math.round(a[2]+(b[2]-a[2])*t)}`;
    }
  }
  return '200,30,20';
}

function updateLiveUI(data) {
  const levelColors = { Low:'var(--green)', Moderate:'var(--yellow)', High:'var(--orange)', Critical:'var(--red)' };

  // ── Live stats bar (below video) ──
  const lsbP = document.getElementById('lsb_people');
  const lsbD = document.getElementById('lsb_density');
  const lsbL = document.getElementById('lsb_level');
  const lsbI = document.getElementById('lsb_inference');
  const lsbPk= document.getElementById('lsb_peak');
  if (lsbP)  lsbP.textContent  = data.total_people ?? '—';
  if (lsbD)  lsbD.textContent  = (data.density_pct != null ? data.density_pct + '%' : '—');
  if (lsbL) { lsbL.textContent = (data.level || '—').toUpperCase(); lsbL.style.color = levelColors[data.level] || 'var(--accent)'; }
  if (lsbI)  lsbI.textContent  = data.inference_ms != null ? `${data.inference_ms}ms` : '—';
  if (lsbPk) lsbPk.textContent = state.livePeakCount;

  // ── Frame counter in video header ──
  const fc = document.getElementById('liveFrameCounter');
  if (fc) fc.textContent = `FRAMES: ${state.liveFrameCount} · PEAK: ${state.livePeakCount}`;

  // ── Summary cards ──
  const scPeople = document.getElementById('scPeople');
  if (scPeople) scPeople.textContent = data.total_people ?? '—';
  const scDensity = document.getElementById('scDensity');
  if (scDensity) { scDensity.textContent = (data.level || '—').toUpperCase(); scDensity.style.color = levelColors[data.level] || 'var(--text)'; }
  const scTime = document.getElementById('scTime');
  if (scTime) scTime.textContent = data.inference_ms != null ? `${data.inference_ms}ms` : '—';

  // ── Gauge ──
  animateGauge(data.density_pct || 0, data.level_color || '#22d96a', data.level || '—');

  // ── Stat boxes ──
  const gsP  = document.getElementById('gs_people'); if (gsP)  gsP.textContent  = data.total_people ?? '—';
  const gsR  = document.getElementById('gs_res');    if (gsR)  gsR.textContent  = data.resolution || '—';
  const gsT  = document.getElementById('gs_time');   if (gsT)  gsT.textContent  = data.inference_ms != null ? `${data.inference_ms}ms` : '—';
  const gsPk = document.getElementById('gs_peak');   if (gsPk) gsPk.textContent = state.livePeakCount;

  showPostAnalysis();
  renderGrid(data.cell_counts, data.grid_rows, data.grid_cols);
  if (data.hotspots?.length) renderHotspots(data.hotspots);
  renderZoneBreakdown(data.cell_counts, data.grid_rows, data.grid_cols);

  // ── ROW DISTRIBUTION — incremental update ──
  if (data.cell_counts && data.grid_rows) {
    const row_series = data.cell_counts.map((row, i) => ({
      label: `Row ${i + 1}`,
      count: row.reduce((a, b) => a + b, 0),
    }));
    updateRowChart(row_series);
  }

  // ── Live timeline — incremental update (no destroy/recreate) ──
  updateTimelineChart(state.liveTimeline, state.livePeakCount);

  // ── Alert banner ──
  const alertEl = document.getElementById('alertBanner');
  if (alertEl) {
    if (data.density_pct >= 75) {
      alertEl.classList.remove('hidden');
      const txt = document.getElementById('alertText');
      if (txt) txt.textContent = `⚠ CRITICAL DENSITY DETECTED — ${data.total_people} PEOPLE`;
    } else { alertEl.classList.add('hidden'); }
  }

  log(`Frame: ${data.total_people} people · ${data.density_pct}% · ${data.inference_ms}ms`, '');
}

// ─── RENDER RESULTS ───────────────────────────────────────────────────────────
function renderResults(data) {
  document.getElementById('emptyState').classList.add('hidden');
  document.getElementById('results').classList.remove('hidden');

  // Restore hidden sections
  document.getElementById('summaryCards').style.visibility = '';
  document.getElementById('gaugeRow').style.visibility     = '';

  // Store server images
  state.b64.annotated = data.annotated_b64 || '';
  state.b64.heatmap   = data.heatmap_b64   || '';
  if (data.original_b64) state.b64.original = data.original_b64;

  // Refresh image to show original (or first frame for video)
  if (!state.isVideo) {
    setView('original', document.querySelector('.vtab'));
  }
  // For video, keep showing the inline player; image tabs update with first-frame
  if (state.isVideo && state.b64.original) {
    document.getElementById('imgBlock').style.display = '';
    // Put first-frame image tabs below the video player
    setView('original', document.querySelector('.vtab'));
  }

  // ── Alert banner ──
  const level = data.level || '—';
  if (level === 'Critical') {
    document.getElementById('alertText').textContent =
      `⚠ CRITICAL DENSITY — ${data.total_people ?? data.avg_people} PEOPLE DETECTED — IMMEDIATE RESPONSE REQUIRED`;
    document.getElementById('alertBanner').classList.remove('hidden');
  } else {
    document.getElementById('alertBanner').classList.add('hidden');
  }

  // ── Summary cards ──
  const peopleLabel = state.isVideo ? `${data.avg_people} avg` : String(data.total_people ?? '—');
  document.getElementById('scPeople').textContent  = peopleLabel;
  document.getElementById('scDensity').textContent = level.toUpperCase();
  const levelColors = { Low:'var(--green)', Moderate:'var(--yellow)', High:'var(--orange)', Critical:'var(--red)' };
  document.getElementById('scDensity').style.color = levelColors[level] || 'var(--text)';
  document.getElementById('scTime').textContent    = state.isVideo ? `${data.inference_s}s` : `${data.inference_ms}ms`;

  // ── Gauge ──
  animateGauge(data.density_pct ?? 0, data.level_color ?? '#00e5ff', level);

  // ── Stat cards ──
  document.getElementById('gs_people').textContent = peopleLabel;
  document.getElementById('gs_res').textContent    = data.resolution || '—';
  document.getElementById('gs_time').textContent   = state.isVideo ? `${data.inference_s}s` : `${data.inference_ms}ms`;
  document.getElementById('gs_peak').textContent   = state.isVideo ? String(data.peak_people ?? '—') : '—';

  // ── Grid ──
  renderGrid(data.cell_counts, data.grid_rows, data.grid_cols);

  // ── Line chart (below grid, always) ──
  renderLineChart(data);

  // ── Video timeline ──
  const tsEl = document.getElementById('timelineSection');
  if (state.isVideo && data.timeline?.length) {
    tsEl.classList.remove('hidden');
    renderTimeline(data.timeline, data.peak_people);
  } else {
    tsEl.classList.add('hidden');
  }

  // ── Hotspots ──
  renderHotspots(data.hotspots || []);

  // ── Zone breakdown ──
  renderZoneBreakdown(data.cell_counts, data.grid_rows, data.grid_cols);

  // ── Log ──
  if (state.isVideo) {
    log(`Duration: ${data.duration_s}s | FPS: ${data.fps} | Sampled: ${data.sampled_frames} frames`, '');
    log(`Avg people: ${data.avg_people} | Peak: ${data.peak_people} | Density: ${data.density_pct}%`, 'ok');
    log(`Resolution: ${data.orig_resolution} → processed at ${data.resolution}`, '');
  } else {
    log(`Detected: ${data.total_people} people (YOLOv8: ${data.yolo_detected} + Cascade: ${data.extra_detected})`, 'ok');
    log(`Density: ${data.density_pct}% — ${level} | Inference: ${data.inference_ms}ms`, data.level==='Critical'?'err':'ok');
    log(`Input: ${data.orig_resolution} | Processed: ${data.resolution}`, '');
  }

  // Show sections
  showPostAnalysis();
}

function hidePostAnalysis() {
  ['gridHeader','gridWrap','lineChartSection','hotspotHeader','hotspotList'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = 'none';
  });
  // Also hide timeline section on reset
  const ts = document.getElementById('timelineSection');
  if (ts) ts.classList.add('hidden');
}
function showPostAnalysis() {
  [
    {id:'gridHeader',      display:'block'},
    {id:'gridWrap',        display:'block'},
    {id:'lineChartSection',display:'block'},
    {id:'hotspotHeader',   display:'block'},
  ].forEach(({id,display}) => {
    const el = document.getElementById(id);
    if (el) el.style.display = display;
  });
  const hl = document.getElementById('hotspotList');
  if (hl) hl.style.display = 'flex';
}

// ─── LINE CHART ───────────────────────────────────────────────────────────────
/*
 * Image mode  → X axis = grid row, Y axis = people in that row
 * Video mode  → X axis = time (s), Y axis = people count at that frame sample
 * Both use a line chart — smooth, area-filled, color-coded by density.
 */
function renderLineChart(data) {
  const canvas = document.getElementById('lineChart');
  if (!canvas) return;
  if (state.lineChart) { state.lineChart.destroy(); state.lineChart = null; }

  let labels, counts, title, xLabel;

  if (state.isVideo && data.timeline?.length) {
    labels = data.timeline.map(t => `${t.time_s}s`);
    counts = data.timeline.map(t => t.count);
    title  = 'CROWD COUNT OVER TIME';
    xLabel = 'Time (seconds)';
    document.getElementById('lineChartTitle').textContent = title;
  } else if (data.row_series?.length) {
    labels = data.row_series.map(r => r.label);
    counts = data.row_series.map(r => r.count);
    title  = 'CROWD DENSITY — PEOPLE PER GRID ROW';
    xLabel = 'Grid Row';
    document.getElementById('lineChartTitle').textContent = title;
  } else {
    return; // No data
  }

  // Colour-code the area under the curve by average density
  const maxV    = Math.max(...counts, 1);
  const avgRatio = counts.reduce((a,b)=>a+b,0) / counts.length / maxV;
  let lineColor, fillColor;
  if (avgRatio < 0.25)      { lineColor='#22d96a'; fillColor='rgba(34,217,106,0.10)'; }
  else if (avgRatio < 0.50) { lineColor='#f5b800'; fillColor='rgba(245,184,0,0.10)'; }
  else if (avgRatio < 0.75) { lineColor='#ff7c30'; fillColor='rgba(255,124,48,0.10)'; }
  else                      { lineColor='#ff3d5a'; fillColor='rgba(255,61,90,0.10)'; }

  const datasets = [{
    label:           'People',
    data:            counts,
    borderColor:     lineColor,
    backgroundColor: fillColor,
    borderWidth:     2.5,
    pointRadius:     4,
    pointHoverRadius:6,
    pointBackgroundColor: lineColor,
    fill:     true,
    tension:  0.38,
  }];

  // For video, add a dashed peak reference line
  if (state.isVideo && data.peak_people != null) {
    datasets.push({
      label:       'Peak',
      data:        counts.map(() => data.peak_people),
      borderColor: 'rgba(255,61,90,0.45)',
      borderDash:  [6,4],
      borderWidth: 1.5,
      pointRadius: 0,
      fill:        false,
    });
    // Add average line
    const avg = Math.round(data.avg_people);
    datasets.push({
      label:       'Average',
      data:        counts.map(() => avg),
      borderColor: 'rgba(0,229,255,0.35)',
      borderDash:  [3,3],
      borderWidth: 1.5,
      pointRadius: 0,
      fill:        false,
    });
  }

  state.lineChart = new Chart(canvas.getContext('2d'), {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          display: state.isVideo,
          labels: { color: '#8099aa', font: { family: "'IBM Plex Mono', monospace", size: 12 }, boxWidth: 16 },
        },
        tooltip: {
          backgroundColor: '#111722', borderColor: '#253040', borderWidth: 1,
          titleColor: '#00e5ff', bodyColor: '#d0dcea',
          titleFont: { family: "'IBM Plex Mono',monospace", size: 12 },
          bodyFont:  { family: "'Barlow',sans-serif",       size: 13 },
          callbacks: {
            label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y} people`,
          },
        },
      },
      scales: {
        x: {
          grid:  { color: 'rgba(255,255,255,0.04)' },
          ticks: { color: '#4f6070', font: { family:"'IBM Plex Mono',monospace", size: 12 }, maxTicksLimit: 14 },
          title: { display: true, text: xLabel, color: '#4f6070', font: { size: 13 } },
        },
        y: {
          min:   0,
          grid:  { color: 'rgba(255,255,255,0.06)' },
          ticks: { color: '#4f6070', font: { family:"'IBM Plex Mono',monospace", size: 12 }, stepSize: 1 },
          title: { display: true, text: 'People', color: '#4f6070', font: { size: 13 } },
        },
      },
    },
  });
}

// ─── ROW DISTRIBUTION CHART (live incremental) ───────────────────────────────────
// Updates the row distribution chart WITHOUT destroying/recreating it.
// This prevents the blank-flash on every frame analysis.
function updateRowChart(row_series) {
  const canvas = document.getElementById('lineChart');
  if (!canvas) return;

  const labels = row_series.map(r => r.label);
  const counts = row_series.map(r => r.count);

  // Compute color from current distribution
  const maxV     = Math.max(...counts, 1);
  const avgRatio = counts.reduce((a, b) => a + b, 0) / counts.length / maxV;
  let lineColor, fillColor;
  if (avgRatio < 0.25)      { lineColor = '#22d96a'; fillColor = 'rgba(34,217,106,0.10)'; }
  else if (avgRatio < 0.50) { lineColor = '#f5b800'; fillColor = 'rgba(245,184,0,0.10)'; }
  else if (avgRatio < 0.75) { lineColor = '#ff7c30'; fillColor = 'rgba(255,124,48,0.10)'; }
  else                      { lineColor = '#ff3d5a'; fillColor = 'rgba(255,61,90,0.10)'; }

  // Update existing chart without recreating
  if (state.lineChart && !state.lineChart.destroyed) {
    state.lineChart.data.labels                              = labels;
    state.lineChart.data.datasets[0].data                   = counts;
    state.lineChart.data.datasets[0].borderColor            = lineColor;
    state.lineChart.data.datasets[0].backgroundColor        = fillColor;
    state.lineChart.data.datasets[0].pointBackgroundColor   = lineColor;
    state.lineChart.update('none');
    const titleEl = document.getElementById('lineChartTitle');
    if (titleEl) titleEl.textContent = 'ROW DISTRIBUTION — PEOPLE PER ROW';
    return;
  }

  // First time: destroy any stale chart and create fresh
  if (state.lineChart) { state.lineChart.destroy(); state.lineChart = null; }
  const titleEl = document.getElementById('lineChartTitle');
  if (titleEl) titleEl.textContent = 'ROW DISTRIBUTION — PEOPLE PER ROW';

  state.lineChart = new Chart(canvas.getContext('2d'), {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label:            'People',
        data:             counts,
        borderColor:      lineColor,
        backgroundColor:  fillColor,
        borderWidth:      2.5,
        pointRadius:      3,
        pointHoverRadius: 6,
        pointBackgroundColor: lineColor,
        fill:    true,
        tension: 0.38,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#111722', borderColor: '#253040', borderWidth: 1,
          titleColor: '#00e5ff', bodyColor: '#d0dcea',
          callbacks: { label: ctx => `${ctx.parsed.y} people` },
        },
      },
      scales: {
        x: {
          grid:  { color: 'rgba(255,255,255,0.04)' },
          ticks: { color: '#4f6070', font: { family: "'IBM Plex Mono',monospace", size: 11 }, maxTicksLimit: 14 },
          title: { display: true, text: 'Grid Row', color: '#4f6070', font: { size: 12 } },
        },
        y: {
          min:   0,
          grid:  { color: 'rgba(255,255,255,0.06)' },
          ticks: { color: '#4f6070', font: { family: "'IBM Plex Mono',monospace", size: 11 }, stepSize: 1 },
          title: { display: true, text: 'People', color: '#4f6070', font: { size: 12 } },
        },
      },
    },
  });
}

// ─── VIDEO TIMELINE ───────────────────────────────────────────────────────────
// Initial creation of the timeline chart (called once from startLiveAnalysis)
function renderTimeline(timeline, peak) {
  const canvas = document.getElementById('timelineChart');
  if (!canvas) return;
  if (state.timelineChart) { state.timelineChart.destroy(); state.timelineChart = null; }

  const MAX_PTS = 60; // cap visible window
  const slice   = timeline.slice(-MAX_PTS);

  state.timelineChart = new Chart(canvas.getContext('2d'), {
    type: 'line',
    data: {
      labels: slice.map(t => `${t.time_s}s`),
      datasets: [{
        label:           'People',
        data:            slice.map(t => t.count),
        borderColor:     '#00e5ff',
        backgroundColor: 'rgba(0,229,255,0.08)',
        borderWidth: 2, pointRadius: 2, pointHoverRadius: 5,
        fill: true, tension: 0.35,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: false, // disable animation for live updates
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#111722', borderColor: '#253040', borderWidth: 1,
          titleColor: '#00e5ff', bodyColor: '#d0dcea',
          callbacks: { label: ctx => `${ctx.parsed.y} people` },
        },
      },
      scales: {
        x: { grid:{color:'rgba(255,255,255,0.04)'}, ticks:{color:'#4f6070',font:{size:11},maxTicksLimit:10} },
        y: { min:0, grid:{color:'rgba(255,255,255,0.06)'}, ticks:{color:'#4f6070',font:{size:11}} },
      },
    },
  });
}

// Incremental update — much faster than destroy+recreate every frame
function updateTimelineChart(timeline, peak) {
  const MAX_PTS = 60;
  const slice   = timeline.slice(-MAX_PTS);

  if (state.timelineChart && !state.timelineChart.destroyed) {
    state.timelineChart.data.labels                    = slice.map(t => `${t.time_s}s`);
    state.timelineChart.data.datasets[0].data          = slice.map(t => t.count);
    state.timelineChart.update('none'); // 'none' skips animation for live feel
    return;
  }
  // Chart doesn't exist yet — create it
  renderTimeline(timeline, peak);
}

// ─── GRID TABLE ───────────────────────────────────────────────────────────────
function renderGrid(cell_counts, rows, cols) {
  const wrap = document.getElementById('gridTable');
  if (!cell_counts || !rows || !cols) { wrap.innerHTML = ''; return; }

  const maxVal = Math.max(...cell_counts.flat(), 1);
  wrap.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
  wrap.innerHTML = '';

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const cnt   = cell_counts[r][c] ?? 0;
      const ratio = cnt / maxVal;
      const cell  = document.createElement('div');
      cell.className = 'grid-cell';
      cell.style.background = heatColor(ratio);
      cell.style.color = ratio > 0.38 ? '#fff' : 'rgba(255,255,255,0.38)';
      cell.title = `Row ${r+1}, Col ${c+1}: ${cnt} person${cnt!==1?'s':''}`;
      cell.textContent = cnt > 0 ? cnt : '';
      wrap.appendChild(cell);
    }
  }
}

// ─── HOTSPOTS ─────────────────────────────────────────────────────────────────
function renderHotspots(hotspots) {
  const el  = document.getElementById('hotspotList');
  const max = hotspots[0]?.count || 1;
  el.innerHTML = hotspots.length
    ? hotspots.map((h, i) => {
        const ratio = h.count / max;
        let barColor = '#22d96a';
        if (ratio > 0.74) barColor = '#ff3d5a';
        else if (ratio > 0.49) barColor = '#ff7c30';
        else if (ratio > 0.24) barColor = '#f5b800';
        return `<div class="hotspot-item">
          <span class="hotspot-rank">#${i+1}</span>
          <span class="hotspot-loc">Row ${h.row} · Col ${h.col}</span>
          <div class="hotspot-bar-wrap">
            <div class="hotspot-bar" style="width:${Math.round(ratio*100)}%;background:${barColor}"></div>
          </div>
          <span class="hotspot-count">${h.count}</span>
        </div>`;
      }).join('')
    : `<div class="dim small" style="padding:10px 0">No hotspot zones detected.</div>`;
}

// ─── ZONE BREAKDOWN ───────────────────────────────────────────────────────────
function renderZoneBreakdown(cell_counts, rows, cols) {
  if (!cell_counts) return;
  const el   = document.getElementById('zoneBreakdown');
  const midR = Math.floor(rows / 2);
  const midC = Math.floor(cols / 2);
  const quads = { NW: 0, NE: 0, SW: 0, SE: 0 };
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++)
      quads[(r<midR?'N':'S')+(c<midC?'W':'E')] += cell_counts[r][c];

  const maxQ   = Math.max(...Object.values(quads), 1);
  const colors = { NW:'#00e5ff', NE:'#22d96a', SW:'#f5b800', SE:'#ff7c30' };
  el.innerHTML = Object.entries(quads).map(([q,v]) => `
    <div class="zb-item">
      <span class="zb-label">${q}</span>
      <div class="zb-bar-wrap">
        <div class="zb-bar" style="width:${Math.round(v/maxQ*100)}%;background:${colors[q]}"></div>
      </div>
      <span class="zb-val">${v}</span>
    </div>`).join('');
}

// ─── VIEW TABS ────────────────────────────────────────────────────────────────
function setView(view, btn) {
  document.querySelectorAll('.vtab').forEach(t => t.classList.remove('active'));
  if (btn) btn.classList.add('active');

  const labels = { original:'ORIGINAL', annotated:'GRID + BOXES', heatmap:'HEATMAP' };
  const img = document.getElementById('displayImg');
  const b64 = state.b64[view] || '';

  img.src = b64.startsWith('data:') ? b64 : (b64 ? `data:image/jpeg;base64,${b64}` : '');
  document.getElementById('imgLabel').textContent = labels[view] || view.toUpperCase();
}

// ─── GAUGE ────────────────────────────────────────────────────────────────────
function animateGauge(targetPct, color, label) {
  const canvas = document.getElementById('gaugeCanvas');
  const ctx    = canvas.getContext('2d');
  let   cur    = 0;

  if (state.gaugeAnim) cancelAnimationFrame(state.gaugeAnim);

  function draw(pct) {
    const cx = canvas.width/2, cy = canvas.height/2, r = 70;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Track
    ctx.beginPath();
    ctx.arc(cx, cy, r, Math.PI*0.75, Math.PI*2.25);
    ctx.strokeStyle = '#1e2a3a'; ctx.lineWidth = 14; ctx.lineCap = 'round'; ctx.stroke();

    // Fill
    if (pct > 0) {
      ctx.beginPath();
      ctx.arc(cx, cy, r, Math.PI*0.75, Math.PI*0.75 + (pct/100)*Math.PI*1.5);
      ctx.strokeStyle = color; ctx.lineWidth = 14; ctx.lineCap = 'round';
      ctx.shadowBlur = 20; ctx.shadowColor = color; ctx.stroke(); ctx.shadowBlur = 0;
    }
  }

  function step() {
    cur += (targetPct - cur) * 0.09;
    draw(cur);
    document.getElementById('gaugePct').textContent  = Math.round(cur) + '%';
    document.getElementById('gaugeLabel').textContent = label.toUpperCase();
    document.getElementById('gaugePct').style.color   = color;
    if (Math.abs(cur - targetPct) > 0.4)
      state.gaugeAnim = requestAnimationFrame(step);
    else {
      draw(targetPct);
      document.getElementById('gaugePct').textContent = targetPct + '%';
    }
  }
  step();
}

// ─── HELPERS ──────────────────────────────────────────────────────────────────
function heatColor(ratio) {
  const stops = [
    [0,    [10,  20,  48]],
    [0.25, [0,   100, 180]],
    [0.5,  [0,   200, 100]],
    [0.75, [240, 180, 0]],
    [1.0,  [220, 50,  30]],
  ];
  for (let i = 1; i < stops.length; i++) {
    if (ratio <= stops[i][0]) {
      const t = (ratio - stops[i-1][0]) / (stops[i][0] - stops[i-1][0]);
      const a = stops[i-1][1], b = stops[i][1];
      return `rgb(${lerp(a[0],b[0],t)},${lerp(a[1],b[1],t)},${lerp(a[2],b[2],t)})`;
    }
  }
  return 'rgb(220,50,30)';
}
const lerp = (a,b,t) => Math.round(a+(b-a)*t);

function formatBytes(n) {
  if (n < 1024)       return n + ' B';
  if (n < 1048576)    return (n/1024).toFixed(1) + ' KB';
  return (n/1048576).toFixed(1) + ' MB';
}

function setStatus(type, text) {
  document.getElementById('statusLed').className = 'status-led ' + type;
  document.getElementById('statusText').textContent = text;
}

function showProgress(show) {
  document.getElementById('progressWrap').classList.toggle('hidden', !show);
}
function setProgress(pct, label) {
  document.getElementById('progressBar').style.width    = pct + '%';
  document.getElementById('progressLabel').textContent  = label;
}

function log(msg, cls = '') {
  const panel = document.getElementById('logPanel');
  const line  = document.createElement('div');
  line.className = 'log-line' + (cls ? ' ' + cls : '');
  line.textContent = `[${new Date().toLocaleTimeString('en-GB',{hour12:false})}] ${msg}`;
  panel.appendChild(line);
  // Cap log at 80 lines to prevent DOM bloat
  while (panel.children.length > 80) panel.removeChild(panel.firstChild);
  panel.scrollTop = panel.scrollHeight;
}