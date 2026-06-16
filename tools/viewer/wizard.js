// Dataset Wizard frontend

// ── State ─────────────────────────────────────────────────────────────────────

let _browserTarget  = null;  // input element that triggered the file browser
let _browserPath    = null;  // path currently shown in file browser
let _currentJob     = null;  // active EventSource
let _datasetMeta    = {};    // path → { state_names, action_names }
let _previewCharts  = [];    // active Chart.js instances in preview panel

// ── Bootstrap ─────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
    setupConfig();
    setupPipeline();
    setupRun();
    setupPreview();
    setupFileBrowser();
    await loadConfig();
    loadPreviewDatasets();
});

// ── Config ────────────────────────────────────────────────────────────────────

function setupConfig() {
    document.getElementById('btn-browse-base')
        .addEventListener('click', () => openFileBrowser(document.getElementById('base-path')));
    document.getElementById('btn-refresh-datasets')
        .addEventListener('click', () => refreshDatasets());
    document.getElementById('base-path')
        .addEventListener('change', () => refreshDatasets());
    document.getElementById('btn-select-all')
        .addEventListener('click', () => setAllChecked(true));
    document.getElementById('btn-deselect-all')
        .addEventListener('click', () => setAllChecked(false));
}

function setAllChecked(checked) {
    document.querySelectorAll('#dataset-checklist input[type=checkbox]')
        .forEach(cb => { cb.checked = checked; });
}

async function loadConfig() {
    try {
        const cfg = await api('/api/wizard/config');
        if (cfg.error) return;

        if (cfg.base_path)   set('base-path',   cfg.base_path);
        if (cfg.merged_name) set('merged-name',  cfg.merged_name);
        if (cfg.start_from)  set('start-from',   cfg.start_from);
        if (cfg.stop_at)     set('stop-at',       cfg.stop_at);

        const jee  = cfg.joint_to_ee || {};
        if (jee.ee_frame)   set('ee-frame', jee.ee_frame);
        if (jee.rot_repr)   set('ee-rot-repr', jee.rot_repr);
        document.getElementById('ee-include-joint-repr').checked =
            jee.include_joint_repr !== false;  // default true when absent
        document.getElementById('ee-enabled').checked = jee.enabled !== false;
        updateEEEnabled();

        const sftp = cfg.sftp || {};
        if (sftp.hostname)    set('sftp-hostname', sftp.hostname);
        if (sftp.port)        set('sftp-port',     sftp.port);
        if (sftp.username)    set('sftp-username',  sftp.username);
        // password intentionally not loaded — must be entered each session
        if (sftp.remote_path) set('sftp-remote',    sftp.remote_path);

        updatePipelineHighlight();

        if (cfg.base_path) {
            await refreshDatasets(cfg.datasets || null);
        }
    } catch (e) {
        console.warn('Could not load config:', e);
    }
}

function buildConfig() {
    const datasets = [...document.querySelectorAll('#dataset-checklist input[type=checkbox]:checked')]
        .map(cb => cb.value);

    const cfg = {
        base_path:   get('base-path'),
        merged_name: get('merged-name') || 'merged_dataset',
        start_from:  get('start-from'),
        stop_at:     get('stop-at'),
        datasets,
        joint_to_ee: {
            enabled:            document.getElementById('ee-enabled').checked,
            ee_frame:           get('ee-frame'),
            rot_repr:           get('ee-rot-repr') || 'both',
            include_joint_repr: document.getElementById('ee-include-joint-repr').checked,
        },
    };

    const hostname = get('sftp-hostname');
    if (hostname) {
        cfg.sftp = {
            hostname,
            port:        parseInt(get('sftp-port')) || 22,
            username:    get('sftp-username'),
            password:    get('sftp-password'),
            remote_path: get('sftp-remote'),
        };
    }
    return cfg;
}

async function saveConfig() {
    try {
        await apiPost('/api/wizard/config', buildConfig());
        showStatus('Config saved', 'ok');
    } catch (e) {
        showStatus('Save failed: ' + e.message, 'err');
    }
}

async function refreshDatasets(preChecked = null) {
    const basePath = get('base-path');
    const list = document.getElementById('dataset-checklist');

    if (!basePath) {
        list.innerHTML = '<span class="hint">Set a base path first</span>';
        return;
    }

    list.innerHTML = '<span class="hint">Loading…</span>';

    try {
        const datasets = await api(`/api/wizard/datasets?path=${encodeURIComponent(basePath)}`);

        if (!Array.isArray(datasets) || datasets.length === 0) {
            list.innerHTML = '<span class="hint">No datasets found at this path</span>';
            return;
        }

        // Determine which names to check
        const existingChecked = new Set(
            [...document.querySelectorAll('#dataset-checklist input:checked')].map(cb => cb.value)
        );
        const checkSet = preChecked === null ? existingChecked
            : preChecked.length === 0       ? null           // null = check all
            : new Set(preChecked);

        list.innerHTML = datasets.map(d => {
            const checked = checkSet === null || checkSet.has(d.name) ? 'checked' : '';
            return `<label class="check-item">
                <input type="checkbox" value="${d.name}" ${checked}>
                <span class="check-name">${d.name}</span>
                <span class="check-meta">${d.total_episodes} ep · ${d.total_frames} fr</span>
            </label>`;
        }).join('');
    } catch (e) {
        list.innerHTML = `<span class="hint err">Error: ${e.message}</span>`;
    }
}

// ── Pipeline ──────────────────────────────────────────────────────────────────

const STAGES = ['conversion', 'merge', 'ee_conversion', 'compress', 'upload'];

function setupPipeline() {
    document.getElementById('start-from').addEventListener('change', updatePipelineHighlight);
    document.getElementById('stop-at').addEventListener('change', updatePipelineHighlight);
    document.getElementById('ee-enabled').addEventListener('change', () => {
        updateEEEnabled();
        updatePipelineHighlight();
    });
    updateEEEnabled();
    updatePipelineHighlight();
}

function updateEEEnabled() {
    const on = document.getElementById('ee-enabled').checked;
    document.getElementById('details-ee').style.display = on ? '' : 'none';
}

function updatePipelineHighlight() {
    const si   = STAGES.indexOf(get('start-from'));
    const ei   = STAGES.indexOf(get('stop-at'));
    const eeOn = document.getElementById('ee-enabled').checked;

    document.querySelectorAll('.stage-node').forEach((node, i) => {
        const stage  = node.dataset.stage;
        const active = i >= si && i <= ei && !(stage === 'ee_conversion' && !eeOn);
        node.classList.toggle('active', active);
        node.classList.toggle('dim',    !active);
    });
    document.querySelectorAll('.stage-connector').forEach((el, i) => {
        el.classList.toggle('active', i >= si && i < ei);
    });
}

// ── Run ───────────────────────────────────────────────────────────────────────

function setupRun() {
    document.getElementById('btn-save-config').addEventListener('click', saveConfig);
    document.getElementById('btn-run').addEventListener('click', runPipeline);
    document.getElementById('btn-stop').addEventListener('click', stopPipeline);
    document.getElementById('btn-clear-log').addEventListener('click', () => {
        document.getElementById('log-output').innerHTML = '';
    });
}

async function runPipeline() {
    const cfg = buildConfig();
    try {
        await apiPost('/api/wizard/config', cfg);
    } catch (e) {
        showStatus('Failed to save config: ' + e.message, 'err');
        return;
    }

    let result;
    try {
        result = await apiPost('/api/wizard/run', {
            start_from:    cfg.start_from,
            stop_at:       cfg.stop_at,
            ee_frame:      cfg.joint_to_ee?.ee_frame,
            ee_rot_repr:   cfg.joint_to_ee?.rot_repr || 'both',
            ee_joint_repr: cfg.joint_to_ee?.include_joint_repr !== false,
            skip_ee:       cfg.joint_to_ee?.enabled === false,
        });
    } catch (e) {
        showStatus('Failed to start pipeline: ' + e.message, 'err');
        return;
    }

    document.getElementById('card-log').style.display = '';
    document.getElementById('log-output').textContent = '';
    document.getElementById('btn-run').disabled = true;
    document.getElementById('btn-stop').style.display = '';
    showStatus('Running…', 'running');

    _currentJob = new EventSource(`/api/wizard/log?job=${result.job_id}`);

    _currentJob.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        if (msg.done) {
            _currentJob.close();
            _currentJob = null;
            document.getElementById('btn-run').disabled = false;
            document.getElementById('btn-stop').style.display = 'none';
            const ok = msg.returncode === 0;
            showStatus(ok ? '✓ Done' : `✗ Exited with code ${msg.returncode}`, ok ? 'ok' : 'err');
            if (ok) loadPreviewDatasets();
        } else {
            appendLog(msg.line);
        }
    };

    _currentJob.onerror = () => {
        if (_currentJob) { _currentJob.close(); _currentJob = null; }
        document.getElementById('btn-run').disabled = false;
        document.getElementById('btn-stop').style.display = 'none';
        showStatus('Connection lost', 'err');
    };
}

function stopPipeline() {
    if (_currentJob) { _currentJob.close(); _currentJob = null; }
    document.getElementById('btn-run').disabled = false;
    document.getElementById('btn-stop').style.display = 'none';
    showStatus('Stopped', 'err');
}

function appendLog(line) {
    const el   = document.getElementById('log-output');
    const node = document.createElement('span');
    node.innerHTML = ansiToHtml(line) + '\n';
    el.appendChild(node);
    el.scrollTop = el.scrollHeight;
}

// Convert ANSI SGR escape sequences to HTML spans.
// Handles: 8-color, 256-color, true-color (24-bit), bold, dim, italic.
function ansiToHtml(text) {
    const esc = s => s
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    // ANSI 4-bit palette mapped to dark-theme colors
    const ansi16 = [
        '#21262d', '#ff6b6b', '#51cf66', '#ffd43b',
        '#74c0fc', '#da77f2', '#3bc9db', '#abb8c3',
        '#484f58', '#ff8787', '#69db7c', '#ffe066',
        '#91c7f3', '#e599f7', '#66d9e8', '#f1f3f5',
    ];

    let state = { fg: null, bold: false, dim: false, italic: false };
    let currentStyle = null;
    let html = '';

    function stateStyle(s) {
        let st = '';
        if (s.fg)     st += `color:${s.fg};`;
        if (s.bold)   st += 'font-weight:bold;';
        if (s.dim)    st += 'opacity:0.55;';
        if (s.italic) st += 'font-style:italic;';
        return st || null;
    }

    function syncSpan() {
        const ns = stateStyle(state);
        if (ns === currentStyle) return;
        if (currentStyle !== null) html += '</span>';
        if (ns !== null) html += `<span style="${ns}">`;
        currentStyle = ns;
    }

    const re = /\x1b\[([0-9;]*)([A-Za-z])/g;
    let last = 0, m;

    while ((m = re.exec(text)) !== null) {
        if (m.index > last) { syncSpan(); html += esc(text.slice(last, m.index)); }
        last = re.lastIndex;
        if (m[2] !== 'm') continue;

        const codes = m[1] ? m[1].split(';').map(Number) : [0];
        let i = 0;
        while (i < codes.length) {
            const c = codes[i];
            if (c === 0)  { state = { fg: null, bold: false, dim: false, italic: false }; }
            else if (c === 1)  { state.bold   = true; }
            else if (c === 2)  { state.dim    = true; }
            else if (c === 3)  { state.italic = true; }
            else if (c === 22) { state.bold   = false; state.dim = false; }
            else if (c === 39) { state.fg = null; }
            else if (c >= 30 && c <= 37) { state.fg = ansi16[c - 30]; }
            else if (c >= 90 && c <= 97) { state.fg = ansi16[c - 90 + 8]; }
            else if (c === 38) {
                if (codes[i + 1] === 5 && i + 2 < codes.length) {
                    const idx = codes[i + 2];
                    if (idx < 16) {
                        state.fg = ansi16[idx];
                    } else if (idx < 232) {
                        const n = idx - 16;
                        const r = Math.floor(n / 36), g = Math.floor((n % 36) / 6), b = n % 6;
                        state.fg = `rgb(${r * 51},${g * 51},${b * 51})`;
                    } else {
                        const v = (idx - 232) * 10 + 8;
                        state.fg = `rgb(${v},${v},${v})`;
                    }
                    i += 2;
                } else if (codes[i + 1] === 2 && i + 4 < codes.length) {
                    state.fg = `rgb(${codes[i + 2]},${codes[i + 3]},${codes[i + 4]})`;
                    i += 4;
                }
            }
            i++;
        }
    }

    if (last < text.length) { syncSpan(); html += esc(text.slice(last)); }
    if (currentStyle !== null) html += '</span>';
    return html;
}

function showStatus(msg, type) {
    const el = document.getElementById('run-status');
    el.textContent = msg;
    el.className = 'run-status ' + (type || '');
}

// ── Dataset Preview ───────────────────────────────────────────────────────────

function setupPreview() {
    document.getElementById('btn-refresh-preview')
        .addEventListener('click', loadPreviewDatasets);
    document.getElementById('preview-dataset')
        .addEventListener('change', onPreviewDatasetChange);
    document.getElementById('preview-episode')
        .addEventListener('change', onPreviewEpisodeChange);
}

async function loadPreviewDatasets() {
    const sel      = document.getElementById('preview-dataset');
    const cur      = sel.value;
    const basePath = get('base-path');
    const url      = basePath
        ? `/api/datasets?path=${encodeURIComponent(basePath)}`
        : '/api/datasets';
    try {
        const datasets = await api(url);
        _datasetMeta = {};
        for (const d of datasets) {
            _datasetMeta[d.path] = {
                state_names:  d.state_names  || [],
                action_names: d.action_names || [],
            };
        }
        sel.innerHTML = '<option value="">— select —</option>'
            + datasets.map(d =>
                `<option value="${d.path}"${d.path === cur ? ' selected' : ''}>${d.name}</option>`
            ).join('');
    } catch (e) {
        console.warn('Could not load datasets for preview:', e);
    }
}

async function onPreviewDatasetChange() {
    const dsPath = get('preview-dataset');
    const epSel  = document.getElementById('preview-episode');
    epSel.innerHTML = '<option value="">—</option>';
    epSel.disabled  = true;

    if (!dsPath) {
        previewMsg('Select a dataset and episode to visualize data');
        return;
    }
    previewMsg('Loading episodes…');

    try {
        const episodes = await api(`/api/episodes?dataset=${encodeURIComponent(dsPath)}`);
        epSel.innerHTML = '<option value="">— select —</option>'
            + episodes.map(ep =>
                `<option value="${ep.episode}">Episode ${ep.episode} (${ep.frames} frames)</option>`
            ).join('');
        epSel.disabled = false;

        const total = episodes.reduce((s, e) => s + e.frames, 0);
        previewMsg(`${episodes.length} episodes · ${total} total frames — select an episode`);
    } catch (e) {
        previewMsg('Failed to load episodes: ' + e.message);
    }
}

async function onPreviewEpisodeChange() {
    const dsPath = get('preview-dataset');
    const epIdx  = get('preview-episode');
    if (!dsPath || epIdx === '') {
        previewMsg('Select an episode');
        return;
    }
    previewMsg('Loading frames…');

    try {
        const result = await api(
            `/api/frames?dataset=${encodeURIComponent(dsPath)}&episode=${epIdx}`
        );
        if (!result.frames?.length) {
            previewMsg('No frames found for this episode');
            return;
        }
        renderPreview(result.frames);
    } catch (e) {
        previewMsg('Failed to load frames: ' + e.message);
    }
}

function previewMsg(text) {
    document.getElementById('preview-body').innerHTML =
        `<div class="preview-hint">${text}</div>`;
}

function renderPreview(frames) {
    // destroy previous Chart.js instances
    for (const c of _previewCharts) c.destroy();
    _previewCharts = [];

    const stateData  = frames.map(f => f['observation.state']).filter(Boolean);
    const actionData = frames.map(f => f['action']).filter(Boolean);
    const stateDim   = stateData[0]?.length  ?? 0;
    const actionDim  = actionData[0]?.length ?? 0;

    const meta        = _datasetMeta[get('preview-dataset')] || {};
    const stateNames  = meta.state_names  || [];
    const actionNames = meta.action_names || [];

    const body = document.getElementById('preview-body');
    body.innerHTML = `
        <div class="preview-info">
            <span>${frames.length} frames</span>
            <span>state ${stateDim}D</span>
            <span>action ${actionDim}D</span>
        </div>
        <div class="charts-grid">
            <div class="chart-wrap">
                <div class="chart-title">observation.state — ${stateDim} dims</div>
                <div class="chart-cj-container"><canvas id="chart-state"></canvas></div>
            </div>
            <div class="chart-wrap">
                <div class="chart-title">action — ${actionDim} dims</div>
                <div class="chart-cj-container"><canvas id="chart-action"></canvas></div>
            </div>
        </div>`;

    if (stateData.length  > 1) {
        const c = makeWizardChart(
            document.getElementById('chart-state'), stateData, stateNames,
            'observation.state'
        );
        if (c) _previewCharts.push(c);
    }
    if (actionData.length > 1) {
        const c = makeWizardChart(
            document.getElementById('chart-action'), actionData, actionNames,
            'action'
        );
        if (c) _previewCharts.push(c);
    }
}

// ── Chart rendering ───────────────────────────────────────────────────────────

let _wizTooltipEl = null;

function getWizTooltipEl() {
    if (!_wizTooltipEl) {
        _wizTooltipEl = document.createElement('div');
        _wizTooltipEl.style.cssText = [
            'position:fixed',
            'background:#161b22',
            'border:1px solid #30363d',
            'border-radius:4px',
            'padding:6px 10px',
            'font:10px Consolas,Menlo,Monaco,monospace',
            'color:#e6edf3',
            'pointer-events:none',
            'z-index:9999',
            'display:none',
            'white-space:nowrap',
            'max-height:50vh',
            'overflow-y:auto',
        ].join(';');
        document.body.appendChild(_wizTooltipEl);
    }
    return _wizTooltipEl;
}

function externalWizardTooltip({ chart, tooltip }) {
    const el = getWizTooltipEl();
    if (!tooltip.opacity) { el.style.display = 'none'; return; }

    const pts = tooltip.dataPoints || [];
    if (!pts.length) { el.style.display = 'none'; return; }

    let html = `<div style="color:#8b949e;margin-bottom:4px;font-weight:700">frame ${pts[0].label}</div>`;
    for (const pt of pts) {
        const color = pt.dataset.borderColor;
        html += `<div style="display:flex;align-items:center;gap:5px;margin-bottom:1px">` +
            `<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:${color};flex-shrink:0"></span>` +
            `<span style="color:#8b949e">${pt.dataset.label}:</span>` +
            `<span>${typeof pt.raw === 'number' ? pt.raw.toFixed(4) : pt.raw}</span>` +
            `</div>`;
    }
    el.innerHTML = html;
    el.style.display = 'block';

    const canvasRect = chart.canvas.getBoundingClientRect();
    let x = canvasRect.left + tooltip.caretX + 14;
    let y = canvasRect.top  + tooltip.caretY - el.offsetHeight / 2;

    const elW = el.offsetWidth, elH = el.offsetHeight;
    if (x + elW > window.innerWidth  - 8) x = canvasRect.left + tooltip.caretX - elW - 14;
    if (y < 8)                             y = 8;
    if (y + elH > window.innerHeight - 8)  y = window.innerHeight - elH - 8;

    el.style.left = x + 'px';
    el.style.top  = y + 'px';
}

const PALETTE = [
    '#4a9eff','#e34c26','#4ec994','#e5b600','#c084fc',
    '#fb7185','#38bdf8','#a3e635','#f97316','#818cf8',
    '#f472b6','#34d399','#facc15','#60a5fa','#a78bfa',
    '#fb923c','#2dd4bf','#e879f9','#4ade80','#fbbf24',
];

function makeWizardChart(canvas, data, names, title) {
    if (!data || data.length < 2 || !data[0]?.length) return null;
    const nDim   = data[0].length;
    const labels = data.map((_, i) => i);

    const datasets = Array.from({ length: nDim }, (_, s) => ({
        label:       names[s] ?? `dim ${s}`,
        data:        data.map(row => row[s]),
        borderColor: PALETTE[s % PALETTE.length],
        backgroundColor: PALETTE[s % PALETTE.length] + '22',
        borderWidth: 1.5,
        pointRadius: 0,
        tension:     0,
    }));

    return new Chart(canvas, {
        type: 'line',
        data: { labels, datasets },
        options: {
            animation:            false,
            maintainAspectRatio:  false,
            responsive:           true,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: {
                    display:  true,
                    position: 'bottom',
                    labels: {
                        color:    '#8b949e',
                        boxWidth: 10,
                        font:     { size: 10, family: 'Consolas,Menlo,Monaco,monospace' },
                    },
                },
                title: { display: false },
                tooltip: {
                    enabled:  false,
                    external: externalWizardTooltip,
                },
                zoom: {
                    pan:   { enabled: true,  mode: 'x' },
                    zoom:  { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'x' },
                },
            },
            scales: {
                x: {
                    ticks: {
                        color:    '#484f58',
                        maxRotation: 0,
                        font:     { size: 9 },
                        maxTicksLimit: 8,
                    },
                    grid: { color: '#21262d' },
                },
                y: {
                    ticks: { color: '#484f58', font: { size: 9 } },
                    grid:  { color: '#21262d' },
                },
            },
        },
    });
}

// ── File browser ──────────────────────────────────────────────────────────────

function setupFileBrowser() {
    document.getElementById('btn-close-browser')
        .addEventListener('click', closeFileBrowser);
    document.getElementById('btn-cancel-browser')
        .addEventListener('click', closeFileBrowser);
    document.getElementById('btn-select-dir')
        .addEventListener('click', selectCurrentDir);
    document.getElementById('modal-filebrowser')
        .addEventListener('click', e => { if (e.target === e.currentTarget) closeFileBrowser(); });
}

function openFileBrowser(targetInput) {
    _browserTarget = targetInput;
    document.getElementById('modal-filebrowser').style.display = '';
    browseDir(targetInput.value.trim() || '~');
}

function closeFileBrowser() {
    document.getElementById('modal-filebrowser').style.display = 'none';
    _browserTarget = null;
}

function selectCurrentDir() {
    if (_browserTarget && _browserPath) {
        _browserTarget.value = _browserPath;
        _browserTarget.dispatchEvent(new Event('change'));
    }
    closeFileBrowser();
}

async function browseDir(path) {
    const entries = document.getElementById('browser-entries');
    entries.innerHTML = '<div class="browser-msg">Loading…</div>';

    try {
        const result = await api(`/api/files?path=${encodeURIComponent(path)}`);
        if (result.error) {
            entries.innerHTML = `<div class="browser-msg err">${result.error}</div>`;
            return;
        }

        _browserPath = result.path;
        document.getElementById('browser-current-path').textContent = result.path;

        let html = '';
        if (result.parent) {
            html += `<div class="browser-item browser-up" data-path="${result.parent}">
                <span class="bi-icon">↑</span><span class="bi-name">..</span>
            </div>`;
        }
        for (const e of result.entries) {
            const badge = e.is_dataset
                ? '<span class="ds-badge">dataset</span>' : '';
            html += `<div class="browser-item${e.is_dataset ? ' is-dataset' : ''}" data-path="${e.path}">
                <span class="bi-icon">📁</span>
                <span class="bi-name">${e.name}</span>
                ${badge}
            </div>`;
        }
        if (!result.entries.length) {
            html += '<div class="browser-msg">Empty directory</div>';
        }

        entries.innerHTML = html;
        entries.querySelectorAll('.browser-item').forEach(item => {
            item.addEventListener('click', () => browseDir(item.dataset.path));
        });
    } catch (e) {
        entries.innerHTML = `<div class="browser-msg err">Error: ${e.message}</div>`;
    }
}

// ── Utilities ─────────────────────────────────────────────────────────────────

function get(id) {
    return document.getElementById(id)?.value ?? '';
}

function set(id, value) {
    const el = document.getElementById(id);
    if (el) el.value = value;
}

async function api(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status} — ${url}`);
    return res.json();
}

async function apiPost(url, body) {
    const res = await fetch(url, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status} — ${url}`);
    return res.json();
}
