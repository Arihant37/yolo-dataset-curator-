/**
 * CleanCrop - Dataset Quality & Curation Dashboard
 * 1. Loads the precomputed report (neighbors, duplicate pairs, outlier scores,
 *    out-of-fold label predictions and PCA coordinates come from Python, so the
 *    browser stays fast with thousands of boxes)
 * 2. Live threshold tuning + human review (keep / relabel / delete)
 * 3. Review decisions are auto-saved to the local server and can be written
 *    back to the YOLO label files (with automatic backups)
 */

(function () {
    // APP STATE
    let reportData = null;
    let activeTab = 'overview';
    let classNames = [];

    // Review decisions (auto-saved to the server)
    const deletedIds = new Set();
    const relabeledLabels = new Map();          // id -> new class name
    const keptIds = new Set();                  // reviewed and confirmed correct
    const ignoredDuplicates = new Set();        // "idA|idB" pairs marked as not duplicates
    const actionsLog = [];
    let serverAvailable = false;
    let saveTimer = null;

    // Thresholds (sliders)
    let dupThreshold = 0.97;
    let lofThreshold = 1.5;
    let mislabelConf = 0.7;

    // Flag lists derived from the thresholds
    let mislabelList = [];
    let outlierList = [];
    let issueList = [];
    let duplicateList = [];

    // Paging for big lists
    const PAGE = 150;
    const shown = { mislabels: PAGE, outliers: PAGE, issues: PAGE, explorer: 300, duplicates: PAGE, backgrounds: 120 };

    // PCA / Plot State (used by the canvas code)
    let pcaCoordinates = [];
    let map2dCoordinates = [];
    let classColors = {};
    let defaultColor = '#58a6ff';

    const canvas = document.getElementById('embedding-canvas');
    const ctx = canvas.getContext('2d');
    let viewZoom = 1.0;
    let viewPanX = 0.0;
    let viewPanY = 0.0;
    let isDrawing = false;
    let dragStart = { x: 0, y: 0 };
    let dragPanStart = { x: 0, y: 0 };
    let hoveredIndex = -1;
    let selectedIndex = -1;
    let showAnomalyRings = true;

    let is3D = false;
    let rotationX = -0.5;
    let rotationY = 0.6;
    let dragRotationStart = { x: 0, y: 0 };

    // Mislabel review cursor (position in mislabelList)
    let misCursor = -1;
    // Unusual Boxes keyboard cursor (position in outlierList)
    let outCursor = -1;
    // Keyboard / map-selection undo stack: each entry is a list of crop ids decided together
    const reviewUndo = [];
    // "reviewed N / M" per review tab
    const reviewProgress = { mislabels: [0, 0], outliers: [0, 0] };
    // Point colouring: 'class' or 'cluster' (cluster ids come from the analysis)
    let colorBy = 'class';
    // Lasso selection on the 2D map (shift+drag)
    let lassoPts = null;
    const lassoSel = new Set();
    let suppressClick = false;
    let lastDragDist = 0;

    // 3D camera: orbit centre (data space), perspective distance, auto-rotate, pan drag
    let orbitCenter = [0, 0, 0];
    let camDist = 6;            // in data units; set from the data extent
    let dataExtent = 1;
    let autoRotate = false;
    let isPanning = false;
    let projCache = [];          // per item: [sx, sy, z, radius] or null when hidden (what was drawn)
    // Global filters (plot + grids)
    const gFilter = { cls: '', cam: '', period: '', flagged: false, unreviewed: false };
    let clusterFilter = null;    // cluster id or null
    let clusterHulls = null;     // cache: [{cluster, hull:[[x,y]...]}] in map2d data space
    // Pinned point window: navigation list mode
    let popNavMode = 'cluster';  // 'cluster' | 'flagged'
    let popPos = null;           // remembered position after the user drags it

    const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    const pct = v => (v === null || v === undefined) ? 'n/a' : `${(v * 100).toFixed(0)}%`;

    // ==========================================================================
    // INIT
    // ==========================================================================

    window.addEventListener('DOMContentLoaded', async () => {
        if (!window.CURATION_REPORT) {
            showErrorState("No analysis found. Run: python dataset_curator_yolo.py <your dataset folder>");
            return;
        }

        document.querySelectorAll('.menu-item').forEach(item => {
            item.addEventListener('click', () => switchTab(item.getAttribute('data-tab')));
        });
        setupSliders();
        setupCanvas();
        setupActionButtons();
        setupFrameViewer();
        setupKeyboard();
        setupUiExtras();

        await loadSavedState();
        loadDataset(window.CURATION_REPORT);
    });

    async function loadSavedState() {
        try {
            const res = await fetch('/api/state', { cache: 'no-store' });
            if (!res.ok) throw new Error(res.status);
            const st = await res.json();
            serverAvailable = true;
            (st.deleted || []).forEach(id => deletedIds.add(id));
            Object.entries(st.relabeled || {}).forEach(([id, lbl]) => relabeledLabels.set(id, lbl));
            (st.kept || []).forEach(id => keptIds.add(id));
            (st.ignoredDuplicates || []).forEach(k => ignoredDuplicates.add(k));
            (st.log || []).forEach(a => actionsLog.push(a));
            if (st.thresholds) {
                dupThreshold = st.thresholds.dup ?? dupThreshold;
                lofThreshold = st.thresholds.lof ?? lofThreshold;
                mislabelConf = st.thresholds.mis ?? mislabelConf;
            }
            window.__restoredThresholds = !!st.thresholds;
        } catch (e) {
            serverAvailable = false;
        }
        const note = document.getElementById('save-status');
        note.textContent = serverAvailable ? 'Progress auto-saves' : 'Opened as a file: progress is NOT saved and changes cannot be written. Start it with CleanCrop.bat.';
        note.classList.toggle('warn', !serverAvailable);
    }

    function saveState() {
        if (!serverAvailable) return;
        clearTimeout(saveTimer);
        saveTimer = setTimeout(() => {
            fetch('/api/state', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    deleted: [...deletedIds],
                    relabeled: Object.fromEntries(relabeledLabels),
                    kept: [...keptIds],
                    ignoredDuplicates: [...ignoredDuplicates],
                    log: actionsLog,
                    thresholds: { dup: dupThreshold, lof: lofThreshold, mis: mislabelConf },
                }),
            }).catch(() => {
                document.getElementById('save-status').textContent = 'Could not save progress - is the CleanCrop window still open?';
            });
        }, 300);
    }

    // ==========================================================================
    // DATA LOADING & FLAG ENGINE
    // ==========================================================================

    function loadDataset(report) {
        reportData = report;
        classNames = report.class_names && report.class_names.length
            ? report.class_names
            : [...new Set(report.items.map(it => it.label))];

        document.getElementById('dataset-name').textContent = report.dataset_name || 'Dataset';
        document.getElementById('dataset-meta').textContent =
            `${report.items.filter(it => !it.removed).length} boxes · ${report.stats ? report.stats.images : '?'} images · ${(report.model || '').toUpperCase()}`;

        const p = report.params || {};
        if (!window.__restoredThresholds) {
            dupThreshold = p.dup_threshold ?? dupThreshold;
            lofThreshold = p.lof_threshold ?? lofThreshold;
            mislabelConf = p.mislabel_conf ?? mislabelConf;
        }
        const sliderDup = document.getElementById('slider-dup');
        sliderDup.min = Math.min(p.dup_floor ?? 0.9, dupThreshold).toFixed(2);
        sliderDup.value = dupThreshold;
        document.getElementById('slider-lof').value = lofThreshold;
        document.getElementById('slider-mis-conf').value = mislabelConf;
        updateSliderLabels();

        classColors = {};
        const hues = [200, 330, 45, 135, 275, 0, 25, 75, 170, 225, 290, 340];
        classNames.forEach((cls, idx) => {
            classColors[cls] = `hsl(${hues[idx % hues.length]}, 85%, 60%)`;
        });

        const filterClass = document.getElementById('filter-class');
        filterClass.innerHTML = '<option value="all">All Classes</option>' +
            classNames.map(c => `<option value="${esc(c)}">${esc(c)}</option>`).join('');
        const filterSplit = document.getElementById('filter-split');
        const splits = [...new Set(report.items.map(it => it.split))];
        filterSplit.innerHTML = '<option value="all">All Splits</option>' +
            splits.map(s => `<option value="${esc(s)}">${esc(s)}</option>`).join('');
        document.getElementById('exp-relabel-select').innerHTML = classOptions();

        pcaCoordinates = report.items.map(it => it.pca || [0, 0, 0]);
        map2dCoordinates = report.items.map(it => it.map2d || null);
        // 3D camera: orbit around the data centre, perspective distance from the data extent
        const n = pcaCoordinates.length || 1;
        orbitCenter = [0, 1, 2].map(k => pcaCoordinates.reduce((s, c) => s + (c[k] || 0), 0) / n);
        dataExtent = Math.max(1e-6, ...pcaCoordinates.map(c => Math.hypot(c[0] - orbitCenter[0], c[1] - orbitCenter[1], (c[2] || 0) - orbitCenter[2])));
        camDist = dataExtent * 4;
        // camera / day-night per box (newer reports carry them; older ones: parse the file name)
        report.items.forEach(it => {
            const m = /^(.+?)_(nvr-\d+_\d+)_(\d{8})_(\d{6})/.exec(it.image || '');
            if (!it.camera) it.camera = m ? m[2] : 'unknown';
            if (!it.period) { const h = m ? +m[4].slice(0, 2) : null; it.period = h === null ? 'unknown' : (h >= 6 && h < 18 ? 'day' : 'night'); }
        });
        const fillSel = (id, all, vals) => {
            const el = document.getElementById(id);
            if (el) el.innerHTML = `<option value="">${all}</option>` + vals.map(v => `<option value="${esc(v)}">${esc(v)}</option>`).join('');
        };
        fillSel('gf-class', 'All classes', classNames);
        fillSel('gf-camera', 'All cameras', [...new Set(report.items.map(it => it.camera))].sort());
        fillSel('gf-period', 'Day + night', [...new Set(report.items.map(it => it.period))].sort());
        clusterHulls = null;

        recomputeCuration();
        resetCanvasView();
    }

    function classOptions(selected) {
        return classNames.map(c => `<option value="${esc(c)}" ${c === selected ? 'selected' : ''}>${esc(c)}</option>`).join('');
    }

    function recomputeCuration() {
        if (!reportData) return;
        const items = reportData.items;

        items.forEach(item => {
            item.curLabel = relabeledLabels.get(item.id) || item.label;
            item.curDeleted = deletedIds.has(item.id) || !!item.removed;
            item.curKept = keptIds.has(item.id);
            const reviewed = item.curKept || relabeledLabels.has(item.id) || !!item.relabeled_from;
            const active = !item.curDeleted;

            item.is_mislabeled = active && !reviewed && !!item.pred && item.pred !== item.curLabel &&
                (item.pred_conf || 0) >= mislabelConf;
            item.is_outlier = active && !reviewed && item.lof >= lofThreshold;
            item.has_issue = active && !item.curKept && item.issues && item.issues.length > 0;
            item.is_duplicate = false;
        });

        duplicateList = [];
        (reportData.duplicates || []).forEach(([i, j, sim]) => {
            if (sim < dupThreshold) return;
            const a = items[i], b = items[j];
            if (!a || !b || a.curDeleted || b.curDeleted) return;
            const key = `${a.id}|${b.id}`;
            if (ignoredDuplicates.has(key)) return;
            a.is_duplicate = true;
            b.is_duplicate = true;
            duplicateList.push({ pairKey: key, a, b, indexA: i, indexB: j, similarity: sim });
        });
        // Train/val leaks first - they inflate your validation scores
        duplicateList.sort((x, y) => (y.a.split !== y.b.split) - (x.a.split !== x.b.split) || y.similarity - x.similarity);

        mislabelList = items.filter(it => it.is_mislabeled && passesFilter(it)).sort((a, b) => b.pred_conf - a.pred_conf);
        outlierList = items.filter(it => it.is_outlier && passesFilter(it)).sort((a, b) => b.lof - a.lof);
        issueList = items.filter(it => it.has_issue);

        const active = items.filter(it => !it.curDeleted);
        const flagged = active.filter(it => it.is_mislabeled || it.is_outlier || it.has_issue || it.is_duplicate).length;
        reportData.summary = {
            total_samples: active.length,
            duplicate_pairs_count: duplicateList.length,
            outlier_count: outlierList.length,
            mislabel_count: mislabelList.length,
            issue_count: issueList.length,
            clean_count: active.length - flagged,
            health_score: active.length ? ((active.length - flagged) / active.length) * 100 : 100,
        };

        // Review progress: boxes flagged at the current thresholds, reviewed or not
        const decided = it => deletedIds.has(it.id) || keptIds.has(it.id) || relabeledLabels.has(it.id) || !!it.relabeled_from;
        let mR = 0, mT = 0, oR = 0, oT = 0;
        items.forEach(it => {
            if (it.removed) return;
            const d = decided(it);
            const wasMis = !!it.pred && it.pred !== (it.relabeled_from || it.label) && (it.pred_conf || 0) >= mislabelConf;
            if (it.is_mislabeled) mT++;
            else if (wasMis && d) { mT++; mR++; }
            if (it.is_outlier) oT++;
            else if (it.lof >= lofThreshold && d) { oT++; oR++; }
        });
        reviewProgress.mislabels = [mR, mT];
        reviewProgress.outliers = [oR, oT];

        updateUI();
    }

    // ==========================================================================
    // UI UPDATES & TAB ROUTING
    // ==========================================================================

    function updateUI() {
        if (!reportData) return;
        const s = reportData.summary;

        document.getElementById('badge-dup-count').textContent = s.duplicate_pairs_count;
        document.getElementById('badge-out-count').textContent = s.outlier_count;
        document.getElementById('badge-mis-count').textContent = s.mislabel_count;
        document.getElementById('badge-issue-count').textContent = s.issue_count;

        document.getElementById('stat-health-score').textContent = `${s.health_score.toFixed(1)}%`;
        document.getElementById('stat-duplicate-count').textContent = s.duplicate_pairs_count;
        document.getElementById('stat-outlier-count').textContent = s.outlier_count;
        document.getElementById('stat-mislabel-count').textContent = s.mislabel_count;
        document.getElementById('stat-issue-count').textContent = s.issue_count;
        const nBg = (reportData.backgrounds || []).length;
        const nImg = reportData.stats ? reportData.stats.images : 0;
        document.getElementById('stat-bg-count').textContent = nImg ? `${nBg} (${(100 * nBg / nImg).toFixed(1)}%)` : `${nBg}`;
        document.getElementById('badge-bg-count').textContent = nBg;

        renderActiveTab();
        drawCanvas();
        updateExportBar();
        updateTopStatus();
        renderClusterGrid();
        updateFsSide();
    }

    const TAB_TITLES = {
        overview: ["Overview", "Dataset health at a glance. Start with Suspected Mislabels, then Unusual Boxes."],
        mislabels: ["Suspected Mislabels", "A classifier that never saw these boxes disagrees with their label. Keys: A/K = label is correct, X = delete box, R = accept suggestion, 1-9 = relabel to class N, arrows = move, U = undo, F = full frame."],
        outliers: ["Unusual Boxes", "Boxes that look unlike the rest of their class: wrong object, bad crop, background, other species... Keys: A approve, X remove, 1-9 relabel, arrows move, U undo."],
        issues: ["Broken / Double Boxes", "Boxes annotated twice, conflicting labels on the same object, boxes outside the image or unknown class ids."],
        duplicates: ["Near-Duplicates", "Almost identical boxes from different images. Train↔val pairs are listed first: they inflate validation scores."],
        explorer: ["All Boxes", "Search, filter and review any box. Click a thumbnail to inspect it."],
        backgrounds: ["Backgrounds", "Images with no boxes (empty or missing label file). They teach the model what NOT to detect. Click one to see it full size."],
        distribution: ["Distribution", "Where the data comes from (cameras, devices, day/night, box sizes) and whether it is ready for a golden test set."],
        ready: ["Training ready?", "A last check before training: what is still unreviewed, backgrounds, golden test set. Then export a clean copy of the dataset."],
    };

    function switchTab(tabId) {
        activeTab = tabId;
        document.querySelectorAll('.menu-item').forEach(item => {
            item.classList.toggle('active', item.getAttribute('data-tab') === tabId);
        });
        document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));
        document.getElementById(`tab-${tabId}`).classList.add('active');
        document.getElementById('current-tab-title').textContent = TAB_TITLES[tabId][0];
        document.getElementById('current-tab-desc').textContent = TAB_TITLES[tabId][1];
        renderActiveTab();
        if (tabId === 'overview') resetCanvasView();
        if (tabId === 'mislabels' && mislabelList.length && misCursor < 0) selectMislabelItem(0);
        if (tabId === 'outliers' && outlierList.length && outCursor < 0) selectOutlierCard(0);
        if (tabId !== 'overview') closePointPopup();
    }

    function renderActiveTab() {
        if (!reportData) return;
        switch (activeTab) {
            case 'overview': renderOverviewTab(); break;
            case 'duplicates': renderDuplicatesTab(); break;
            case 'outliers': renderCardGrid('outliers'); break;
            case 'issues': renderCardGrid('issues'); break;
            case 'mislabels': renderMislabelsTab(); break;
            case 'explorer': renderExplorerTab(); break;
            case 'backgrounds': renderBackgroundsTab(); break;
            case 'distribution': renderDistributionTab(); break;
            case 'ready': renderReadyTab(); break;
        }
    }

    function thumb(item, extra = '') {
        return `<img src="${esc(item.path)}" alt="${esc(item.id)}" loading="lazy" data-frame="${esc(item.id)}" title="Click to see the full frame" ${extra}>`;
    }

    function pill(label) {
        const c = classColors[label] || defaultColor;
        return `<span class="class-pill" style="background-color:${c}22;color:${c};border:1px solid ${c}55">${esc(label)}</span>`;
    }

    function moreButton(key, total) {
        return total > shown[key]
            ? `<button class="btn btn-secondary btn-small load-more" data-more="${key}">Show more (${total - shown[key]} left)</button>`
            : '';
    }

    function wireMoreButtons(container) {
        container.querySelectorAll('[data-more]').forEach(btn => {
            btn.addEventListener('click', () => {
                shown[btn.dataset.more] += btn.dataset.more === 'explorer' ? 300 : PAGE;
                renderActiveTab();
            });
        });
    }

    // --- 1. OVERVIEW ---
    function renderOverviewTab() {
        const items = reportData.items;
        const classCounts = {};
        let activeCount = 0;
        items.forEach(item => {
            if (item.curDeleted) return;
            classCounts[item.curLabel] = (classCounts[item.curLabel] || 0) + 1;
            activeCount++;
        });

        document.getElementById('class-distribution-list').innerHTML = Object.entries(classCounts)
            .sort((a, b) => b[1] - a[1])
            .map(([cls, count]) => {
                const p = activeCount ? (count / activeCount) * 100 : 0;
                return `<div class="bar-row">
                    <div class="bar-label"><span class="bar-class-name">${esc(cls)}</span>
                    <span class="bar-count-val">${count} boxes (${p.toFixed(1)}%)</span></div>
                    <div class="bar-outer"><div class="bar-inner" style="width:${Math.max(p, 0.5)}%;background-color:${classColors[cls] || defaultColor};"></div></div>
                </div>`;
            }).join('');

        renderPlotLegend();

        // Dataset facts from the scan
        const st = reportData.stats || {};
        const splitTxt = st.splits ? Object.entries(st.splits).map(([k, v]) => `${esc(k)}: ${v.images} images / ${v.boxes} boxes`).join('<br>') : '';
        const problems = reportData.problems || [];
        document.getElementById('dataset-facts').innerHTML = `
            <div class="fact-row"><span>Images</span><b>${st.images ?? '?'}</b></div>
            <div class="fact-row"><span>Images with no boxes</span><b>${st.background_images ?? '?'}${st.images ? ` (${(100 * (st.background_images || 0) / st.images).toFixed(1)}%)` : ''}</b></div>
            <div class="fact-row"><span>Boxes</span><b>${activeCount}</b></div>
            <div class="fact-row"><span>Unreadable label lines</span><b class="${problems.length ? 'bad' : ''}">${problems.length}</b></div>
            <div class="fact-split">${splitTxt}</div>`;

        const totalFlags = mislabelList.length + outlierList.length + issueList.length + duplicateList.length;
        const done = deletedIds.size + relabeledLabels.size + keptIds.size;
        const progressPct = totalFlags + done > 0 ? (done / (totalFlags + done)) * 100 : 100;
        document.getElementById('curation-progress').style.width = `${progressPct}%`;
        document.getElementById('curation-progress-text').textContent = `${done} reviewed · ${totalFlags} still flagged`;
    }

    // --- 2. DUPLICATES ---
    function renderDuplicatesTab() {
        const listContainer = document.getElementById('dup-list-container');
        document.getElementById('dup-pairs-title').textContent = `Duplicate Pairs (${duplicateList.length})`;

        if (duplicateList.length === 0) {
            listContainer.innerHTML = `<div class="empty-state-msg" style="height:100px;"><i class="fa-solid fa-copy" style="font-size:24px;"></i><p>No near-duplicates at this similarity threshold.</p></div>`;
            document.getElementById('dup-empty-state').classList.remove('hidden');
            document.getElementById('dup-inspector-content').classList.add('hidden');
            return;
        }

        listContainer.innerHTML = duplicateList.slice(0, shown.duplicates).map((pair, idx) => {
            const leak = pair.a.split !== pair.b.split;
            return `<div class="item-list-card" data-pair-idx="${idx}">
                <div class="list-item-meta">
                    <div class="item-thumbnail"><img src="${esc(pair.a.path)}" loading="lazy"></div>
                    <div class="item-thumbnail"><img src="${esc(pair.b.path)}" loading="lazy"></div>
                    <div class="item-text-info">
                        <span class="item-text-title">${esc(pair.a.id)}<br>${esc(pair.b.id)}</span>
                        <span class="item-text-subtitle">${leak ? `<b class="leak">${esc(pair.a.split)} ↔ ${esc(pair.b.split)} leak</b>` : esc(pair.a.split)} · ${esc(pair.a.curLabel)}</span>
                    </div>
                </div>
                <div class="item-list-stat dup">${(pair.similarity * 100).toFixed(1)}%</div>
            </div>`;
        }).join('') + moreButton('duplicates', duplicateList.length);

        listContainer.querySelectorAll('.item-list-card').forEach(card => {
            card.addEventListener('click', () => selectDuplicatePair(parseInt(card.dataset.pairIdx)));
        });
        wireMoreButtons(listContainer);
        document.getElementById('dup-empty-state').classList.remove('hidden');
        document.getElementById('dup-inspector-content').classList.add('hidden');
    }

    function selectDuplicatePair(idx) {
        const pair = duplicateList[idx];
        if (!pair) return;
        document.querySelectorAll('#dup-list-container .item-list-card').forEach((card, cidx) => {
            card.classList.toggle('active', cidx === idx);
        });
        document.getElementById('dup-empty-state').classList.add('hidden');
        const inspector = document.getElementById('dup-inspector-content');
        inspector.classList.remove('hidden');

        document.getElementById('dup-similarity-value').textContent = `${(pair.similarity * 100).toFixed(1)}%`;
        document.getElementById('dup-id-a').textContent = `${pair.a.id} (${pair.a.split})`;
        document.getElementById('dup-label-a').textContent = pair.a.curLabel;
        document.getElementById('dup-img-a').src = pair.a.path;
        document.getElementById('dup-img-a').dataset.frame = pair.a.id;
        document.getElementById('dup-id-b').textContent = `${pair.b.id} (${pair.b.split})`;
        document.getElementById('dup-label-b').textContent = pair.b.curLabel;
        document.getElementById('dup-img-b').src = pair.b.path;
        document.getElementById('dup-img-b').dataset.frame = pair.b.id;

        inspector.querySelector('.btn-delete-crop[data-target="a"]').onclick = () => excludeCrop(pair.a.id, `Deleted duplicate of ${pair.b.id}`);
        inspector.querySelector('.btn-delete-crop[data-target="b"]').onclick = () => excludeCrop(pair.b.id, `Deleted duplicate of ${pair.a.id}`);
        document.getElementById('btn-dup-keep-both').onclick = () => {
            ignoredDuplicates.add(pair.pairKey);
            registerAction({ type: 'ignore_duplicate', id: pair.pairKey, description: `Kept both ${pair.a.id} and ${pair.b.id}.` });
            recomputeCuration();
        };
    }

    // --- 3/4. OUTLIERS + BOX ISSUES (shared card grid) ---
    function renderCardGrid(kind) {
        const list = kind === 'outliers' ? outlierList : issueList;
        const grid = document.getElementById(`${kind}-grid-container`);
        document.getElementById(`${kind}-stats`).textContent = `${list.length} boxes to review` +
            (kind === 'outliers' ? ` (most unusual first) · reviewed ${reviewProgress.outliers[0]} / ${reviewProgress.outliers[1]}` +
                ' · keys: A approve, X remove, 1-9 relabel, arrows move, U undo' : '');

        if (list.length === 0) {
            grid.innerHTML = `<div class="empty-state-msg" style="grid-column:1/-1;height:200px;">
                <i class="fa-solid fa-circle-check" style="font-size:32px;color:var(--color-clean)"></i>
                <h3>Nothing left to review here</h3>
                ${kind === 'outliers' ? '<p>Lower the "Unusual box" slider to see more borderline boxes.</p>' : ''}</div>`;
            return;
        }

        grid.innerHTML = list.slice(0, shown[kind]).map((it, idx) => {
            const reason = kind === 'outliers'
                ? `Unusual for '${esc(it.curLabel)}' (score ${it.lof.toFixed(2)})` + (it.pred && it.pred !== it.curLabel ? `<br>Model thinks: <b>${esc(it.pred)}</b> (${pct(it.pred_conf)})` : '')
                : it.issues.map(esc).join('<br>');
            return `<div class="outlier-card${kind === 'outliers' && idx === outCursor ? ' kb-active' : ''}" data-idx="${idx}">
                <div class="outlier-badge ${kind === 'issues' ? 'issue' : ''}">${kind === 'issues' ? 'BOX ISSUE' : 'UNUSUAL'}</div>
                <div class="crop-image-wrapper">${thumb(it)}</div>
                <span class="crop-id" title="${esc(it.image)}">${esc(it.id)} · ${esc(it.split)}</span>
                ${pill(it.curLabel)}
                <p class="outlier-reason">${reason}</p>
                <div class="action-buttons-group" style="width:100%;">
                    <button class="btn btn-secondary btn-block btn-small" data-act="keep" data-id="${esc(it.id)}"><i class="fa-solid fa-check"></i> Correct, keep</button>
                    <select class="card-relabel" data-id="${esc(it.id)}"><option value="">Relabel to…</option>${classNames.filter(c => c !== it.curLabel).map(c => `<option value="${esc(c)}">${esc(c)}</option>`).join('')}</select>
                    <button class="btn btn-danger btn-block btn-small" data-act="delete" data-id="${esc(it.id)}"><i class="fa-solid fa-trash"></i> Delete box</button>
                </div>
            </div>`;
        }).join('') + `<div style="grid-column:1/-1">${moreButton(kind, list.length)}</div>`;

        grid.querySelectorAll('[data-act="keep"]').forEach(btn => btn.addEventListener('click', () => approveCrop(btn.dataset.id, `Confirmed ${kind === 'issues' ? 'box issue' : 'unusual box'} is fine.`)));
        grid.querySelectorAll('[data-act="delete"]').forEach(btn => btn.addEventListener('click', () => excludeCrop(btn.dataset.id, `Deleted ${kind === 'issues' ? 'broken/double' : 'unusual'} box.`)));
        grid.querySelectorAll('.card-relabel').forEach(sel => sel.addEventListener('change', () => {
            const it = itemById(sel.dataset.id);
            if (sel.value && it) relabelCrop(it.id, it.curLabel, sel.value);
        }));
        wireMoreButtons(grid);
        if (kind === 'outliers') {
            grid.querySelectorAll('.outlier-card').forEach(card => card.addEventListener('click', e => {
                if (!e.target.closest('button, select, img')) selectOutlierCard(parseInt(card.dataset.idx));
            }));
            if (outCursor >= 0) selectOutlierCard(Math.min(outCursor, Math.min(list.length, shown.outliers) - 1));
        }
    }

    // --- 5. MISLABELS ---
    function renderMislabelsTab() {
        const listContainer = document.getElementById('mis-list-container');
        document.getElementById('mis-items-title').textContent = `Suspected Wrong Labels (${mislabelList.length})`;
        document.getElementById('mis-progress').textContent = `reviewed ${reviewProgress.mislabels[0]} / ${reviewProgress.mislabels[1]}`;

        if (mislabelList.length === 0) {
            listContainer.innerHTML = `<div class="empty-state-msg" style="height:100px;"><i class="fa-solid fa-circle-check" style="font-size:24px;color:var(--color-clean)"></i><p>No suspected mislabels at this confidence. Lower the slider to see borderline cases.</p></div>`;
            document.getElementById('mis-empty-state').classList.remove('hidden');
            document.getElementById('mis-inspector-content').classList.add('hidden');
            misCursor = -1;
            return;
        }

        listContainer.innerHTML = mislabelList.slice(0, shown.mislabels).map((it, idx) => `
            <div class="item-list-card ${idx === misCursor ? 'active' : ''}" data-mis-idx="${idx}">
                <div class="list-item-meta">
                    <div class="item-thumbnail"><img src="${esc(it.path)}" loading="lazy"></div>
                    <div class="item-text-info">
                        <span class="item-text-title">${esc(it.id)}</span>
                        <span class="item-text-subtitle">Labeled <b style="color:var(--color-outlier);">${esc(it.curLabel)}</b> → looks like <b style="color:var(--color-clean);">${esc(it.pred)}</b></span>
                    </div>
                </div>
                <div class="item-list-stat mis">${pct(it.pred_conf)}</div>
            </div>`).join('') + moreButton('mislabels', mislabelList.length);

        listContainer.querySelectorAll('.item-list-card').forEach(card => {
            card.addEventListener('click', () => selectMislabelItem(parseInt(card.dataset.misIdx)));
        });
        wireMoreButtons(listContainer);

        if (misCursor >= 0) {
            selectMislabelItem(Math.min(misCursor, mislabelList.length - 1));
        } else {
            document.getElementById('mis-empty-state').classList.remove('hidden');
            document.getElementById('mis-inspector-content').classList.add('hidden');
        }
    }

    function selectMislabelItem(idx) {
        const it = mislabelList[idx];
        if (!it) return;
        misCursor = idx;
        selectedIndex = it.index;

        document.querySelectorAll('#mis-list-container .item-list-card').forEach(card => {
            const active = parseInt(card.dataset.misIdx) === idx;
            card.classList.toggle('active', active);
            if (active) card.scrollIntoView({ block: 'nearest' });
        });
        document.getElementById('mis-empty-state').classList.add('hidden');
        document.getElementById('mis-inspector-content').classList.remove('hidden');

        document.getElementById('mis-label-current').textContent = it.curLabel;
        document.getElementById('mis-label-suggested').textContent = it.pred;
        const img = document.getElementById('mis-img-main');
        img.src = it.path;
        img.dataset.frame = it.id;
        document.getElementById('mis-id-main').textContent = `${it.id} · ${it.split}`;

        const nbrs = (it.nbrs || []).map(([j, sim]) => ({ item: reportData.items[j], sim })).filter(n => n.item && !n.item.curDeleted);
        const agree = nbrs.filter(n => n.item.curLabel === it.pred).length;
        document.getElementById('mis-reason-text').innerHTML =
            `Classifier: <b>${pct(it.pred_conf)}</b> sure this is '<b>${esc(it.pred)}</b>' (only ${pct(it.given_conf)} for '${esc(it.curLabel)}').<br>` +
            `${agree} of ${nbrs.length} most similar boxes are labeled '${esc(it.pred)}'.`;
        document.getElementById('btn-relabel-text').textContent = it.pred;

        document.getElementById('mis-neighbors-list').innerHTML = nbrs.slice(0, 8).map(n => `
            <div class="neighbor-card">
                <div class="neighbor-meta">
                    <div class="neighbor-thumb">${thumb(n.item)}</div>
                    <span class="neighbor-id">${esc(n.item.id)}</span>
                    ${pill(n.item.curLabel)}
                </div>
                <span class="neighbor-sim">${(n.sim * 100).toFixed(1)}%</span>
            </div>`).join('');

        document.getElementById('mis-relabel-other').innerHTML = '<option value="">Other class…</option>' +
            classNames.filter(c => c !== it.curLabel && c !== it.pred).map(c => `<option value="${esc(c)}">${esc(c)}</option>`).join('');
        document.getElementById('mis-relabel-other').onchange = e => { if (e.target.value) relabelCrop(it.id, it.curLabel, e.target.value); };

        document.getElementById('btn-mis-relabel').onclick = () => relabelCrop(it.id, it.curLabel, it.pred);
        document.getElementById('btn-mis-keep').onclick = () => approveCrop(it.id, `Confirmed label '${it.curLabel}'.`);
        document.getElementById('btn-mis-delete').onclick = () => excludeCrop(it.id, 'Deleted suspected mislabeled box.');
        document.getElementById('btn-mis-frame').onclick = () => openFrame(it.id);
    }

    // --- 6. EXPLORER ---
    function renderExplorerTab() {
        const grid = document.getElementById('explorer-crops-grid');
        const searchVal = document.getElementById('explorer-search').value.toLowerCase().trim();
        const selectedClass = document.getElementById('filter-class').value;
        const selectedSplit = document.getElementById('filter-split').value;
        const selectedFlag = document.getElementById('filter-anomaly').value;
        const sortBy = document.getElementById('sort-by').value;

        let filtered = reportData.items.filter(it => !it.removed && passesFilter(it));
        if (searchVal) filtered = filtered.filter(it => it.id.toLowerCase().includes(searchVal) || it.image.toLowerCase().includes(searchVal));
        if (selectedClass !== 'all') filtered = filtered.filter(it => it.curLabel === selectedClass);
        if (selectedSplit !== 'all') filtered = filtered.filter(it => it.split === selectedSplit);
        const flagFilters = {
            clean: it => !it.curDeleted && !it.is_duplicate && !it.is_outlier && !it.is_mislabeled && !it.has_issue,
            flagged: it => !it.curDeleted && (it.is_duplicate || it.is_outlier || it.is_mislabeled || it.has_issue),
            duplicate: it => !it.curDeleted && it.is_duplicate,
            outlier: it => !it.curDeleted && it.is_outlier,
            mislabel: it => !it.curDeleted && it.is_mislabeled,
            issue: it => !it.curDeleted && it.has_issue,
            resolved: it => it.curDeleted || it.curKept || relabeledLabels.has(it.id),
        };
        if (flagFilters[selectedFlag]) filtered = filtered.filter(flagFilters[selectedFlag]);

        const conf = it => it.given_conf ?? 1;
        if (sortBy === 'lof-desc') filtered.sort((a, b) => b.lof - a.lof);
        else if (sortBy === 'conf-asc') filtered.sort((a, b) => conf(a) - conf(b));
        else if (sortBy === 'image') filtered.sort((a, b) => a.image.localeCompare(b.image) || a.line - b.line);

        document.getElementById('explorer-grid-title').textContent = `Boxes (${filtered.length})`;
        if (filtered.length === 0) {
            grid.innerHTML = `<p style="grid-column:1/-1;color:var(--text-secondary);text-align:center;padding:20px;">No boxes match the current filters.</p>`;
            return;
        }

        grid.innerHTML = filtered.slice(0, shown.explorer).map(it => {
            let ind = '';
            if (it.curDeleted) ind = '';
            else if (it.curKept || relabeledLabels.has(it.id)) ind = 'res';
            else if (it.is_mislabeled) ind = 'mis';
            else if (it.is_outlier || it.has_issue) ind = 'out';
            else if (it.is_duplicate) ind = 'dup';
            return `<div class="explorer-crop-thumb ${it.curDeleted ? 'resolved' : ''} ${selectedIndex === it.index ? 'active' : ''}" data-idx="${it.index}" title="${esc(it.id)} (${esc(it.curLabel)})">
                <img src="${esc(it.path)}" alt="" loading="lazy">
                ${ind ? `<div class="thumb-indicator ${ind}"></div>` : ''}
            </div>`;
        }).join('') + `<div style="grid-column:1/-1">${moreButton('explorer', filtered.length)}</div>`;

        grid.querySelectorAll('.explorer-crop-thumb').forEach(t => {
            t.addEventListener('click', () => selectExplorerItem(parseInt(t.dataset.idx)));
        });
        wireMoreButtons(grid);

        if (selectedIndex >= 0) showExplorerDetail(selectedIndex);
        else {
            document.getElementById('exp-empty-state').classList.remove('hidden');
            document.getElementById('exp-inspector-content').classList.add('hidden');
        }
    }

    function selectExplorerItem(idx) {
        selectedIndex = idx;
        document.querySelectorAll('.explorer-crop-thumb').forEach(t => {
            t.classList.toggle('active', parseInt(t.dataset.idx) === idx);
        });
        showExplorerDetail(idx);
        drawCanvas();
    }

    function showExplorerDetail(idx) {
        const item = reportData.items[idx];
        if (!item) return;
        document.getElementById('exp-empty-state').classList.add('hidden');
        document.getElementById('exp-inspector-content').classList.remove('hidden');

        const img = document.getElementById('exp-img');
        img.src = item.path;
        img.dataset.frame = item.id;
        document.getElementById('exp-id').textContent = item.id;
        document.getElementById('exp-image').textContent = `${item.split}/${item.image} (line ${item.line + 1})`;

        const labelPill = document.getElementById('exp-label');
        const color = classColors[item.curLabel] || defaultColor;
        labelPill.textContent = item.curLabel;
        labelPill.style.backgroundColor = color + "22";
        labelPill.style.color = color;
        labelPill.style.border = `1px solid ${color}55`;

        document.getElementById('exp-model-says').textContent = item.pred ? `${item.pred} (${pct(item.pred_conf)})` : 'n/a';
        document.getElementById('exp-sim-lof').textContent = item.lof ? item.lof.toFixed(2) : 'n/a';

        const glow = document.getElementById('exp-glow-border');
        glow.className = "crop-image-wrapper border-glow";
        if (item.is_mislabeled) glow.classList.add('mis');
        else if (item.is_outlier || item.has_issue) glow.classList.add('out');
        else if (item.is_duplicate) glow.classList.add('dup');

        const badges = [];
        if (item.curDeleted) badges.push(`<span class="exp-badge out">Will be deleted</span>`);
        else if (relabeledLabels.has(item.id)) badges.push(`<span class="exp-badge res">Relabeled from ${esc(item.label)}</span>`);
        else if (item.curKept) badges.push(`<span class="exp-badge res">Confirmed correct</span>`);
        else {
            if (item.is_mislabeled) badges.push(`<span class="exp-badge mis">Suspected mislabel</span>`);
            if (item.is_outlier) badges.push(`<span class="exp-badge out">Unusual</span>`);
            if (item.has_issue) badges.push(`<span class="exp-badge out">${esc(item.issues.join('; '))}</span>`);
            if (item.is_duplicate) badges.push(`<span class="exp-badge dup">Duplicate</span>`);
            if (!badges.length) badges.push(`<span class="exp-badge clean">Clean</span>`);
        }
        document.getElementById('exp-status-badges').innerHTML = badges.join('');

        const btnKeep = document.getElementById('btn-exp-keep');
        const btnRelabel = document.getElementById('btn-exp-relabel');
        const btnDelete = document.getElementById('btn-exp-delete');
        const changed = item.curDeleted || relabeledLabels.has(item.id) || item.curKept;
        if (changed && !item.removed) {
            btnKeep.innerHTML = '<i class="fa-solid fa-rotate-left"></i> Undo my change';
            btnKeep.disabled = false;
            btnKeep.onclick = () => undoCrop(item.id);
        } else {
            btnKeep.innerHTML = '<i class="fa-solid fa-circle-check"></i> Label is correct';
            btnKeep.disabled = !!item.removed;
            btnKeep.onclick = () => approveCrop(item.id, "Confirmed in explorer.");
        }
        btnRelabel.disabled = item.curDeleted;
        btnDelete.disabled = item.curDeleted;
        btnDelete.onclick = () => excludeCrop(item.id, "Deleted in explorer.");
        document.getElementById('relabel-dropdown-container').classList.add('hidden');
        btnRelabel.onclick = () => {
            document.getElementById('relabel-dropdown-container').classList.toggle('hidden');
            document.getElementById('exp-relabel-select').innerHTML = classOptions(item.curLabel);
        };
    }

    // --- 7. BACKGROUNDS ---
    function bgUrl(bg, w) {
        return `/api/bgimage?path=${encodeURIComponent(`${bg.split}/${bg.image}`)}${w ? `&w=${w}` : ''}`;
    }

    function renderBackgroundsTab() {
        const bgs = reportData.backgrounds || [];
        const st = reportData.stats || {};
        document.getElementById('bg-stats').textContent = bgs.length
            ? `${bgs.length} background images (${st.images ? (100 * bgs.length / st.images).toFixed(1) : '?'}% of ${st.images ?? '?'} images). A healthy share is roughly 10-35%.`
            : 'No background images: every image has at least one box. Consider adding 10-35% empty frames to reduce false positives.';
        const grid = document.getElementById('bg-grid');
        if (!serverAvailable) {
            grid.innerHTML = `<p style="grid-column:1/-1;color:var(--text-secondary)">Background thumbnails need the CleanCrop server (start it with CleanCrop.bat).</p>`;
            return;
        }
        grid.innerHTML = bgs.slice(0, shown.backgrounds).map((bg, i) => `<div class="bg-card">
                <img src="${bgUrl(bg, 400)}" loading="lazy" data-bg="${i}" alt="">
                <div>${esc(bg.split)} · ${esc(bg.image)}${bg.camera && bg.camera !== 'unknown' ? ` · ${esc(bg.camera)}` : ''}${bg.period ? ` · ${esc(bg.period)}` : ''}${bg.missing_label ? ' · <b>no label file</b>' : ''}</div>
            </div>`).join('') + `<div style="grid-column:1/-1">${moreButton('backgrounds', bgs.length)}</div>`;
        grid.querySelectorAll('img[data-bg]').forEach(img => img.addEventListener('click', () => {
            const bg = bgs[+img.dataset.bg];
            document.getElementById('frame-img').src = bgUrl(bg);
            document.getElementById('frame-caption').textContent = `${bg.split}/${bg.image} — background image (no boxes). Click or press Esc to close.`;
            document.getElementById('frame-modal').classList.remove('hidden');
        }));
        wireMoreButtons(grid);
    }

    // --- 8. DISTRIBUTION ---
    function barsHtml(rows, color) {
        const max = Math.max(1, ...rows.map(r => r[1]));
        return rows.map(([name, val, note]) => `<div class="bar-row">
                <div class="bar-label"><span class="bar-class-name">${esc(name)}</span><span class="bar-count-val">${val}${note ? ` · ${esc(note)}` : ''}</span></div>
                <div class="bar-outer"><div class="bar-inner" style="width:${Math.max(100 * val / max, 0.5)}%;background-color:${color || defaultColor};"></div></div>
            </div>`).join('');
    }

    function distCard(title, icon, body) {
        return `<div class="card"><div class="card-header"><div class="card-title"><i class="fa-solid ${icon}"></i> ${esc(title)}</div></div>
            <div class="card-body"><div class="bar-chart-list">${body}</div></div></div>`;
    }

    function renderDistributionTab() {
        const d = reportData.distribution;
        const list = document.getElementById('readiness-list');
        const grid = document.getElementById('dist-grid');
        if (!d) {
            list.innerHTML = '<p class="meter-desc">This report was made by an older CleanCrop version. Run the analysis again to see the distribution.</p>';
            grid.innerHTML = '';
            return;
        }
        // The review check is live: it uses the current decisions instead of the analysis-time count
        const unresolved = mislabelList.length + outlierList.length;
        const checks = (d.readiness || []).map(c => c.check.startsWith('Mislabels') ? {
            ...c, status: unresolved ? 'warn' : 'pass',
            reason: unresolved ? `${mislabelList.length} suspected mislabels + ${outlierList.length} unusual boxes not reviewed yet` : 'all flagged boxes reviewed',
        } : c);
        list.innerHTML = checks.map(c => `<div class="readiness-row"><span class="st ${c.status}">${c.status.toUpperCase()}</span>
            <div><div>${esc(c.check)}</div><div class="why">${esc(c.reason)}</div></div></div>`).join('') +
            (reportData.golden_test ? `<p class="meter-desc">Golden test list: <b>${reportData.golden_test.images}</b> images / ${reportData.golden_test.boxes} boxes in <code>${esc(reportData.golden_test.file)}</code></p>`
                : '<p class="meter-desc">Create a golden test list with <code>--golden-split 0.15</code> (stratified by camera × class × day/night, diverse images, near-duplicates kept together).</p>');

        const pc = Object.entries(d.per_class || {});
        const classBoxes = barsHtml(pc.map(([c, v]) => [c, v.boxes, `${v.images} images`]), null);
        const classColored = pc.map(([c, v]) => barsHtml([[c, v.boxes, `${v.images} images`]], classColors[c])).join('');
        const dayNight = pc.map(([c, v]) => `<tr><td>${esc(c)}</td><td>${v.day}</td><td>${v.night}</td><td>${v.unknown || 0}</td></tr>`).join('');
        const sizes = pc.map(([c, v]) => `<tr><td>${esc(c)}</td><td>${v.small}</td><td>${v.medium}</td><td>${v.large}</td><td>${v.diversity ?? 'n/a'}</td></tr>`).join('');
        const rules = d.rules || {};
        grid.innerHTML = [
            distCard('Boxes per class', 'fa-tags', classColored || classBoxes),
            distCard('Images per camera', 'fa-video', barsHtml(Object.entries(d.per_camera || {}).map(([k, v]) => [k, v.images, `${v.boxes} boxes`]))),
            distCard('Images per source device', 'fa-microchip', barsHtml(Object.entries(d.per_device || {}).map(([k, v]) => [k, v.images, `${v.boxes} boxes`]))),
            distCard('Day vs night (images)', 'fa-circle-half-stroke', barsHtml(Object.entries(d.per_period || {}).map(([k, v]) => [k, v.images, `${v.boxes} boxes`])) +
                `<table class="dist-table"><tr><th>Boxes per class</th><th>day</th><th>night</th><th>unknown</th></tr>${dayNight}</table>
                 <p class="meter-desc">Day = ${rules.day_hours ? `${rules.day_hours[0]}:00-${rules.day_hours[1]}:00` : '06-18'} from the file-name time; without a time, mean brightness &lt; ${rules.night_brightness ?? 70} = night.</p>`),
            distCard('Box size (longer side)', 'fa-up-right-and-down-left-from-center', barsHtml(Object.entries(d.box_sizes || {}).map(([k, v]) => [k, v])) +
                `<table class="dist-table"><tr><th>Class</th><th>small</th><th>medium</th><th>large</th><th>diversity</th></tr>${sizes}</table>
                 <p class="meter-desc">Small &lt; ${rules.small_px ?? 32}px, large &gt; ${rules.large_px ?? 96}px. Diversity = mean cosine distance between the boxes of a class (higher = more varied; near 0 = near-copies).</p>`),
            distCard('Backgrounds', 'fa-image', barsHtml([['with boxes', d.images - d.backgrounds], ['backgrounds', d.backgrounds, `${d.background_pct}%`]])),
        ].join('');
    }

    function itemById(id) {
        return reportData.items.find(it => it.id === id);
    }

    // ==========================================================================
    // REVIEW ACTIONS
    // ==========================================================================

    // Snapshot of the decisions for some boxes, so a toast "Undo" can put them back exactly
    function takeSnap(ids) {
        return ids.map(id => ({
            id, del: deletedIds.has(id), kept: keptIds.has(id),
            rel: relabeledLabels.has(id) ? relabeledLabels.get(id) : null,
            log: actionsLog.find(a => a.id === id) || null,
        }));
    }

    function restoreSnap(snap) {
        snap.forEach(s => {
            if (s.del) deletedIds.add(s.id); else deletedIds.delete(s.id);
            if (s.kept) keptIds.add(s.id); else keptIds.delete(s.id);
            if (s.rel !== null) relabeledLabels.set(s.id, s.rel); else relabeledLabels.delete(s.id);
            for (let i = actionsLog.length - 1; i >= 0; i--) if (actionsLog[i].id === s.id) actionsLog.splice(i, 1);
            if (s.log) actionsLog.push(s.log);
        });
        saveState();
        recomputeCuration();
        if (pointPopupIdx >= 0) refreshPointPopup();
    }

    function excludeCrop(id, description = "") {
        const snap = takeSnap([id]);
        deletedIds.add(id);
        keptIds.delete(id);
        registerAction({ type: 'delete', id, description: description || `Deleted ${id}.` });
        recomputeCuration();
        showToast('Box marked for removal.', snap);
    }

    function approveCrop(id, description = "") {
        const snap = takeSnap([id]);
        deletedIds.delete(id);
        keptIds.add(id);
        registerAction({ type: 'keep', id, description: description || `Confirmed ${id}.` });
        recomputeCuration();
        showToast('Box approved.', snap);
    }

    function relabelCrop(id, oldLabel, newLabel) {
        const snap = takeSnap([id]);
        const item = itemById(id);
        if (item && newLabel === item.label) relabeledLabels.delete(id);
        else relabeledLabels.set(id, newLabel);
        registerAction({ type: 'relabel', id, old_label: oldLabel, new_label: newLabel, description: `Relabeled ${id}: '${oldLabel}' → '${newLabel}'.` });
        recomputeCuration();
        showToast(`Relabeled ${oldLabel} → ${newLabel}.`, snap);
    }

    function undoCrop(id) {
        const snap = takeSnap([id]);
        deletedIds.delete(id);
        relabeledLabels.delete(id);
        keptIds.delete(id);
        for (let i = actionsLog.length - 1; i >= 0; i--) {
            if (actionsLog[i].id === id) actionsLog.splice(i, 1);
        }
        saveState();
        recomputeCuration();
        showToast('Decision undone.', snap, 'Redo');
    }

    let toastTimer = null;
    function showToast(msg, snap, undoText = 'Undo') {
        const t = document.getElementById('toast');
        if (!t) return;
        t.innerHTML = `<span>${esc(msg)}</span>${snap ? `<button class="btn btn-small" id="toast-undo">${esc(undoText)}</button>` : ''}<button class="toast-x" title="Close">×</button>`;
        const host = document.fullscreenElement || document.body;
        if (t.parentElement !== host) host.appendChild(t);
        t.classList.remove('hidden');
        if (snap) t.querySelector('#toast-undo').onclick = () => { t.classList.add('hidden'); restoreSnap(snap); };
        t.querySelector('.toast-x').onclick = () => t.classList.add('hidden');
        clearTimeout(toastTimer);
        toastTimer = setTimeout(() => t.classList.add('hidden'), 6000);
    }

    function registerAction(act) {
        // One entry per crop: the latest decision wins
        const existingIdx = actionsLog.findIndex(a => a.id === act.id);
        if (existingIdx >= 0) actionsLog.splice(existingIdx, 1);
        act.time = new Date().toISOString();
        actionsLog.push(act);
        saveState();
    }

    function updateExportBar() {
        const bar = document.querySelector('.export-floating-bar');
        const pending = deletedIds.size + relabeledLabels.size;
        if (actionsLog.length > 0) {
            bar.style.display = 'block';
            document.getElementById('export-actions-summary').textContent =
                `${deletedIds.size} to delete · ${relabeledLabels.size} to relabel · ${keptIds.size} confirmed`;
            document.getElementById('btn-commit-writeback').disabled = pending === 0 || !serverAvailable;
        } else {
            bar.style.display = 'none';
        }
    }

    function updateSliderLabels() {
        document.getElementById('val-dup').textContent = dupThreshold.toFixed(2);
        document.getElementById('val-lof').textContent = lofThreshold.toFixed(2);
        document.getElementById('val-mis-conf').textContent = `${Math.round(mislabelConf * 100)}%`;
    }

    // ==========================================================================
    // INTERACTIVE CANVAS PLOT RENDERING
    // ==========================================================================
    
    // Canvas colours follow the theme (CSS variables can't be used on a canvas directly)
    let themeColors = { bg: '#090c10', grid: 'rgba(255,255,255,0.04)', axis: 'rgba(255,255,255,0.55)', text: '#c9d1d9', floor: 'rgba(255,255,255,0.07)', ring: '#ffffff' };
    function readThemeColors() {
        const cs = getComputedStyle(document.documentElement);
        const v = (n, fb) => (cs.getPropertyValue(n) || '').trim() || fb;
        themeColors = {
            bg: v('--canvas-bg', '#090c10'), grid: v('--canvas-grid', 'rgba(255,255,255,0.04)'),
            axis: v('--canvas-axis', 'rgba(255,255,255,0.55)'), text: v('--canvas-text', '#c9d1d9'),
            floor: v('--canvas-floor', 'rgba(255,255,255,0.07)'), ring: v('--canvas-ring', '#ffffff'),
        };
    }

    // data point (3D, PCA space) -> [x, y, depth, perspective scale] around the orbit centre
    function project3(x, y, z) {
        x -= orbitCenter[0]; y -= orbitCenter[1]; z -= orbitCenter[2];
        const cosY = Math.cos(rotationY), sinY = Math.sin(rotationY);
        const cosX = Math.cos(rotationX), sinX = Math.sin(rotationX);
        const x1 = x * cosY - z * sinY;
        const z1 = x * sinY + z * cosY;
        const y2 = y * cosX - z1 * sinX;
        const z2 = y * sinX + z1 * cosX;
        // perspective: larger z2 = closer to the camera = drawn bigger
        const f = camDist / Math.max(camDist * 0.2, camDist - z2);
        return [x1 * f, y2 * f, z2, f];
    }

    function getProjectedCoord(idx) {
        const coords = pcaCoordinates[idx];
        if (!coords) return [0, 0, 0, 1];
        if (!is3D) {
            const m = map2dCoordinates[idx];
            return m ? [m[0], m[1], 0, 1] : [coords[0], coords[1], 0, 1];
        }
        return project3(coords[0], coords[1], coords[2] || 0);
    }

    const toScreen = p => [p[0] * viewZoom + viewPanX, p[1] * viewZoom + viewPanY];
    let fitZoom = 1;
    let lastMouse = null;

    function dataCentre() {
        const n = pcaCoordinates.length || 1;
        return [0, 1, 2].map(k => pcaCoordinates.reduce((s, c) => s + (c[k] || 0), 0) / n);
    }

    function resetCanvasView() {
        if (pcaCoordinates.length === 0) return;
        if (is3D) orbitCenter = dataCentre();
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        pcaCoordinates.forEach((_, i) => {
            const [x, y] = getProjectedCoord(i);
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
        });
        const padX = (maxX - minX) * 0.15 || 1.0;
        const padY = (maxY - minY) * 0.15 || 1.0;
        minX -= padX; maxX += padX;
        minY -= padY; maxY += padY;
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;
        viewZoom = Math.min(width / (maxX - minX), height / (maxY - minY));
        if (!isFinite(viewZoom) || viewZoom <= 0) viewZoom = 1;
        fitZoom = viewZoom;
        viewPanX = width / 2 - (minX + maxX) / 2 * viewZoom;
        viewPanY = height / 2 - (minY + maxY) / 2 * viewZoom;
        drawCanvas();
    }

    function isVisible(it) {
        if (it.removed) return false;
        if (clusterFilter !== null && it.cluster !== clusterFilter) return false;
        return passesFilter(it);
    }

    function drawLine3(a, b) {
        const pa = toScreen(project3(...a)), pb = toScreen(project3(...b));
        ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); ctx.stroke();
        return [pa, pb];
    }

    function drawFloorAndAxes() {
        const c = dataCentre();
        const E = dataExtent * 0.9;
        // light grid floor under the cloud (data +y is screen-down at the default view)
        ctx.save();
        ctx.strokeStyle = themeColors.floor;
        ctx.lineWidth = 1;
        const fy = c[1] + E, steps = 8;
        for (let s = 0; s <= steps; s++) {
            const t = -E + (2 * E * s) / steps;
            drawLine3([c[0] + t, fy, c[2] - E], [c[0] + t, fy, c[2] + E]);
            drawLine3([c[0] - E, fy, c[2] + t], [c[0] + E, fy, c[2] + t]);
        }
        // X / Y / Z axes through the data centre, with labels
        const axes = [['X (PC1)', [E, 0, 0], '#f87171'], ['Y (PC2)', [0, E, 0], '#4ade80'], ['Z (PC3)', [0, 0, E], '#60a5fa']];
        ctx.lineWidth = 1.6;
        ctx.font = '600 12px "Plus Jakarta Sans", sans-serif';
        axes.forEach(([name, d, col]) => {
            ctx.strokeStyle = col;
            ctx.globalAlpha = 0.85;
            const [, pb] = drawLine3([c[0] - d[0], c[1] - d[1], c[2] - d[2]], [c[0] + d[0], c[1] + d[1], c[2] + d[2]]);
            ctx.globalAlpha = 1;
            ctx.fillStyle = col;
            ctx.beginPath(); ctx.arc(pb[0], pb[1], 3, 0, Math.PI * 2); ctx.fill();
            ctx.fillText(name, pb[0] + 6, pb[1] - 6);
        });
        ctx.restore();
    }

    // ---- convex hulls of clusters on the 2D map ----
    function convexHull(pts) {
        if (pts.length < 3) return pts.slice();
        const p = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
        const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
        const lower = [], upper = [];
        for (const q of p) { while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop(); lower.push(q); }
        for (let i = p.length - 1; i >= 0; i--) { const q = p[i]; while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop(); upper.push(q); }
        upper.pop(); lower.pop();
        return lower.concat(upper);
    }

    function getClusterHulls() {
        if (clusterHulls) return clusterHulls;
        const groups = new Map();
        reportData.items.forEach((it, i) => {
            if (it.removed || it.cluster === null || it.cluster === undefined || !map2dCoordinates[i]) return;
            if (!groups.has(it.cluster)) groups.set(it.cluster, []);
            groups.get(it.cluster).push(map2dCoordinates[i]);
        });
        clusterHulls = [...groups.entries()].map(([cluster, pts]) => ({ cluster, hull: convexHull(pts) }));
        return clusterHulls;
    }

    function drawClusterHulls() {
        ctx.save();
        getClusterHulls().forEach(({ cluster, hull }) => {
            if (hull.length < 3 || (clusterFilter !== null && cluster !== clusterFilter)) return;
            const col = clusterColor(cluster);
            ctx.beginPath();
            hull.forEach((p, k) => { const [x, y] = toScreen(p); k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
            ctx.closePath();
            ctx.globalAlpha = 0.07; ctx.fillStyle = col; ctx.fill();
            ctx.globalAlpha = 0.8; ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.setLineDash([6, 4]); ctx.stroke();
            ctx.setLineDash([]);
            const top = hull.reduce((a, b) => (b[1] < a[1] ? b : a));
            const [tx, ty] = toScreen(top);
            ctx.globalAlpha = 1; ctx.fillStyle = col; ctx.font = '700 12px "Plus Jakarta Sans", sans-serif';
            ctx.fillText(`C${cluster}`, tx - 8, ty - 6);
        });
        ctx.restore();
    }

    function drawCanvas() {
        if (!reportData || pcaCoordinates.length === 0) return;
        const N = pcaCoordinates.length;
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        const width = rect.width, height = rect.height;

        ctx.fillStyle = themeColors.bg;
        ctx.fillRect(0, 0, width, height);

        if (is3D) {
            drawFloorAndAxes();
        } else {
            ctx.strokeStyle = themeColors.grid;
            ctx.lineWidth = 1;
            let gridSize = 0.1 * viewZoom;
            while (gridSize < 30) gridSize *= 2;
            while (gridSize > 120) gridSize /= 2;
            for (let x = ((viewPanX % gridSize) + gridSize) % gridSize; x < width; x += gridSize) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke(); }
            for (let y = ((viewPanY % gridSize) + gridSize) % gridSize; y < height; y += gridSize) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke(); }
            if (hasClusters() && (colorBy === 'cluster' || clusterFilter !== null)) drawClusterHulls();
        }

        const items = reportData.items;
        // Project everything once; this cache is also what hover / click / lasso use
        let minZ = Infinity, maxZ = -Infinity;
        projCache = new Array(N);
        for (let i = 0; i < N; i++) {
            if (!isVisible(items[i])) { projCache[i] = null; continue; }
            const p = getProjectedCoord(i);
            const [sx, sy] = toScreen(p);
            projCache[i] = [sx, sy, p[2], 5 * (is3D ? p[3] * p[3] : 1)];
            if (p[2] < minZ) minZ = p[2];
            if (p[2] > maxZ) maxZ = p[2];
        }
        const renderOrder = [];
        for (let i = 0; i < N; i++) if (projCache[i]) renderOrder.push(i);
        if (is3D) renderOrder.sort((a, b) => projCache[a][2] - projCache[b][2]);
        const depthOf = z => (is3D && maxZ > minZ) ? (z - minZ) / (maxZ - minZ) : 1;
        // point size by depth (3D): perspective^2 times a near/far factor, kept in a readable range
        if (is3D) renderOrder.forEach(i => { const p = projCache[i]; p[3] = Math.max(2, Math.min(11, p[3] * (0.7 + 0.5 * depthOf(p[2])))); });

        if (showAnomalyRings) {
            renderOrder.forEach(i => {
                const item = items[i];
                if (item.curDeleted) return;
                const [sx, sy, z, r] = projCache[i];
                if (sx < -20 || sx > width + 20 || sy < -20 || sy > height + 20) return;
                ctx.save();
                if (is3D) ctx.globalAlpha = Math.max(0.15, 0.3 + 0.6 * depthOf(z));
                const R = r * 2.8;
                if (item.is_mislabeled) drawGlowingHalo(sx, sy, R, 'rgba(245, 158, 11, 0.35)', 'rgba(245, 158, 11, 0)');
                else if (item.is_outlier || item.has_issue) drawGlowingHalo(sx, sy, R, 'rgba(244, 63, 94, 0.25)', 'rgba(244, 63, 94, 0)');
                else if (item.is_duplicate) drawGlowingHalo(sx, sy, R, 'rgba(0, 242, 254, 0.25)', 'rgba(0, 242, 254, 0)');
                ctx.restore();
            });
        }

        renderOrder.forEach(i => {
            const item = items[i];
            const [sx, sy, z, r] = projCache[i];
            if (sx < -12 || sx > width + 12 || sy < -12 || sy > height + 12) return;
            ctx.save();
            if (is3D && !item.curDeleted) ctx.globalAlpha = Math.max(0.3, 0.35 + 0.65 * depthOf(z));
            ctx.beginPath();
            if (item.curDeleted) {
                ctx.strokeStyle = 'rgba(244, 63, 94, 0.7)';
                ctx.lineWidth = 1.5;
                const size = Math.max(3, r * 0.8);
                ctx.moveTo(sx - size, sy - size); ctx.lineTo(sx + size, sy + size);
                ctx.moveTo(sx + size, sy - size); ctx.lineTo(sx - size, sy + size);
                ctx.stroke();
            } else {
                ctx.arc(sx, sy, r, 0, Math.PI * 2);
                ctx.fillStyle = pointColor(item);
                ctx.fill();
                ctx.strokeStyle = 'rgba(0, 0, 0, 0.4)';
                ctx.lineWidth = 1;
                ctx.stroke();
                if (relabeledLabels.has(item.id)) {
                    const q = r + 3;
                    ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = 2;
                    ctx.strokeRect(sx - q, sy - q, q * 2, q * 2);
                } else if (item.curKept) {
                    ctx.beginPath(); ctx.arc(sx, sy, r + 3, 0, Math.PI * 2);
                    ctx.strokeStyle = '#10b981'; ctx.lineWidth = 2; ctx.stroke();
                }
            }
            ctx.restore();
        });

        // Selected / pinned / hovered points last, on top of everything, so they are always visible
        [[selectedIndex, 'sel'], [pointPopupIdx, 'sel'], [hoveredIndex, 'hov']].forEach(([i, kind]) => {
            if (i < 0 || i >= N || !projCache[i] || items[i].curDeleted) return;
            const [sx, sy, , r] = projCache[i];
            const R = Math.max(r, 5);
            ctx.save();
            ctx.beginPath(); ctx.arc(sx, sy, R + (kind === 'hov' ? 2 : 3), 0, Math.PI * 2);
            ctx.fillStyle = pointColor(items[i]); ctx.fill();
            ctx.strokeStyle = themeColors.ring; ctx.lineWidth = kind === 'hov' ? 2 : 2.5; ctx.stroke();
            ctx.beginPath(); ctx.arc(sx, sy, R + (kind === 'hov' ? 7 : 9), 0, Math.PI * 2);
            ctx.strokeStyle = kind === 'hov' ? '#fde047' : pointColor(items[i]); ctx.lineWidth = 1.5; ctx.stroke();
            ctx.restore();
        });

        if (lassoSel.size) {
            ctx.save();
            ctx.strokeStyle = '#fde047';
            ctx.lineWidth = 2;
            lassoSel.forEach(i => {
                if (!projCache[i] || items[i].curDeleted) return;
                ctx.beginPath();
                ctx.arc(projCache[i][0], projCache[i][1], projCache[i][3] + 4, 0, Math.PI * 2);
                ctx.stroke();
            });
            ctx.restore();
        }
        if (lassoPts && lassoPts.length > 1) {
            ctx.save();
            ctx.strokeStyle = '#fde047';
            ctx.fillStyle = 'rgba(253, 224, 71, 0.08)';
            ctx.setLineDash([5, 4]);
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(lassoPts[0][0], lassoPts[0][1]);
            lassoPts.slice(1).forEach(([x, y]) => ctx.lineTo(x, y));
            ctx.closePath();
            ctx.fill();
            ctx.stroke();
            ctx.restore();
        }

        if (hoveredIndex >= 0 && hoveredIndex < N && projCache[hoveredIndex] && !items[hoveredIndex].curDeleted && hoveredIndex !== pointPopupIdx) {
            const [sx, sy] = lastMouse || projCache[hoveredIndex];
            updateHoverPreview(sx, sy, items[hoveredIndex]);
        } else {
            updateHoverPreview(0, 0, null);
        }
        drawGizmo();
    }

    // nearest point under the cursor; in 3D the one closest to the camera wins (it is drawn on top)
    function pickAt(mx, my) {
        let best = -1, bestZ = -Infinity, bestD = Infinity;
        const items = reportData ? reportData.items : [];
        for (let i = 0; i < projCache.length; i++) {
            const p = projCache[i];
            if (!p || items[i].curDeleted) continue;
            const d = Math.hypot(mx - p[0], my - p[1]);
            if (d > p[3] + 4) continue;
            if (is3D) {
                if (p[2] > bestZ + 1e-9 || (Math.abs(p[2] - bestZ) <= 1e-9 && d < bestD)) { best = i; bestZ = p[2]; bestD = d; }
            } else if (d < bestD) { best = i; bestD = d; }
        }
        return best;
    }

    // ---- orientation gizmo (3D): click an axis end or a Front/Top/Side button ----
    const GIZMO_AXES = [['X', [1, 0, 0], '#f87171', 'side'], ['Y', [0, 1, 0], '#4ade80', 'top'], ['Z', [0, 0, 1], '#60a5fa', 'front']];
    function gizmoProject(v) {
        const [x, y, z] = v;
        const cosY = Math.cos(rotationY), sinY = Math.sin(rotationY), cosX = Math.cos(rotationX), sinX = Math.sin(rotationX);
        const x1 = x * cosY - z * sinY, z1 = x * sinY + z * cosY;
        return [x1, y * cosX - z1 * sinX, y * sinX + z1 * cosX];
    }
    function drawGizmo() {
        const g = document.getElementById('gizmo');
        if (!g) return;
        g.classList.toggle('hidden', !is3D);
        if (!is3D) return;
        const gc = document.getElementById('gizmo-canvas');
        const gx = gc.getContext('2d');
        const S = 96;
        if (gc.width !== S) { gc.width = S; gc.height = S; }
        gx.clearRect(0, 0, S, S);
        gx.fillStyle = themeColors.bg; gx.globalAlpha = 0.75;
        gx.beginPath(); gx.arc(S / 2, S / 2, S / 2 - 2, 0, Math.PI * 2); gx.fill(); gx.globalAlpha = 1;
        const L = 22;
        const corners = [];
        for (const a of [-1, 1]) for (const b of [-1, 1]) for (const c of [-1, 1]) corners.push(gizmoProject([a * 0.6, b * 0.6, c * 0.6]));
        gx.strokeStyle = themeColors.axis; gx.lineWidth = 1; gx.globalAlpha = 0.5;
        [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]].forEach(([a, b]) => {
            gx.beginPath(); gx.moveTo(S / 2 + corners[a][0] * L, S / 2 + corners[a][1] * L); gx.lineTo(S / 2 + corners[b][0] * L, S / 2 + corners[b][1] * L); gx.stroke();
        });
        gx.globalAlpha = 1;
        const ends = GIZMO_AXES.map(([name, v, col, view]) => ({ name, col, view, p: gizmoProject(v) })).sort((a, b) => a.p[2] - b.p[2]);
        ends.forEach(e => {
            const x = S / 2 + e.p[0] * 34, y = S / 2 + e.p[1] * 34;
            gx.strokeStyle = e.col; gx.lineWidth = 2;
            gx.beginPath(); gx.moveTo(S / 2, S / 2); gx.lineTo(x, y); gx.stroke();
            gx.fillStyle = e.col; gx.beginPath(); gx.arc(x, y, 8, 0, Math.PI * 2); gx.fill();
            gx.fillStyle = '#0b0f14'; gx.font = '700 10px sans-serif'; gx.textAlign = 'center'; gx.textBaseline = 'middle';
            gx.fillText(e.name, x, y + 0.5);
        });
        gc._ends = ends.map(e => ({ view: e.view, x: S / 2 + e.p[0] * 34, y: S / 2 + e.p[1] * 34 })).reverse();  // front-most first
    }

    const VIEWS = { front: [0, 0], top: [-Math.PI / 2, 0], side: [0, Math.PI / 2] };
    let viewAnim = null;
    function snapView(name) {
        const target = VIEWS[name];
        if (!target || !is3D) return;
        cancelAnimationFrame(viewAnim);
        setAutoRotate(false);
        const from = [rotationX, rotationY];
        const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));  // take the short way round
        const d = [wrap(target[0] - from[0]), wrap(target[1] - from[1])];
        const t0 = performance.now();
        const step = now => {
            const t = Math.min(1, (now - t0) / 350), e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
            rotationX = from[0] + d[0] * e; rotationY = from[1] + d[1] * e;
            drawCanvas();
            if (t < 1) viewAnim = requestAnimationFrame(step);
        };
        viewAnim = requestAnimationFrame(step);
        plotCard.dataset.view = name;
    }

    let autoRotFrame = null;
    function setAutoRotate(on) {
        autoRotate = on;
        const btn = document.getElementById('btn-plot-autorotate');
        if (btn) btn.classList.toggle('active', on);
        cancelAnimationFrame(autoRotFrame);
        const tick = () => {
            if (!autoRotate) return;
            if (is3D && !isDrawing && !isPanning && activeTab === 'overview') { rotationY += 0.004; drawCanvas(); }
            autoRotFrame = requestAnimationFrame(tick);
        };
        if (on) autoRotFrame = requestAnimationFrame(tick);
    }

    function itemFlags(item) {
        const f = [];
        if (item.is_mislabeled) f.push(`looks like '${item.pred}' (${pct(item.pred_conf)})`);
        if (item.is_outlier) f.push(`unusual (LOF ${item.lof.toFixed(2)})`);
        if (item.has_issue) f.push(...item.issues);
        if (item.is_duplicate) f.push('near-duplicate');
        return f;
    }

    function itemDecision(item) {
        if (item.curDeleted) return ['To be removed', '#f43f5e'];
        if (relabeledLabels.has(item.id)) return [`Relabeled ${item.label} → ${item.curLabel}`, '#f59e0b'];
        if (item.curKept) return ['Approved', '#10b981'];
        return null;
    }

    let hoverShownId = null;
    function updateHoverPreview(sx, sy, item) {
        const box = document.getElementById('plot-hover');
        if (!item) { box.classList.add('hidden'); hoverShownId = null; return; }
        if (hoverShownId !== item.id) {
            hoverShownId = item.id;
            const flags = itemFlags(item);
            const dec = itemDecision(item);
            box.innerHTML = `<img src="${esc(item.path)}" alt="">
                <div><b>${esc(item.curLabel)}</b>${item.pred ? ` · model: ${esc(item.pred)} ${pct(item.pred_conf)}` : ''}</div>
                <div style="word-break:break-all">${esc(item.split)}/${esc(item.image)}</div>
                ${flags.length ? `<div class="ph-flags">${flags.map(esc).join('<br>')}</div>` : ''}
                ${dec ? `<div class="ph-decision" style="color:${dec[1]}">${esc(dec[0])}</div>` : ''}
                <div style="margin-top:4px;opacity:.7">Click to inspect</div>`;
        }
        box.classList.remove('hidden');
        const W = canvas.clientWidth, H = canvas.clientHeight;
        const bw = box.offsetWidth || 200, bh = box.offsetHeight || 260;
        let x = sx + 16, y = sy - bh / 2;
        if (x + bw > W - 8) x = sx - bw - 16;
        y = Math.max(8, Math.min(H - bh - 8, y));
        box.style.left = `${Math.max(8, x)}px`;
        box.style.top = `${y}px`;
    }
    
    function drawGlowingHalo(x, y, r, innerColor, outerColor) {
        const grad = ctx.createRadialGradient(x, y, 4, x, y, r);
        grad.addColorStop(0, innerColor);
        grad.addColorStop(1, outerColor);
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fillStyle = grad;
        ctx.fill();
    }
    
    function drawTooltip(x, y, item) {
        ctx.save();
        
        const titleText = `${item.id}`;
        const labelText = `Label: ${item.curLabel}`;
        
        let tags = [];
        if (item.is_mislabeled) tags.push("MISLABELED");
        if (item.is_outlier) tags.push("UNUSUAL");
        if (item.has_issue) tags.push("BOX ISSUE");
        if (item.is_duplicate) tags.push("DUPLICATE");
        
        // Calculate dynamic dimensions
        ctx.font = 'bold 12px var(--font-sans)';
        const w1 = ctx.measureText(titleText).width;
        ctx.font = '11px var(--font-sans)';
        const w2 = ctx.measureText(labelText).width;
        
        const boxWidth = Math.max(140, w1 + 30, w2 + 20);
        const boxHeight = tags.length > 0 ? 64 : 46;
        
        // Align tooltip above point
        const tx = Math.max(10, Math.min(canvas.clientWidth - boxWidth - 10, x - boxWidth / 2));
        const ty = y - boxHeight - 12;
        
        // Draw background card with blur border
        ctx.fillStyle = 'rgba(13, 17, 23, 0.95)';
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
        ctx.lineWidth = 1;
        
        // Rounded Rect
        const r = 6;
        ctx.beginPath();
        ctx.moveTo(tx + r, ty);
        ctx.lineTo(tx + boxWidth - r, ty);
        ctx.quadraticCurveTo(tx + boxWidth, ty, tx + boxWidth, ty + r);
        ctx.lineTo(tx + boxWidth, ty + boxHeight - r);
        ctx.quadraticCurveTo(tx + boxWidth, ty + boxHeight, tx + boxWidth - r, ty + boxHeight);
        ctx.lineTo(tx + r, ty + boxHeight);
        ctx.quadraticCurveTo(tx, ty + boxHeight, tx, ty + boxHeight - r);
        ctx.lineTo(tx, ty + r);
        ctx.quadraticCurveTo(tx, ty, tx + r, ty);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        
        // Text Strings
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 12px var(--font-sans)';
        ctx.fillText(titleText, tx + 10, ty + 18);
        
        ctx.fillStyle = '#8b949e';
        ctx.font = '11px var(--font-sans)';
        ctx.fillText(labelText, tx + 10, ty + 32);
        
        // Draw sub-indicators if anomalies exist
        if (tags.length > 0) {
            let txOffset = tx + 10;
            tags.forEach(tag => {
                // Canvas can't resolve CSS variables, so use literal colors
                let badgeColor = '#8b949e';
                if (tag === 'DUPLICATE') badgeColor = '#00f2fe';
                else if (tag === 'UNUSUAL' || tag === 'BOX ISSUE') badgeColor = '#f43f5e';
                else if (tag === 'MISLABELED') badgeColor = '#f59e0b';
                
                ctx.fillStyle = badgeColor;
                ctx.font = 'bold 9px var(--font-sans)';
                const tagW = ctx.measureText(tag).width;
                
                ctx.fillRect(txOffset - 2, ty + 39, tagW + 4, 14);
                ctx.fillStyle = '#090c10';
                ctx.fillText(tag, txOffset, ty + 50);
                txOffset += tagW + 10;
            });
        }
        
        ctx.restore();
    }
    
    // ==========================================================================
    // INTERACTION CONTROLLERS (Sliders, Canvas, Upload, Export)
    // ==========================================================================
    
    function setupCanvas() {
        readThemeColors();
        const mousePos = e => { const r = canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
        canvas.addEventListener('contextmenu', e => e.preventDefault());

        // left-drag: rotate (3D) / pan (2D) · right-drag or ctrl-drag: pan · shift-drag: lasso (2D and 3D)
        canvas.addEventListener('mousedown', (e) => {
            if (e.button === 0 && e.shiftKey) {
                lassoPts = [mousePos(e)];
                if (!(e.ctrlKey || e.metaKey)) lassoSel.clear();
                e.preventDefault();
                return;
            }
            if (e.button !== 0 && e.button !== 2) return;
            isPanning = e.button === 2 || e.ctrlKey || e.metaKey || !is3D;
            isDrawing = true;
            dragStart.x = e.clientX;
            dragStart.y = e.clientY;
            dragPanStart.x = viewPanX;
            dragPanStart.y = viewPanY;
            dragRotationStart.x = rotationX;
            dragRotationStart.y = rotationY;
            canvas.classList.add(isPanning ? 'grabbing' : 'rotating');
            e.preventDefault();
        });

        window.addEventListener('mousemove', (e) => {
            const [mouseX, mouseY] = mousePos(e);
            if (lassoPts) {
                lassoPts.push([mouseX, mouseY]);
                drawCanvas();
                return;
            }
            if (isDrawing) {
                const dx = e.clientX - dragStart.x;
                const dy = e.clientY - dragStart.y;
                if (!isPanning) {
                    rotationY = dragRotationStart.y + dx * 0.007;
                    rotationX = dragRotationStart.x + dy * 0.007;
                } else {
                    viewPanX = dragPanStart.x + dx;
                    viewPanY = dragPanStart.y + dy;
                }
                drawCanvas();
                return;
            }
            if (!reportData || e.target !== canvas) return;
            lastMouse = [mouseX, mouseY];
            const prev = hoveredIndex;
            hoveredIndex = pickAt(mouseX, mouseY);
            canvas.style.cursor = hoveredIndex >= 0 ? 'pointer' : '';
            if (hoveredIndex !== prev) drawCanvas();
            else if (hoveredIndex >= 0 && hoveredIndex !== pointPopupIdx) updateHoverPreview(mouseX, mouseY, reportData.items[hoveredIndex]);
        });

        window.addEventListener('mouseup', (e) => {
            if (lassoPts) {
                finishLasso();
                suppressClick = true;
                return;
            }
            if (isDrawing) lastDragDist = Math.hypot(e.clientX - dragStart.x, e.clientY - dragStart.y);
            isDrawing = false;
            isPanning = false;
            canvas.classList.remove('grabbing', 'rotating');
        });

        // wheel: zoom toward the cursor (2D and 3D)
        canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const [mouseX, mouseY] = mousePos(e);
            const dataX = (mouseX - viewPanX) / viewZoom;
            const dataY = (mouseY - viewPanY) / viewZoom;
            const zoomFactor = Math.exp(-Math.max(-100, Math.min(100, e.deltaY)) * 0.0015);
            viewZoom = Math.max(fitZoom * 0.2, Math.min(fitZoom * 200, viewZoom * zoomFactor));
            viewPanX = mouseX - dataX * viewZoom;
            viewPanY = mouseY - dataY * viewZoom;
            drawCanvas();
        }, { passive: false });

        // double-click: centre (and orbit around, in 3D) the point under the cursor
        canvas.addEventListener('dblclick', (e) => {
            const [mx, my] = mousePos(e);
            const i = pickAt(mx, my);
            if (i >= 0) centreOnPoint(i, 1.6);
        });

        document.getElementById('btn-plot-reset').addEventListener('click', () => {
            if (is3D) { rotationX = -0.5; rotationY = 0.6; }
            resetCanvasView();
        });

        document.getElementById('btn-plot-toggle-flags').addEventListener('click', (e) => {
            showAnomalyRings = !showAnomalyRings;
            e.currentTarget.classList.toggle('active', showAnomalyRings);
            drawCanvas();
        });

        document.getElementById('btn-plot-toggle-3d').addEventListener('click', () => setPlotMode(!is3D));
        document.getElementById('btn-plot-autorotate').addEventListener('click', () => {
            if (!is3D) setPlotMode(true);
            setAutoRotate(!autoRotate);
        });

        // gizmo: buttons + clicking an axis end
        document.querySelectorAll('#gizmo [data-view]').forEach(b => b.addEventListener('click', () => snapView(b.dataset.view)));
        document.getElementById('gizmo-canvas').addEventListener('click', e => {
            const gc = e.currentTarget, r = gc.getBoundingClientRect();
            const x = (e.clientX - r.left) * gc.width / r.width, y = (e.clientY - r.top) * gc.height / r.height;
            const hit = (gc._ends || []).find(p => Math.hypot(p.x - x, p.y - y) < 12);
            if (hit) snapView(hit.view);
        });

        // click: pick fresh at the click position (not a stale hover), open the pinned window
        canvas.addEventListener('click', (e) => {
            if (suppressClick) { suppressClick = false; return; }
            if (lastDragDist > 4) { lastDragDist = 0; return; }
            const [mx, my] = mousePos(e);
            const i = pickAt(mx, my);
            if (i >= 0) {
                hoveredIndex = i;
                selectedIndex = i;
                openPointPopup(i);
                drawCanvas();
            }
        });

        canvas.addEventListener('mouseleave', () => {
            lastMouse = null;
            if (hoveredIndex >= 0 && !isDrawing) { hoveredIndex = -1; drawCanvas(); }
        });

        document.getElementById('btn-plot-fullscreen').addEventListener('click', togglePlotFullscreen);
        document.addEventListener('fullscreenchange', () => {
            if (!document.fullscreenElement) plotCard.classList.remove('plot-fullscreen');
            syncFullscreenUi();
        });
        let resizeTimer = null;
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimer);
            resizeTimer = setTimeout(() => { if (activeTab === 'overview') resetCanvasView(); }, 80);
        });
        setupPointModal();
        setupSelectionBar();
        setupPopupDrag();
    }

    function setPlotMode(threeD) {
        is3D = threeD;
        if (!is3D) setAutoRotate(false);
        document.getElementById('btn-plot-toggle-3d').classList.toggle('active', is3D);
        document.getElementById('btn-plot-toggle-3d').innerHTML = is3D ? '<i class="fa-solid fa-map"></i> 2D map' : '<i class="fa-solid fa-cube"></i> 3D view';
        plotCard.classList.toggle('is-3d', is3D);
        const cardTitle = document.getElementById('plot-card-title');
        const instructions = document.getElementById('plot-instructions');
        if (is3D) {
            cardTitle.innerHTML = '<i class="fa-solid fa-cube"></i> 3D view (PCA: X/Y/Z = first 3 components)';
            instructions.innerHTML = 'Drag: rotate · Right/Ctrl-drag: pan · Wheel: zoom at cursor · Double-click: centre · Shift-drag: select · ?: help';
        } else {
            cardTitle.innerHTML = '<i class="fa-solid fa-circle-nodes"></i> Similarity map (2D, t-SNE)';
            instructions.innerHTML = 'Drag: pan · Wheel: zoom at cursor · Double-click: centre · Click: inspect · Shift-drag: select · ?: help';
        }
        resetCanvasView();
    }

    function centreOnPoint(i, zoomBy = 1) {
        const W = canvas.clientWidth, H = canvas.clientHeight;
        if (is3D) {
            const c = pcaCoordinates[i];
            orbitCenter = [c[0], c[1], c[2] || 0];
            viewZoom = Math.min(fitZoom * 200, viewZoom * zoomBy);
            viewPanX = W / 2; viewPanY = H / 2;     // the orbit centre projects to (0, 0)
        } else {
            viewZoom = Math.min(fitZoom * 200, viewZoom * zoomBy);
            const [px, py] = getProjectedCoord(i);
            viewPanX = W / 2 - px * viewZoom;
            viewPanY = H / 2 - py * viewZoom;
        }
        drawCanvas();
    }

    // ==========================================================================
    // SIMILARITY MAP: FULL SCREEN + POINT INSPECTOR
    // ==========================================================================

    const plotCard = document.getElementById('plot-card');

    function isPlotFullscreen() {
        return document.fullscreenElement === plotCard || plotCard.classList.contains('plot-fullscreen');
    }

    function syncFullscreenUi() {
        const fs = isPlotFullscreen();
        document.body.classList.toggle('plot-fs-lock', fs);
        const btn = document.getElementById('btn-plot-fullscreen');
        btn.classList.toggle('active', fs);
        btn.innerHTML = fs ? '<i class="fa-solid fa-compress"></i> Exit full screen' : '<i class="fa-solid fa-expand"></i> Full screen';
        // Overlays must live inside the fullscreen element to be visible
        const host = document.fullscreenElement || document.body;
        const pm = document.getElementById('point-modal');
        const fm = document.getElementById('frame-modal');
        if (pm.parentElement !== host) host.appendChild(pm);
        if (fm.parentElement !== host) host.appendChild(fm);
        requestAnimationFrame(() => resetCanvasView());
    }

    function togglePlotFullscreen() {
        if (isPlotFullscreen()) {
            plotCard.classList.remove('plot-fullscreen');
            if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
            syncFullscreenUi();
            return;
        }
        if (activeTab !== 'overview') switchTab('overview');
        const fallback = () => { plotCard.classList.add('plot-fullscreen'); syncFullscreenUi(); };
        if (plotCard.requestFullscreen) {
            plotCard.requestFullscreen().catch(fallback);  // fullscreenchange syncs the UI on success
        } else {
            fallback();
        }
    }

    let pointModalIdx = -1;

    function setupPointModal() {
        const modal = document.getElementById('point-modal');
        modal.addEventListener('click', e => { if (e.target === modal) closePointModal(); });
        document.getElementById('pm-close').onclick = closePointModal;
        const cur = () => reportData && reportData.items[pointModalIdx];
        document.getElementById('pm-approve').onclick = () => {
            const it = cur(); if (!it) return;
            approveCrop(it.id, `Confirmed '${it.curLabel}' from the similarity map.`);
            refreshPointModal();
        };
        document.getElementById('pm-remove').onclick = () => {
            const it = cur(); if (!it) return;
            excludeCrop(it.id, 'Deleted box from the similarity map.');
            refreshPointModal();
        };
        document.getElementById('pm-relabel-btn').onclick = () => {
            const it = cur(); if (!it) return;
            const nl = document.getElementById('pm-relabel').value;
            if (nl && nl !== it.curLabel) relabelCrop(it.id, it.curLabel, nl);
            refreshPointModal();
        };
        document.getElementById('pm-undo').onclick = () => {
            const it = cur(); if (!it) return;
            undoCrop(it.id);
            refreshPointModal();
        };
        document.getElementById('pm-editor').onclick = () => {
            const it = cur(); if (!it) return;
            if (!serverAvailable || !window.CleanCropEditor) { alert('The box editor needs the CleanCrop server. Start the dashboard with CleanCrop.bat.'); return; }
            closePointModal();
            if (isPlotFullscreen()) togglePlotFullscreen();  // the editor overlay lives outside the plot card
            window.CleanCropEditor.open(it.id);
        };
    }

    function openPointModal(idx) {
        const it = reportData.items[idx];
        if (!it) return;
        pointModalIdx = idx;
        updateHoverPreview(0, 0, null);
        const host = document.fullscreenElement || document.body;
        const modal = document.getElementById('point-modal');
        if (modal.parentElement !== host) host.appendChild(modal);
        document.getElementById('pm-img').src = serverAvailable ? `/api/frame?id=${encodeURIComponent(it.id)}&t=${Date.now()}` : it.path;
        document.getElementById('pm-crop').src = it.path;
        refreshPointModal();
        modal.classList.remove('hidden');
    }

    function refreshPointModal() {
        const it = reportData && reportData.items[pointModalIdx];
        if (!it) return;
        const flags = itemFlags(it);
        const dec = itemDecision(it);
        document.getElementById('point-modal-title').textContent = `${it.split}/${it.image} · box line ${it.line + 1}`;
        document.getElementById('pm-info').innerHTML = `
            <div>Label: ${pill(it.curLabel)}${relabeledLabels.has(it.id) ? ` (was ${esc(it.label)})` : ''}</div>
            <div>Model thinks: <b>${esc(it.pred || 'n/a')}</b> ${it.pred ? pct(it.pred_conf) : ''} · confidence in label: ${pct(it.given_conf)}</div>
            <div>Unusual score (LOF): <b>${it.lof.toFixed(2)}</b></div>
            ${flags.length ? `<div style="color:var(--color-mislabel)">${flags.map(esc).join('<br>')}</div>` : '<div>No flags</div>'}
            <div>Decision: <b style="color:${dec ? dec[1] : 'inherit'}">${dec ? esc(dec[0]) : 'none yet'}</b>${serverAvailable ? '' : ' (not saved: no server)'}</div>`;
        document.getElementById('pm-relabel').innerHTML = classOptions(it.curLabel);
        document.getElementById('pm-undo').disabled = !dec;
        document.getElementById('pm-editor').disabled = !!it.removed;
    }

    function closePointModal() {
        document.getElementById('point-modal').classList.add('hidden');
        pointModalIdx = -1;
    }

    // ==========================================================================
    // CONTROLS
    // ==========================================================================

    function setupSliders() {
        const bind = (id, setter) => {
            document.getElementById(id).addEventListener('input', e => {
                setter(parseFloat(e.target.value));
                updateSliderLabels();
                misCursor = -1;
                outCursor = -1;
                recomputeCuration();
                saveState();
            });
        };
        bind('slider-dup', v => { dupThreshold = v; });
        bind('slider-lof', v => { lofThreshold = v; });
        bind('slider-mis-conf', v => { mislabelConf = v; });
    }

    function setupActionButtons() {
        document.querySelectorAll('.stat-card.clickable').forEach(card => {
            card.addEventListener('click', () => switchTab(card.getAttribute('data-tab-link')));
        });

        document.getElementById('btn-exp-confirm-relabel').onclick = () => {
            const item = reportData.items[selectedIndex];
            if (!item) return;
            relabelCrop(item.id, item.curLabel, document.getElementById('exp-relabel-select').value);
        };
        document.getElementById('btn-exp-cancel-relabel').onclick = () => {
            document.getElementById('relabel-dropdown-container').classList.add('hidden');
        };
        document.getElementById('btn-exp-frame').onclick = () => {
            const item = reportData.items[selectedIndex];
            if (item) openFrame(item.id);
        };

        document.getElementById('btn-export-actions').addEventListener('click', exportCurationReport);
        document.getElementById('btn-commit-writeback').addEventListener('click', commitWriteback);

        ['explorer-search', 'filter-class', 'filter-split', 'filter-anomaly', 'sort-by'].forEach(id => {
            document.getElementById(id).addEventListener(id === 'explorer-search' ? 'input' : 'change', () => {
                shown.explorer = 300;
                renderExplorerTab();
            });
        });
    }

    // Full-frame viewer: click any crop image to see the whole image with all its boxes
    function setupFrameViewer() {
        document.addEventListener('click', e => {
            const img = e.target.closest('img[data-frame]');
            if (img && img.dataset.frame) {
                e.stopPropagation();
                openFrame(img.dataset.frame);
            }
        }, true);
        document.getElementById('frame-modal').addEventListener('click', closeFrame);
    }

    function openFrame(id) {
        const item = itemById(id);
        if (!item) return;
        if (!serverAvailable) {
            alert('The full-frame view needs the CleanCrop server. Start the dashboard with CleanCrop.bat.');
            return;
        }
        document.getElementById('frame-img').src = `/api/frame?id=${encodeURIComponent(id)}&t=${Date.now()}`;
        document.getElementById('frame-caption').textContent =
            `${item.split}/${item.image} — highlighted box: '${item.curLabel}' (line ${item.line + 1}). Other boxes in this image are outlined thinly. Click or press Esc to close.`;
        document.getElementById('frame-modal').classList.remove('hidden');
    }

    function closeFrame() {
        document.getElementById('frame-modal').classList.add('hidden');
    }

    function setupKeyboard() {
        document.addEventListener('keydown', e => {
            if (e.target instanceof Element && e.target.matches('input, select, textarea')) return;
            const modalOpen = !document.getElementById('frame-modal').classList.contains('hidden');
            const pointOpen = !document.getElementById('point-modal').classList.contains('hidden');
            if (e.key === 'Escape') {
                if (pointPopupIdx >= 0 && !modalOpen && !pointOpen) { closePointPopup(); return; }
                if (lassoSel.size && !modalOpen && !pointOpen) { clearSelection(); return; }
                if (modalOpen) closeFrame();
                else if (pointOpen) closePointModal();
                else if (isPlotFullscreen()) togglePlotFullscreen();
                return;
            }
            if (activeTab === 'overview' && !modalOpen && !pointOpen && e.key.toLowerCase() === 'f' && !e.ctrlKey && !e.metaKey && !e.altKey) {
                togglePlotFullscreen();
                e.preventDefault();
                return;
            }
            if (e.ctrlKey || e.metaKey || e.altKey) return;
            const key = e.key.toLowerCase();
            if (key === 'u' && !modalOpen && !pointOpen && ['mislabels', 'outliers', 'overview'].includes(activeTab)) {
                undoLast(); e.preventDefault(); return;
            }
            if ((activeTab !== 'mislabels' && activeTab !== 'outliers') || pointOpen) return;
            const list = activeTab === 'mislabels' ? mislabelList : outlierList;
            const cur = activeTab === 'mislabels' ? misCursor : outCursor;
            if (!list.length) return;
            const isNext = e.key === 'ArrowDown' || e.key === 'ArrowRight' || key === 'd' || key === 'j';
            const isPrev = e.key === 'ArrowUp' || e.key === 'ArrowLeft' || key === 'w';
            if (cur < 0 || cur >= list.length) {
                if (isNext || isPrev) { selectReview(0); e.preventDefault(); }
                return;
            }
            const it = list[cur];
            if (key === 'e') return;  // edit boxes: handled by editor.js
            const tag = activeTab === 'mislabels' ? 'suspected mislabeled' : 'unusual';
            const n = /^[1-9]$/.test(e.key) ? +e.key : 0;
            if (key === 'a' || key === 'k') decideKb(it, () => approveCrop(it.id, `Confirmed label '${it.curLabel}'.`));
            else if (key === 'x' || e.key === 'Delete') decideKb(it, () => excludeCrop(it.id, `Deleted ${tag} box.`));
            else if (key === 'r') { if (!it.pred || it.pred === it.curLabel) return; decideKb(it, () => relabelCrop(it.id, it.curLabel, it.pred)); }
            else if (n) {
                const cls = classNames[n - 1];
                if (!cls || cls === it.curLabel) return;
                decideKb(it, () => relabelCrop(it.id, it.curLabel, cls));
            }
            else if (key === 'f') { modalOpen ? closeFrame() : openFrame(it.id); e.preventDefault(); return; }
            else if (isNext) selectReview(cur + 1);
            else if (isPrev) selectReview(cur - 1);
            else return;
            closeFrame();
            e.preventDefault();
        });
    }

    function selectReview(idx) {
        const list = activeTab === 'mislabels' ? mislabelList : outlierList;
        const max = Math.min(list.length, activeTab === 'mislabels' ? shown.mislabels : shown.outliers) - 1;
        if (max < 0) return;
        idx = Math.max(0, Math.min(idx, max));
        if (activeTab === 'mislabels') selectMislabelItem(idx); else selectOutlierCard(idx);
    }

    function selectOutlierCard(idx) {
        const it = outlierList[idx];
        if (!it) { outCursor = -1; return; }
        outCursor = idx;
        selectedIndex = it.index;
        document.querySelectorAll('#outliers-grid-container .outlier-card').forEach(card => {
            const on = parseInt(card.dataset.idx) === idx;
            card.classList.toggle('kb-active', on);
            if (on) card.scrollIntoView({ block: 'nearest' });
        });
    }

    function decideKb(it, fn) {
        reviewUndo.push([it.id]);
        fn();
    }

    function undoLast() {
        const ids = reviewUndo.pop();
        if (!ids) return;
        ids.forEach(id => {
            deletedIds.delete(id); relabeledLabels.delete(id); keptIds.delete(id);
            for (let i = actionsLog.length - 1; i >= 0; i--) if (actionsLog[i].id === id) actionsLog.splice(i, 1);
        });
        saveState();
        recomputeCuration();
        if (activeTab === 'mislabels' || activeTab === 'outliers') {
            const list = activeTab === 'mislabels' ? mislabelList : outlierList;
            const idx = list.findIndex(x => x.id === ids[0]);
            if (idx >= 0) selectReview(idx);
        }
        if (pointPopupIdx >= 0) refreshPointPopup();
    }

    // ==========================================================================
    // CLUSTERS (colour by class / cluster)
    // ==========================================================================

    const hasClusters = () => !!reportData && reportData.items.some(it => it.cluster !== null && it.cluster !== undefined);
    const clusterColor = c => `hsl(${(c * 137.508 + 20) % 360}, 75%, 58%)`;

    function pointColor(item) {
        if (colorBy === 'cluster' && item.cluster !== null && item.cluster !== undefined) return clusterColor(item.cluster);
        return classColors[item.curLabel] || defaultColor;
    }

    function clusterSummary() {
        // Same rules as cluster_summary() in the Python backend, but on the current (reviewed) labels
        const groups = new Map();
        reportData.items.forEach(it => {
            if (it.curDeleted || it.removed || it.cluster === null || it.cluster === undefined) return;
            if (!groups.has(it.cluster)) groups.set(it.cluster, []);
            groups.get(it.cluster).push(it);
        });
        return [...groups.entries()].sort((a, b) => a[0] - b[0]).map(([c, g]) => {
            const counts = {};
            g.forEach(it => { counts[it.curLabel] = (counts[it.curLabel] || 0) + 1; });
            const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
            const [major, nMajor] = sorted[0];
            const purity = nMajor / g.length;
            const unusual = g.filter(it => it.lof >= lofThreshold).length / g.length;
            const hint = purity < 0.7 ? 'mixed: check labels'
                : unusual >= 0.3 ? 'many unusual boxes: background-like?'
                : purity < 0.9 ? 'check the minority boxes' : 'clean';
            const second = sorted[1] ? `, ${Math.round(100 * sorted[1][1] / g.length)}% ${sorted[1][0]}` : '';
            const name = `C${c}: ${purity >= 0.999 ? 'all' : 'mostly'} ${major}${second} – ${hint === 'clean' ? 'ok' : 'check'}`;
            return { cluster: c, size: g.length, major, purity, hint, name };
        });
    }

    function renderPlotLegend() {
        const el = document.getElementById('plot-legend');
        const btn = document.getElementById('btn-plot-color');
        if (btn) {
            btn.disabled = !hasClusters();
            btn.title = btn.disabled ? 'No clusters in this report: run the analysis again' : 'Colour points by class or by embedding cluster';
            btn.innerHTML = `<i class="fa-solid fa-palette"></i> Color: ${colorBy}`;
        }
        if (colorBy === 'cluster' && hasClusters()) {
            el.innerHTML = clusterSummary().map(s => `<button class="legend-item cluster-legend ${clusterFilter === s.cluster ? 'active' : ''}" data-cluster="${s.cluster}" title="${esc(s.hint)} · click to show only this cluster">
                <span class="legend-dot" style="background-color:${clusterColor(s.cluster)};"></span>
                <span>${esc(s.name)} · ${s.size}</span></button>`).join('') +
                (clusterFilter !== null ? '<button class="legend-item legend-all" data-cluster="all">Show all clusters</button>' : '');
            el.querySelectorAll('[data-cluster]').forEach(b => b.onclick = () => {
                const c = b.dataset.cluster;
                setClusterFilter(c === 'all' || +c === clusterFilter ? null : +c);
            });
        } else {
            el.innerHTML = Object.entries(classColors).map(([cls, color]) =>
                `<button class="legend-item ${gFilter.cls === cls ? 'active' : ''}" data-cls="${esc(cls)}" title="Click to show only this class"><span class="legend-dot" style="background-color:${color};"></span><span>${esc(cls)}</span></button>`).join('');
            el.querySelectorAll('[data-cls]').forEach(b => b.onclick = () => {
                gFilter.cls = gFilter.cls === b.dataset.cls ? '' : b.dataset.cls;
                document.getElementById('gf-class').value = gFilter.cls;
                applyFilters();
            });
        }
    }

    function setClusterFilter(c) {
        clusterFilter = c;
        renderPlotLegend();
        renderClusterGrid();
        drawCanvas();
        updateFsSide();
    }

    function renderClusterGrid() {
        const box = document.getElementById('cluster-grid');
        if (!box || !reportData) return;
        if (clusterFilter === null) { box.classList.add('hidden'); box.innerHTML = ''; return; }
        const s = clusterSummary().find(x => x.cluster === clusterFilter);
        const items = reportData.items.filter(isVisible);
        box.innerHTML = `<div class="cg-head"><b>${esc(s ? s.name : `C${clusterFilter}`)}</b> · ${items.length} boxes shown
                <button class="btn btn-small" id="cg-close"><i class="fa-solid fa-xmark"></i> Show all</button></div>
            <div class="cg-grid">${items.slice(0, 300).map(it => {
                const dec = itemDecision(it);
                return `<button class="cg-thumb ${isFlagged(it) ? 'flag' : ''}" data-idx="${it.index}" title="${esc(it.curLabel)}${dec ? ' · ' + esc(dec[0]) : ''}">
                    <img loading="lazy" src="${esc(it.path)}" alt=""><span>${esc(it.curLabel)}</span>${dec ? `<i style="background:${dec[1]}"></i>` : ''}</button>`;
            }).join('')}</div>${items.length > 300 ? '<p class="meter-desc">First 300 shown.</p>' : ''}`;
        box.classList.remove('hidden');
        box.querySelector('#cg-close').onclick = () => setClusterFilter(null);
        box.querySelectorAll('.cg-thumb').forEach(b => b.onclick = () => { selectedIndex = +b.dataset.idx; openPointPopup(+b.dataset.idx); drawCanvas(); });
    }

    // ==========================================================================
    // GLOBAL FILTERS, STATUS, NEXT PROBLEM
    // ==========================================================================

    // Flagged at the current thresholds, whether or not it has been reviewed yet
    const isFlagged = it => !!(it.is_mislabeled || it.is_outlier || it.has_issue || it.is_duplicate
        || (it.issues && it.issues.length) || it.lof >= lofThreshold
        || (it.pred && it.pred !== (it.relabeled_from || it.label) && (it.pred_conf || 0) >= mislabelConf));

    function passesFilter(it) {
        if (gFilter.cls && it.curLabel !== gFilter.cls) return false;
        if (gFilter.cam && it.camera !== gFilter.cam) return false;
        if (gFilter.period && it.period !== gFilter.period) return false;
        if (gFilter.flagged && !isFlagged(it)) return false;
        if (gFilter.unreviewed && itemDecision(it)) return false;
        return true;
    }

    function applyFilters() {
        gFilter.cls = document.getElementById('gf-class').value;
        gFilter.cam = document.getElementById('gf-camera').value;
        gFilter.period = document.getElementById('gf-period').value;
        gFilter.flagged = document.getElementById('gf-flagged').checked;
        gFilter.unreviewed = document.getElementById('gf-unreviewed').checked;
        const n = reportData ? reportData.items.filter(isVisible).length : 0;
        document.getElementById('gf-count').textContent = reportData ? `${n} shown` : '';
        recomputeCuration();
        renderPlotLegend();
    }

    function reviewCounts() {
        const flagged = reportData.items.filter(it => !it.removed && isFlagged(it));
        const done = flagged.filter(it => itemDecision(it)).length;
        return { total: flagged.length, done, left: flagged.length - done };
    }

    function updateTopStatus() {
        const el = document.getElementById('top-status');
        if (!el || !reportData) return;
        const c = reviewCounts();
        el.innerHTML = `<span class="ts-name" title="${esc(reportData.dataset_dir || '')}">${esc(reportData.dataset_name || 'Dataset')}</span>
            <span class="ts-pill">Reviewed <b>${c.done}</b>/${c.total}</span>
            <span class="ts-pill ${c.left ? 'warn' : 'ok'}"><b>${c.left}</b> left to review</span>`;
        const gc = document.getElementById('gf-count');
        if (gc) gc.textContent = `${reportData.items.filter(isVisible).length} shown`;
    }

    function nextProblem() {
        if (!reportData) return;
        const items = reportData.items;
        const ok = it => !it.removed && !it.curDeleted && isFlagged(it) && !itemDecision(it) && isVisible(it);
        const start = pointPopupIdx >= 0 ? pointPopupIdx : selectedIndex;
        let found = -1;
        for (let k = 1; k <= items.length; k++) {
            const i = ((start < 0 ? -1 : start) + k + items.length) % items.length;
            if (ok(items[i])) { found = i; break; }
        }
        if (found < 0) { showToast('No unreviewed problems left with the current filters.'); return; }
        if (activeTab !== 'overview') switchTab('overview');
        selectedIndex = found;
        centreOnPoint(found, 1);
        openPointPopup(found);
    }

    function updateFsSide() {
        const el = document.getElementById('fs-side');
        if (!el || !reportData) return;
        const c = reviewCounts();
        const it = reportData.items[pointPopupIdx >= 0 ? pointPopupIdx : selectedIndex];
        const filters = [gFilter.cls, gFilter.cam, gFilter.period, gFilter.flagged && 'flagged', gFilter.unreviewed && 'unreviewed', clusterFilter !== null && `C${clusterFilter}`].filter(Boolean);
        el.innerHTML = `<h4>${esc(reportData.dataset_name || 'Dataset')}</h4>
            <div class="fs-row">Reviewed <b>${c.done}</b>/${c.total} · <b>${c.left}</b> left</div>
            <div class="fs-row">Showing <b>${reportData.items.filter(isVisible).length}</b> boxes${filters.length ? ` · filter: ${filters.map(esc).join(', ')}` : ''}</div>
            <button class="btn btn-primary btn-big" id="fs-next"><i class="fa-solid fa-forward"></i> Next problem</button>
            ${it ? `<div class="fs-item"><img src="${esc(it.path)}" alt=""><div>${pill(it.curLabel)} ${it.pred ? `model: <b>${esc(it.pred)}</b> ${pct(it.pred_conf)}` : ''}</div>
                <div>${it.cluster !== null && it.cluster !== undefined ? `Cluster C${it.cluster} · ` : ''}${esc(it.camera || '')} · ${esc(it.period || '')}</div>
                <div class="pp-hint">What to do: ${esc(pointHint(it))}</div></div>` : '<p class="meter-desc">Click a point to see it here.</p>'}
            <p class="meter-desc">Keys: N next problem · ←/→ prev/next · Esc close · ? help</p>`;
        el.querySelector('#fs-next').onclick = nextProblem;
    }

    // ==========================================================================
    // LASSO SELECTION (2D and 3D: uses the projected screen positions)
    // ==========================================================================

    function pointInPolygon(x, y, poly) {
        let inside = false;
        for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
            const [xi, yi] = poly[i], [xj, yj] = poly[j];
            if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
        }
        return inside;
    }

    function finishLasso() {
        const poly = lassoPts;
        lassoPts = null;
        if (poly && poly.length >= 3) {
            const xs = poly.map(p => p[0]), ys = poly.map(p => p[1]);
            const w = Math.max(...xs) - Math.min(...xs), h = Math.max(...ys) - Math.min(...ys);
            // A short straight drag is treated as a box (rectangle) select
            const area = poly.length < 6 ? null : poly;
            const rect = [[Math.min(...xs), Math.min(...ys)], [Math.max(...xs), Math.min(...ys)], [Math.max(...xs), Math.max(...ys)], [Math.min(...xs), Math.max(...ys)]];
            const shape = area || rect;
            if (w > 3 || h > 3) {
                reportData.items.forEach((it, i) => {
                    const p = projCache[i];
                    if (!p || it.curDeleted || !isVisible(it)) return;
                    if (pointInPolygon(p[0], p[1], shape)) lassoSel.add(i);
                });
            }
        }
        updateSelectionBar();
        drawCanvas();
    }

    function updateSelectionBar() {
        const bar = document.getElementById('plot-selection');
        if (!bar) return;
        if (!lassoSel.size) { bar.classList.add('hidden'); return; }
        const counts = {};
        lassoSel.forEach(i => { const it = reportData.items[i]; counts[it.curLabel] = (counts[it.curLabel] || 0) + 1; });
        const mix = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([c, n]) => `${esc(c)} ${n}`).join(', ');
        document.getElementById('ps-count').innerHTML = `<b>${lassoSel.size}</b> selected (${mix})`;
        document.getElementById('ps-relabel').innerHTML = classOptions();
        bar.classList.remove('hidden');
    }

    function clearSelection() {
        lassoSel.clear();
        updateSelectionBar();
        drawCanvas();
    }

    function bulkDecide(kind, newLabel) {
        const ids = [];
        const snap = takeSnap([...lassoSel].map(i => reportData.items[i] && reportData.items[i].id).filter(Boolean));
        lassoSel.forEach(i => {
            const it = reportData.items[i];
            if (!it || it.removed) return;
            ids.push(it.id);
            if (kind === 'approve') {
                deletedIds.delete(it.id); keptIds.add(it.id);
                registerAction({ type: 'keep', id: it.id, description: `Confirmed '${it.curLabel}' (map selection).` });
            } else if (kind === 'remove') {
                deletedIds.add(it.id); keptIds.delete(it.id);
                registerAction({ type: 'delete', id: it.id, description: 'Deleted box (map selection).' });
            } else if (newLabel && newLabel !== it.curLabel) {
                if (newLabel === it.label) relabeledLabels.delete(it.id); else relabeledLabels.set(it.id, newLabel);
                registerAction({ type: 'relabel', id: it.id, old_label: it.curLabel, new_label: newLabel, description: `Relabeled ${it.id}: '${it.curLabel}' → '${newLabel}' (map selection).` });
            }
        });
        if (ids.length) reviewUndo.push(ids);
        lassoSel.clear();
        recomputeCuration();
        updateSelectionBar();
        if (ids.length) showToast(`${ids.length} boxes ${kind === 'approve' ? 'approved' : kind === 'remove' ? 'marked for removal' : `relabeled to ${newLabel}`}.`, snap);
    }

    function setupSelectionBar() {
        const bar = document.getElementById('plot-selection');
        if (!bar) return;
        document.getElementById('ps-approve').onclick = () => bulkDecide('approve');
        document.getElementById('ps-remove').onclick = () => bulkDecide('remove');
        document.getElementById('ps-relabel-btn').onclick = () => bulkDecide('relabel', document.getElementById('ps-relabel').value);
        document.getElementById('ps-clear').onclick = clearSelection;
        document.getElementById('btn-plot-color').addEventListener('click', () => {
            if (!hasClusters()) return;
            colorBy = colorBy === 'class' ? 'cluster' : 'class';
            if (colorBy === 'class' && clusterFilter !== null) clusterFilter = null;
            renderPlotLegend();
            renderClusterGrid();
            drawCanvas();
        });
    }

    // ==========================================================================
    // PINNED POINT WINDOW (click on the map): draggable, crop + full frame, prev/next
    // ==========================================================================

    let pointPopupIdx = -1;

    function pointHint(it) {
        if (it.curDeleted) return 'Marked for removal. Undo to keep it.';
        if (relabeledLabels.has(it.id)) return `Relabeled ${it.label} → ${it.curLabel}. Nothing else to do.`;
        if (it.curKept) return 'Approved. Nothing else to do.';
        if (it.is_mislabeled) return `Model thinks ${it.pred} (${pct(it.pred_conf)}): relabel to ${it.pred} if it is one, else approve.`;
        if (it.has_issue) return `${it.issues[0]}: remove the bad box.`;
        if (it.is_outlier) return 'Unusual box: approve if it is a real ' + it.curLabel + ', else remove.';
        if (it.is_duplicate) return 'Near-duplicate of another box: remove one copy.';
        return 'Looks fine: approve or skip.';
    }

    function popNavList() {
        const cur = reportData.items[pointPopupIdx];
        let list = reportData.items.filter(it => isVisible(it) && !it.removed);
        if (popNavMode === 'flagged') list = list.filter(it => isFlagged(it) && (!itemDecision(it) || it === cur));
        else if (cur && cur.cluster !== null && cur.cluster !== undefined) list = list.filter(it => it.cluster === cur.cluster);
        return list.map(it => it.index);
    }

    function popStep(d) {
        const list = popNavList();
        if (!list.length) return;
        let pos = list.indexOf(pointPopupIdx);
        if (pos < 0) pos = d > 0 ? -1 : 0;
        const next = list[(pos + d + list.length) % list.length];
        selectedIndex = next;
        openPointPopup(next);
        drawCanvas();
    }

    function openPointPopup(idx) {
        const it = reportData.items[idx];
        if (!it) return;
        const wasOpen = pointPopupIdx >= 0;
        pointPopupIdx = idx;
        updateHoverPreview(0, 0, null);
        const pop = document.getElementById('point-pop');
        if (!wasOpen && !popPos) {
            const W = canvas.clientWidth, PW = Math.min(560, W - 16);
            pop.style.left = `${Math.max(8, W - PW - 12)}px`;
            pop.style.top = '12px';
        }
        refreshPointPopup();
        pop.classList.remove('hidden');
        updateFsSide();
    }

    function refreshPointPopup() {
        const it = reportData && reportData.items[pointPopupIdx];
        if (!it) return;
        const pop = document.getElementById('point-pop');
        const others = classNames.filter(c => c !== it.curLabel);
        const def = it.pred && it.pred !== it.curLabel ? it.pred : others[0];
        const list = popNavList();
        const pos = list.indexOf(pointPopupIdx);
        const dec = itemDecision(it);
        pop.innerHTML = `
            <div class="pp-head" title="Drag to move">
                <i class="fa-solid fa-grip-vertical"></i>
                <span class="pp-title">${esc(it.image)}</span>
                <button class="pp-close" title="Close (Esc)">×</button>
            </div>
            <div class="pp-nav">
                <button class="btn btn-small" data-nav="-1" title="Previous (←)"><i class="fa-solid fa-chevron-left"></i></button>
                <span>${pos >= 0 ? pos + 1 : '–'} / ${list.length}</span>
                <button class="btn btn-small" data-nav="1" title="Next (→)"><i class="fa-solid fa-chevron-right"></i></button>
                <select class="pp-mode" title="What prev/next walks through">
                    <option value="cluster" ${popNavMode === 'cluster' ? 'selected' : ''}>same cluster</option>
                    <option value="flagged" ${popNavMode === 'flagged' ? 'selected' : ''}>flagged, unreviewed</option>
                </select>
            </div>
            <div class="pp-imgs">
                <figure><img class="pp-crop" src="${esc(it.path)}" alt="crop"><figcaption>Box</figcaption></figure>
                <figure class="pp-frame-fig"><div class="pp-frame-wrap"><img class="pp-frame" src="/api/frame?id=${encodeURIComponent(it.id)}" alt="frame"></div><figcaption>Full frame</figcaption></figure>
            </div>
            <div class="pp-line">Label ${pill(it.curLabel)} <span>Model guess: <b>${esc(it.pred || 'n/a')}</b> ${it.pred ? pct(it.pred_conf) : ''}</span></div>
            <div class="pp-line">${it.cluster !== null && it.cluster !== undefined ? `Cluster <b>C${it.cluster}</b> · ` : ''}${esc(it.split)} · ${esc(it.camera || '')} · ${esc(it.period || '')}${dec ? ` · <b style="color:${dec[1]}">${esc(dec[0])}</b>` : ''}</div>
            <div class="pp-hint"><b>What to do:</b> ${esc(pointHint(it))}</div>
            <div class="pp-actions">
                <button class="btn pp-ok" data-pp="approve" title="Approve"><i class="fa-solid fa-check"></i> Approve</button>
                <button class="btn pp-del" data-pp="remove" title="Remove box"><i class="fa-solid fa-trash"></i> Remove</button>
                <select class="pp-select">${others.map(c => `<option value="${esc(c)}" ${c === def ? 'selected' : ''}>${esc(c)}</option>`).join('')}</select>
                <button class="btn" data-pp="relabel" title="Relabel">Relabel</button>
                <button class="btn" data-pp="undo" title="Undo decision" ${dec ? '' : 'disabled'}><i class="fa-solid fa-rotate-left"></i> Undo</button>
            </div>`;
        const frame = pop.querySelector('.pp-frame');
        frame.onerror = () => { pop.querySelector('.pp-frame-fig').innerHTML = '<p class="meter-desc">Full frame needs the CleanCrop server.</p>'; };
        pop.querySelector('.pp-close').onclick = closePointPopup;
        pop.querySelector('.pp-mode').onchange = e => { popNavMode = e.target.value; refreshPointPopup(); };
        pop.querySelectorAll('[data-nav]').forEach(b => b.onclick = () => popStep(+b.dataset.nav));
        pop.querySelector('.pp-frame-wrap').onclick = () => openFrame(it.id);
        pop.querySelectorAll('[data-pp]').forEach(b => b.onclick = () => {
            const act = b.dataset.pp;
            if (act === 'approve') { reviewUndo.push([it.id]); approveCrop(it.id, `Confirmed '${it.curLabel}' from the similarity map.`); }
            else if (act === 'remove') { reviewUndo.push([it.id]); excludeCrop(it.id, 'Deleted box from the similarity map.'); }
            else if (act === 'relabel') { const nl = pop.querySelector('.pp-select').value; if (nl && nl !== it.curLabel) { reviewUndo.push([it.id]); relabelCrop(it.id, it.curLabel, nl); } }
            else if (act === 'undo') undoCrop(it.id);
            refreshPointPopup();
            updateFsSide();
        });
    }

    function closePointPopup() {
        const pop = document.getElementById('point-pop');
        if (pop) pop.classList.add('hidden');
        pointPopupIdx = -1;
        updateFsSide();
        drawCanvas();
    }

    function setupPopupDrag() {
        const pop = document.getElementById('point-pop');
        let start = null;
        pop.addEventListener('mousedown', e => {
            if (!e.target.closest('.pp-head') || e.target.closest('button')) return;
            start = [e.clientX, e.clientY, pop.offsetLeft, pop.offsetTop];
            e.preventDefault();
        });
        window.addEventListener('mousemove', e => {
            if (!start) return;
            const host = pop.offsetParent || document.body;
            const x = Math.max(0, Math.min(host.clientWidth - 60, start[2] + e.clientX - start[0]));
            const y = Math.max(0, Math.min(host.clientHeight - 30, start[3] + e.clientY - start[1]));
            pop.style.left = `${x}px`; pop.style.top = `${y}px`;
            popPos = [x, y];
        });
        window.addEventListener('mouseup', () => { start = null; });
    }

    // ==========================================================================
    // TRAINING READY PAGE
    // ==========================================================================

    function renderReadyTab() {
        const el = document.getElementById('ready-content');
        if (!el) return;
        const c = reviewCounts();
        const nBg = (reportData.backgrounds || []).length;
        const nImg = reportData.stats ? reportData.stats.images : 0;
        const bgPct = nImg ? 100 * nBg / nImg : 0;
        const golden = reportData.golden_test ? reportData.golden_test.images : 0;
        const pending = deletedIds.size + relabeledLabels.size;
        const checks = [
            [c.left === 0, `All flagged boxes reviewed (${c.done}/${c.total})`, c.left ? `${c.left} still to review: use Next problem.` : 'Done.'],
            [bgPct >= 5 && bgPct <= 15, `Backgrounds ${bgPct.toFixed(1)}% (${nBg} images)`, 'Aim for about 5–15% images with no boxes.'],
            [golden > 0, `Golden test set: ${golden} images`, golden ? 'Kept out of training as the fixed test split.' : 'None yet: run the analysis with --golden-split 0.15.'],
            [(reportData.summary.issue_count || 0) === 0 || reviewCounts().left === 0, `Broken / double boxes: ${reportData.summary.issue_count || 0}`, 'Fix or remove them before training.'],
            [true, `${pending} pending changes (${deletedIds.size} removals, ${relabeledLabels.size} relabels)`, 'The export applies them to a copy; the original dataset is not touched.'],
        ];
        el.innerHTML = `
            <div class="ready-stats">
                <div class="ready-stat"><div class="rs-num">${bgPct.toFixed(1)}%</div><div>background images</div></div>
                <div class="ready-stat ${c.left ? 'warn' : 'ok'}"><div class="rs-num">${c.left}</div><div>unreviewed flags</div></div>
                <div class="ready-stat"><div class="rs-num">${golden}</div><div>golden test images</div></div>
            </div>
            <div class="ready-checks">${checks.map(([ok, t, why]) => `<div class="readiness-row"><span class="st ${ok ? 'pass' : 'warn'}">${ok ? 'OK' : 'CHECK'}</span><div><div>${esc(t)}</div><div class="why">${esc(why)}</div></div></div>`).join('')}</div>
            <button class="btn btn-primary btn-big" id="btn-export-clean" ${serverAvailable ? '' : 'disabled title="Needs the CleanCrop server"'}><i class="fa-solid fa-box-archive"></i> Export clean dataset</button>
            <div id="export-clean-result" class="meter-desc">Writes a new folder <code>clean_export_&lt;time&gt;</code> next to the report. Nothing in the source dataset changes.</div>`;
        el.querySelector('#btn-export-clean').onclick = exportCleanDataset;
    }

    async function flushState() {
        clearTimeout(saveTimer);
        await fetch('/api/state', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                deleted: [...deletedIds], relabeled: Object.fromEntries(relabeledLabels), kept: [...keptIds],
                ignoredDuplicates: [...ignoredDuplicates], log: actionsLog,
                thresholds: { dup: dupThreshold, lof: lofThreshold, mis: mislabelConf },
            }),
        });
    }

    async function exportCleanDataset() {
        const btn = document.getElementById('btn-export-clean');
        const out = document.getElementById('export-clean-result');
        btn.disabled = true;
        out.textContent = 'Exporting…';
        try {
            await flushState();
            const r = await fetch('/api/export_clean', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
            const j = await r.json();
            if (!r.ok || j.status !== 'ok') throw new Error(j.message || `HTTP ${r.status}`);
            out.innerHTML = `<b>Exported</b> to <code>${esc(j.out_dir)}</code>` +
                Object.entries(j).filter(([k, v]) => !['status', 'out_dir'].includes(k) && typeof v !== 'object').map(([k, v]) => ` · ${esc(k)}: ${esc(String(v))}`).join('');
            showToast('Clean dataset exported.');
        } catch (e) {
            out.innerHTML = `<span class="hint-warn">Export failed: ${esc(e.message)}</span>`;
        } finally {
            btn.disabled = false;
        }
    }

    // ==========================================================================
    // EXTRAS: filters bar, theme, help overlay, keys
    // ==========================================================================

    function setTheme(t) {
        document.documentElement.dataset.theme = t;
        try { localStorage.setItem('cc-theme', t); } catch (e) { /* storage blocked */ }
        const b = document.getElementById('btn-theme');
        if (b) b.innerHTML = t === 'light' ? '<i class="fa-solid fa-moon"></i>' : '<i class="fa-solid fa-sun"></i>';
        readThemeColors();
        drawCanvas();
    }

    function toggleHelp(show) {
        const h = document.getElementById('help-overlay');
        const on = show === undefined ? h.classList.contains('hidden') : show;
        const host = document.fullscreenElement || document.body;
        if (h.parentElement !== host) host.appendChild(h);
        h.classList.toggle('hidden', !on);
    }

    function setupUiExtras() {
        let theme = 'dark';
        try { theme = localStorage.getItem('cc-theme') || 'dark'; } catch (e) { /* storage blocked */ }
        setTheme(theme);
        document.getElementById('btn-theme').onclick = () => setTheme(document.documentElement.dataset.theme === 'light' ? 'dark' : 'light');
        ['gf-class', 'gf-camera', 'gf-period', 'gf-flagged', 'gf-unreviewed'].forEach(id => document.getElementById(id).addEventListener('change', applyFilters));
        document.getElementById('gf-reset').onclick = () => {
            ['gf-class', 'gf-camera', 'gf-period'].forEach(id => { document.getElementById(id).value = ''; });
            document.getElementById('gf-flagged').checked = false;
            document.getElementById('gf-unreviewed').checked = false;
            clusterFilter = null;
            applyFilters();
        };
        document.getElementById('btn-next-problem').onclick = nextProblem;
        document.getElementById('btn-help').onclick = () => toggleHelp();
        document.getElementById('help-overlay').addEventListener('click', e => { if (e.target.id === 'help-overlay' || e.target.closest('.help-close')) toggleHelp(false); });
        document.addEventListener('keydown', e => {
            if (e.target instanceof Element && e.target.matches('input, select, textarea')) return;
            const help = !document.getElementById('help-overlay').classList.contains('hidden');
            if (help && e.key === 'Escape') { toggleHelp(false); e.stopImmediatePropagation(); e.preventDefault(); return; }
            if (e.ctrlKey || e.metaKey || e.altKey) return;
            if (e.key === '?') { toggleHelp(); e.preventDefault(); return; }
            const modalOpen = !document.getElementById('frame-modal').classList.contains('hidden')
                || !document.getElementById('point-modal').classList.contains('hidden');
            if (modalOpen || (activeTab !== 'overview' && activeTab !== 'explorer')) return;
            if (e.key.toLowerCase() === 'n') { nextProblem(); e.preventDefault(); e.stopImmediatePropagation(); return; }
            if (pointPopupIdx >= 0 && (e.key === 'ArrowRight' || e.key === 'ArrowLeft')) {
                popStep(e.key === 'ArrowRight' ? 1 : -1); e.preventDefault(); e.stopImmediatePropagation();
            }
        }, true);
    }

    // --- Audit file export ---
    function exportCurationReport() {
        if (!reportData) return;
        const auditLog = {
            export_timestamp: new Date().toISOString(),
            dataset: reportData.dataset_dir,
            summary: {
                boxes: reportData.items.filter(it => !it.removed).length,
                to_delete: deletedIds.size,
                to_relabel: relabeledLabels.size,
                confirmed_correct: keptIds.size,
            },
            actions: actionsLog,
        };
        const blob = new Blob([JSON.stringify(auditLog, null, 2)], { type: 'application/json' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `cleancrop_review_${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(a.href);
    }

    // --- Write decisions into the YOLO label files ---
    function commitWriteback() {
        if (!reportData) return;
        if (deletedIds.size === 0 && relabeledLabels.size === 0) {
            alert("Nothing to save yet: delete or relabel some boxes first.");
            return;
        }
        const msg = `Save to the dataset label files?\n\n` +
            `• Delete ${deletedIds.size} box(es)\n• Relabel ${relabeledLabels.size} box(es)\n\n` +
            `Dataset: ${reportData.dataset_dir}\n` +
            `A backup of every changed label file is kept in the curation output folder (label_backups).`;
        if (!confirm(msg)) return;

        const actions = [];
        deletedIds.forEach(id => {
            const item = itemById(id);
            if (item && !item.removed) actions.push({ id, action: 'DELETE' });
        });
        relabeledLabels.forEach((newLabel, id) => {
            if (deletedIds.has(id)) return;
            const classId = classNames.indexOf(newLabel);
            if (classId < 0) return;
            actions.push({ id, action: 'RELABEL', new_class_id: classId });
        });

        const btn = document.getElementById('btn-commit-writeback');
        btn.disabled = true;
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Saving...';

        fetch('/api/writeback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ actions }),
        })
            .then(r => r.json())
            .then(data => {
                let text = data.message || 'Done.';
                if (data.backup_dir) text += `\n\nBackup of the original label files:\n${data.backup_dir}`;
                if (data.errors && data.errors.length) text += `\n\nProblems:\n- ${data.errors.slice(0, 10).join('\n- ')}`;
                alert(text);
                if (data.status !== 'error') location.reload();
            })
            .catch(err => alert(`Saving failed. Is the CleanCrop window still open?\n${err.message}`))
            .finally(() => {
                btn.disabled = false;
                btn.innerHTML = '<i class="fa-solid fa-floppy-disk"></i> Save Changes to Dataset';
            });
    }

    function showErrorState(msg) {
        document.querySelector('.content-scroll').innerHTML = `
            <div class="empty-state-msg" style="height:400px;grid-column:1/-1;">
                <i class="fa-solid fa-triangle-exclamation" style="font-size:48px;color:var(--color-outlier);"></i>
                <h2>Nothing to show</h2>
                <p>${esc(msg)}</p>
            </div>`;
    }

})();
