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

        mislabelList = items.filter(it => it.is_mislabeled).sort((a, b) => b.pred_conf - a.pred_conf);
        outlierList = items.filter(it => it.is_outlier).sort((a, b) => b.lof - a.lof);
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

        let filtered = reportData.items.filter(it => !it.removed);
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

    function excludeCrop(id, description = "") {
        deletedIds.add(id);
        keptIds.delete(id);
        registerAction({ type: 'delete', id, description: description || `Deleted ${id}.` });
        recomputeCuration();
    }

    function approveCrop(id, description = "") {
        keptIds.add(id);
        registerAction({ type: 'keep', id, description: description || `Confirmed ${id}.` });
        recomputeCuration();
    }

    function relabelCrop(id, oldLabel, newLabel) {
        const item = itemById(id);
        if (item && newLabel === item.label) relabeledLabels.delete(id);
        else relabeledLabels.set(id, newLabel);
        registerAction({ type: 'relabel', id, old_label: oldLabel, new_label: newLabel, description: `Relabeled ${id}: '${oldLabel}' → '${newLabel}'.` });
        recomputeCuration();
    }

    function undoCrop(id) {
        deletedIds.delete(id);
        relabeledLabels.delete(id);
        keptIds.delete(id);
        for (let i = actionsLog.length - 1; i >= 0; i--) {
            if (actionsLog[i].id === id) actionsLog.splice(i, 1);
        }
        saveState();
        recomputeCuration();
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
    
    function getProjectedCoord(idx) {
        const coords = pcaCoordinates[idx];
        if (!coords) return [0, 0, 0];
        const x = coords[0];
        const y = coords[1];
        const z = coords[2] || 0;
        
        if (!is3D) {
            const m = map2dCoordinates[idx];
            return m ? [m[0], m[1], 0] : [x, y, 0];
        }
        
        const cosY = Math.cos(rotationY);
        const sinY = Math.sin(rotationY);
        const cosX = Math.cos(rotationX);
        const sinX = Math.sin(rotationX);

        // Rotate Y
        const x1 = x * cosY - z * sinY;
        const z1 = x * sinY + z * cosY;

        // Rotate X
        const y2 = y * cosX - z1 * sinX;
        const z2 = y * sinX + z1 * cosX;
        
        return [x1, y2, z2];
    }
    
    function resetCanvasView() {
        if (pcaCoordinates.length === 0) return;
        
        // Find dimensions min/max
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        
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
        
        // Scale to fit canvas
        const scaleX = width / (maxX - minX);
        const scaleY = height / (maxY - minY);
        viewZoom = Math.min(scaleX, scaleY);
        
        // Pan to center
        viewPanX = width / 2 - (minX + maxX) / 2 * viewZoom;
        viewPanY = height / 2 - (minY + maxY) / 2 * viewZoom;
        
        drawCanvas();
    }
    
    function drawCanvas() {
        if (!reportData || pcaCoordinates.length === 0) return;
        
        const N = pcaCoordinates.length;
        
        // Adjust Canvas size for high-DPI displays
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        ctx.scale(dpr, dpr);
        
        const width = rect.width;
        const height = rect.height;
        
        // Clear background
        ctx.fillStyle = '#090c10';
        ctx.fillRect(0, 0, width, height);
        
        // Draw grid lines
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
        ctx.lineWidth = 1;
        const gridSize = 40 * viewZoom;
        
        const startX = viewPanX % gridSize;
        for (let x = startX; x < width; x += gridSize) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        }
        
        const startY = viewPanY % gridSize;
        for (let y = startY; y < height; y += gridSize) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }
        
        const items = reportData.items;
        
        // Calculate all projected 2D coordinates
        const projectedCoords = pcaCoordinates.map((_, i) => getProjectedCoord(i));
        
        // Compute minZ/maxZ for depth cues
        let minZ = Infinity, maxZ = -Infinity;
        if (is3D) {
            projectedCoords.forEach(coord => {
                const z = coord[2];
                if (z < minZ) minZ = z;
                if (z > maxZ) maxZ = z;
            });
        }
        
        // Determine rendering order (depth sorting)
        const renderOrder = [];
        for (let i = 0; i < N; i++) renderOrder.push(i);
        if (is3D) {
            renderOrder.sort((a, b) => projectedCoords[a][2] - projectedCoords[b][2]);
        }
        
        // First Draw: Anomaly Rings/Halos (Bottom Layer)
        if (showAnomalyRings) {
            renderOrder.forEach(i => {
                const item = items[i];
                if (item.curDeleted) return; // Hide rings for deleted
                
                const [px, py, pz] = projectedCoords[i];
                
                // Screen projection coordinates
                const sx = px * viewZoom + viewPanX;
                const sy = py * viewZoom + viewPanY;
                
                if (sx < -20 || sx > width + 20 || sy < -20 || sy > height + 20) return;
                
                let depthFactor = 1.0;
                if (is3D && maxZ > minZ) {
                    depthFactor = 0.4 + 0.8 * ((pz - minZ) / (maxZ - minZ));
                }
                
                ctx.save();
                if (is3D) {
                    ctx.globalAlpha = Math.max(0.1, depthFactor * 0.8);
                }
                
                if (item.is_mislabeled) {
                    drawGlowingHalo(sx, sy, 14 * depthFactor, 'rgba(245, 158, 11, 0.35)', 'rgba(245, 158, 11, 0)');
                } else if (item.is_outlier || item.has_issue) {
                    drawGlowingHalo(sx, sy, 14 * depthFactor, 'rgba(244, 63, 94, 0.25)', 'rgba(244, 63, 94, 0)');
                } else if (item.is_duplicate) {
                    drawGlowingHalo(sx, sy, 14 * depthFactor, 'rgba(0, 242, 254, 0.25)', 'rgba(0, 242, 254, 0)');
                }
                
                ctx.restore();
            });
        }
        
        // Second Draw: Points Core
        renderOrder.forEach(i => {
            const item = items[i];
            if (item.removed) return; // already deleted from the label files
            const [px, py, pz] = projectedCoords[i];
            const sx = px * viewZoom + viewPanX;
            const sy = py * viewZoom + viewPanY;
            
            if (sx < -10 || sx > width + 10 || sy < -10 || sy > height + 10) return;
            
            const color = pointColor(item);
            
            let depthFactor = 1.0;
            if (is3D && maxZ > minZ) {
                depthFactor = 0.4 + 0.8 * ((pz - minZ) / (maxZ - minZ));
            }
            
            ctx.save();
            if (is3D && !item.curDeleted) {
                ctx.globalAlpha = Math.max(0.15, depthFactor);
            }
            
            ctx.beginPath();
            
            if (item.curDeleted) {
                // Render excluded items as small faded gray crosses
                ctx.strokeStyle = 'rgba(244, 63, 94, 0.7)';
                ctx.lineWidth = 1.5;
                const size = 4 * depthFactor;
                ctx.moveTo(sx - size, sy - size);
                ctx.lineTo(sx + size, sy + size);
                ctx.moveTo(sx + size, sy - size);
                ctx.lineTo(sx - size, sy + size);
                ctx.stroke();
            } else {
                // Active points
                const isSelected = selectedIndex === i;
                const isHovered = hoveredIndex === i;
                
                let radius = 5 * depthFactor;
                if (isHovered) radius = 8 * depthFactor;
                if (isSelected) radius = 9 * depthFactor;
                
                ctx.arc(sx, sy, radius, 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.fill();
                
                // Active outline borders
                if (isSelected) {
                    ctx.strokeStyle = '#ffffff';
                    ctx.lineWidth = 2.5;
                    ctx.stroke();
                    
                    // Outer accent ring
                    ctx.beginPath();
                    ctx.arc(sx, sy, radius + 4 * depthFactor, 0, Math.PI * 2);
                    ctx.strokeStyle = color;
                    ctx.lineWidth = 1;
                    ctx.stroke();
                } else if (isHovered) {
                    ctx.strokeStyle = '#ffffff';
                    ctx.lineWidth = 1.5;
                    ctx.stroke();
                } else {
                    ctx.strokeStyle = 'rgba(0, 0, 0, 0.4)';
                    ctx.lineWidth = 1;
                    ctx.stroke();
                }
                // Review decision mark: green ring = approved, amber square = relabeled
                if (relabeledLabels.has(item.id)) {
                    const q = radius + 3;
                    ctx.strokeStyle = '#f59e0b';
                    ctx.lineWidth = 2;
                    ctx.strokeRect(sx - q, sy - q, q * 2, q * 2);
                } else if (item.curKept) {
                    ctx.beginPath();
                    ctx.arc(sx, sy, radius + 3, 0, Math.PI * 2);
                    ctx.strokeStyle = '#10b981';
                    ctx.lineWidth = 2;
                    ctx.stroke();
                }
            }
            
            ctx.restore();
        });
        
        // Map selection rings + lasso path
        if (lassoSel.size) {
            ctx.save();
            ctx.strokeStyle = '#fde047';
            ctx.lineWidth = 2;
            lassoSel.forEach(i => {
                if (!projectedCoords[i] || items[i].curDeleted) return;
                const sx = projectedCoords[i][0] * viewZoom + viewPanX;
                const sy = projectedCoords[i][1] * viewZoom + viewPanY;
                ctx.beginPath();
                ctx.arc(sx, sy, 8, 0, Math.PI * 2);
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

        // Third Draw: Tooltip Overlay (Top Layer)
        if (hoveredIndex >= 0 && hoveredIndex < N) {
            const item = items[hoveredIndex];
            if (!item.curDeleted) {
                const [px, py] = projectedCoords[hoveredIndex];
                const sx = px * viewZoom + viewPanX;
                const sy = py * viewZoom + viewPanY;
                updateHoverPreview(sx, sy, item);
            }
        }
        if (hoveredIndex < 0 || hoveredIndex >= N || items[hoveredIndex].curDeleted) updateHoverPreview(0, 0, null);
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
        // Drag to Pan or Rotate
        canvas.addEventListener('mousedown', (e) => {
            if (e.shiftKey && !is3D) {  // shift+drag = lasso select (2D map)
                const r = canvas.getBoundingClientRect();
                lassoPts = [[e.clientX - r.left, e.clientY - r.top]];
                if (!(e.ctrlKey || e.metaKey)) lassoSel.clear();
                closePointPopup();
                e.preventDefault();
                return;
            }
            isDrawing = true;
            dragStart.x = e.clientX;
            dragStart.y = e.clientY;
            dragPanStart.x = viewPanX;
            dragPanStart.y = viewPanY;
            dragRotationStart.x = rotationX;
            dragRotationStart.y = rotationY;
        });
        
        window.addEventListener('mousemove', (e) => {
            const rect = canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            
            if (lassoPts) {
                lassoPts.push([mouseX, mouseY]);
                drawCanvas();
                return;
            }
            if (isDrawing) {
                const dx = e.clientX - dragStart.x;
                const dy = e.clientY - dragStart.y;
                if (is3D) {
                    // Update 3D rotation based on mouse movement
                    rotationY = dragRotationStart.y + dx * 0.007;
                    rotationX = dragRotationStart.x + dy * 0.007;
                } else {
                    // Calculate Panned offset
                    viewPanX = dragPanStart.x + dx;
                    viewPanY = dragPanStart.y + dy;
                }
                drawCanvas();
            } else {
                // Hover detection
                if (pcaCoordinates.length === 0) return;
                
                let prevHovered = hoveredIndex;
                hoveredIndex = -1;
                
                let minDist = 10; // Click radius
                
                pcaCoordinates.forEach((_, idx) => {
                    const item = reportData.items[idx];
                    if (item.curDeleted) return; // Ignore hovered excluded
                    
                    const [px, py] = getProjectedCoord(idx);
                    const sx = px * viewZoom + viewPanX;
                    const sy = py * viewZoom + viewPanY;
                    
                    const dx = mouseX - sx;
                    const dy = mouseY - sy;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    
                    if (dist < minDist) {
                        minDist = dist;
                        hoveredIndex = idx;
                    }
                });
                
                if (hoveredIndex !== prevHovered) {
                    drawCanvas();
                }
            }
        });
        
        window.addEventListener('mouseup', (e) => {
            if (lassoPts) {
                finishLasso();
                suppressClick = true;
                return;
            }
            if (isDrawing) lastDragDist = Math.hypot(e.clientX - dragStart.x, e.clientY - dragStart.y);
            isDrawing = false;
        });
        
        // Scroll Wheel to Zoom
        canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            
            const rect = canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            
            // Calculate mouse position relative to data space before zoom
            const [dataX, dataY] = is3D ? [0, 0] : [
                (mouseX - viewPanX) / viewZoom,
                (mouseY - viewPanY) / viewZoom
            ];
            
            const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
            viewZoom = Math.max(0.1, Math.min(50, viewZoom * zoomFactor));
            
            if (!is3D) {
                // Reposition pan so mouse point remains static
                viewPanX = mouseX - dataX * viewZoom;
                viewPanY = mouseY - dataY * viewZoom;
            }
            
            drawCanvas();
        }, { passive: false });
        
        // Canvas Toolbar Controls
        document.getElementById('btn-plot-reset').addEventListener('click', () => {
            if (is3D) {
                rotationX = -0.5;
                rotationY = 0.6;
            }
            resetCanvasView();
        });
        
        document.getElementById('btn-plot-toggle-flags').addEventListener('click', (e) => {
            showAnomalyRings = !showAnomalyRings;
            e.target.classList.toggle('active', showAnomalyRings);
            drawCanvas();
        });
        
        // 3D Mode Toggle Listener
        document.getElementById('btn-plot-toggle-3d').addEventListener('click', (e) => {
            is3D = !is3D;
            e.currentTarget.classList.toggle('active', is3D);
            closePointPopup();
            
            const cardTitle = document.getElementById('plot-card-title');
            const instructions = document.getElementById('plot-instructions');
            
            if (is3D) {
                cardTitle.innerHTML = '<i class="fa-solid fa-cube"></i> Interactive 3D Embedding Space (PCA Projection)';
                instructions.innerHTML = '<i class="fa-solid fa-arrows-spin"></i> Drag to rotate | <i class="fa-solid fa-magnifying-glass-plus"></i> Scroll to zoom | Click point to inspect';
            } else {
                cardTitle.innerHTML = '<i class="fa-solid fa-circle-nodes"></i> Interactive 2D Embedding Space (t-SNE)';
                instructions.innerHTML = '<i class="fa-solid fa-arrows-up-down-left-right"></i> Drag to pan | <i class="fa-solid fa-magnifying-glass-plus"></i> Scroll to zoom | Click point to inspect | Shift+drag to select';
            }
            
            resetCanvasView();
        });
        
        // Click to Select item
        canvas.addEventListener('click', (e) => {
            if (suppressClick) { suppressClick = false; return; }  // end of a lasso
            if (lastDragDist > 3) { lastDragDist = 0; return; }   // it was a pan / rotation drag
            if (hoveredIndex >= 0) {
                selectedIndex = hoveredIndex;
                openPointPopup(hoveredIndex);
                drawCanvas();
            } else {
                closePointPopup();
            }
        });

        canvas.addEventListener('mouseleave', () => {
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
        // click anywhere outside the small popup closes it (canvas clicks are handled by the canvas)
        document.addEventListener('mousedown', e => {
            const pop = document.getElementById('point-pop');
            if (pointPopupIdx >= 0 && pop && !pop.contains(e.target) && e.target !== canvas) closePointPopup();
        });
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
            if (it.curDeleted) deletedIds.delete(it.id);
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
            if (it.curDeleted || it.cluster === null || it.cluster === undefined) return;
            if (!groups.has(it.cluster)) groups.set(it.cluster, []);
            groups.get(it.cluster).push(it);
        });
        return [...groups.entries()].sort((a, b) => a[0] - b[0]).map(([c, g]) => {
            const counts = {};
            g.forEach(it => { counts[it.curLabel] = (counts[it.curLabel] || 0) + 1; });
            const [major, nMajor] = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
            const purity = nMajor / g.length;
            const unusual = g.filter(it => it.lof >= lofThreshold).length / g.length;
            const hint = purity < 0.7 ? 'mixed: check labels'
                : unusual >= 0.3 ? 'many unusual boxes: background-like?'
                : purity < 0.9 ? 'check the minority boxes' : 'clean';
            return { cluster: c, size: g.length, major, purity, hint };
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
            el.innerHTML = clusterSummary().map(s => `<div class="legend-item cluster-legend" title="${esc(s.hint)}">
                <div class="legend-dot" style="background-color:${clusterColor(s.cluster)};"></div>
                <span>C${s.cluster} · ${s.size} · ${esc(s.major)} ${Math.round(s.purity * 100)}% · <i class="${s.hint === 'clean' ? 'hint-ok' : 'hint-warn'}">${esc(s.hint)}</i></span></div>`).join('');
        } else {
            el.innerHTML = Object.entries(classColors).map(([cls, color]) =>
                `<div class="legend-item"><div class="legend-dot" style="background-color:${color};"></div><span>${esc(cls)}</span></div>`).join('');
        }
    }

    // ==========================================================================
    // LASSO SELECTION ON THE 2D MAP
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
                    if (it.curDeleted) return;
                    const [px, py] = getProjectedCoord(i);
                    if (pointInPolygon(px * viewZoom + viewPanX, py * viewZoom + viewPanY, shape)) lassoSel.add(i);
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
            renderPlotLegend();
            drawCanvas();
        });
    }

    // ==========================================================================
    // SMALL POINT POPUP (click on the map)
    // ==========================================================================

    let pointPopupIdx = -1;

    function pointHint(it) {
        if (it.curDeleted) return 'Marked for removal. Undo to keep it.';
        if (relabeledLabels.has(it.id)) return `Relabeled ${it.label} → ${it.curLabel}.`;
        if (it.curKept) return 'Approved.';
        if (it.is_mislabeled) return `Model thinks ${it.pred} (${pct(it.pred_conf)}): relabel?`;
        if (it.has_issue) return `${it.issues[0]}: fix or remove.`;
        if (it.is_outlier) return 'Unusual box: check or remove.';
        if (it.is_duplicate) return 'Near-duplicate of another box: consider removing one.';
        return 'Looks fine: approve or skip.';
    }

    function openPointPopup(idx) {
        const it = reportData.items[idx];
        if (!it) return;
        pointPopupIdx = idx;
        updateHoverPreview(0, 0, null);
        const pop = document.getElementById('point-pop');
        const [px, py] = getProjectedCoord(idx);
        const sx = px * viewZoom + viewPanX, sy = py * viewZoom + viewPanY;
        const W = canvas.clientWidth, H = canvas.clientHeight, PW = 240, PH = 300;
        pop.style.left = `${Math.max(8, Math.min(sx + 14 + PW > W ? sx - PW - 14 : sx + 14, W - PW - 8))}px`;
        pop.style.top = `${Math.max(8, Math.min(sy - 40, H - PH - 8))}px`;
        refreshPointPopup();
        pop.classList.remove('hidden');
    }

    function refreshPointPopup() {
        const it = reportData && reportData.items[pointPopupIdx];
        if (!it) return;
        const pop = document.getElementById('point-pop');
        const others = classNames.filter(c => c !== it.curLabel);
        const def = it.pred && it.pred !== it.curLabel ? it.pred : others[0];
        pop.innerHTML = `
            <button class="pp-close" title="Close (Esc)">×</button>
            <img src="${esc(it.path)}" alt="">
            <div class="pp-line">${pill(it.curLabel)} <span>model: <b>${esc(it.pred || 'n/a')}</b> ${it.pred ? pct(it.pred_conf) : ''}</span></div>
            <div class="pp-line">${it.cluster !== null && it.cluster !== undefined ? `cluster <b>C${it.cluster}</b> · ` : ''}${esc(it.split)} · LOF ${it.lof.toFixed(2)}</div>
            <div class="pp-hint">${esc(pointHint(it))}</div>
            <div class="pp-actions">
                <button class="btn btn-small pp-ok" data-pp="approve" title="Approve"><i class="fa-solid fa-check"></i></button>
                <button class="btn btn-small pp-del" data-pp="remove" title="Remove box"><i class="fa-solid fa-trash"></i></button>
                <select class="pp-select">${others.map(c => `<option value="${esc(c)}" ${c === def ? 'selected' : ''}>${esc(c)}</option>`).join('')}</select>
                <button class="btn btn-small" data-pp="relabel" title="Relabel">Relabel</button>
                ${itemDecision(it) ? '<button class="btn btn-small" data-pp="undo" title="Undo decision"><i class="fa-solid fa-rotate-left"></i></button>' : ''}
            </div>
            <a href="#" class="pp-full">view full image</a>`;
        pop.querySelector('.pp-close').onclick = closePointPopup;
        pop.querySelector('.pp-full').onclick = ev => { ev.preventDefault(); const i = pointPopupIdx; closePointPopup(); openPointModal(i); };
        pop.querySelectorAll('[data-pp]').forEach(b => b.onclick = () => {
            const act = b.dataset.pp;
            if (act === 'approve') { if (it.curDeleted) deletedIds.delete(it.id); reviewUndo.push([it.id]); approveCrop(it.id, `Confirmed '${it.curLabel}' from the similarity map.`); }
            else if (act === 'remove') { reviewUndo.push([it.id]); excludeCrop(it.id, 'Deleted box from the similarity map.'); }
            else if (act === 'relabel') { const nl = pop.querySelector('.pp-select').value; if (nl && nl !== it.curLabel) { reviewUndo.push([it.id]); relabelCrop(it.id, it.curLabel, nl); } }
            else if (act === 'undo') undoCrop(it.id);
            refreshPointPopup();
        });
    }

    function closePointPopup() {
        const pop = document.getElementById('point-pop');
        if (pop) pop.classList.add('hidden');
        pointPopupIdx = -1;
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
