(() => {
  "use strict";

  const state = {
    activeTab: "analyze",
    docs: [],
    activeDocId: null,
    activeDoc: null,
    selectedField: null,
    hoveredField: null,
    zoom: null, // null = fit cleanly
    showHighlights: true,
    rightOpen: window.innerWidth >= 1120,
    leftOpen: true,
    typesLeftOpen: true,
    typesData: null,
    selectedTypeId: null,
    optPollTimer: null,
  };

  const $ = (id) => document.getElementById(id);

  function showToast(msg, ms = 3000) {
    const el = $("toast");
    el.textContent = msg;
    el.classList.remove("hidden");
    clearTimeout(showToast._t);
    showToast._t = setTimeout(() => el.classList.add("hidden"), ms);
  }

  async function api(url, opts = {}) {
    const res = await fetch(url, opts);
    const ctype = res.headers.get("content-type") || "";
    const payload = ctype.includes("application/json") ? await res.json() : await res.text();
    if (!res.ok) {
      throw new Error(typeof payload === "object" && payload.error ? payload.error : `HTTP ${res.status}`);
    }
    return payload;
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function highlightJsonText(rawText) {
    const raw = escapeHtml(rawText);
    const colored = raw.replace(
      /(&quot;(?:\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\&]|&(?!quot;))*&quot;(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g,
      (match) => {
        if (match.startsWith("&quot;")) {
          return /:$/.test(match) ? `<span class="j-key">${match}</span>` : `<span class="j-str">${match}</span>`;
        }
        if (/^(true|false)$/.test(match)) return `<span class="j-bool">${match}</span>`;
        if (match === "null") return `<span class="j-null">${match}</span>`;
        return `<span class="j-num">${match}</span>`;
      }
    );
    return rawText.endsWith("\n") ? `${colored} ` : colored;
  }

  function highlightJson(obj) {
    return highlightJsonText(JSON.stringify(obj, null, 2));
  }

  function flattenLeaves(obj, prefix = "", out = []) {
    if (obj && typeof obj === "object" && !Array.isArray(obj)) {
      for (const [k, v] of Object.entries(obj)) {
        flattenLeaves(v, prefix ? `${prefix}.${k}` : k, out);
      }
    } else if (Array.isArray(obj)) {
      obj.forEach((item, idx) => flattenLeaves(item, `${prefix}[${idx}]`, out));
    } else {
      const section = prefix ? prefix.split(".")[0].split("[")[0] : "root";
      out.push({ field: prefix, section, value: obj });
    }
    return out;
  }

  // ==================== DRAWERS & NAV ====================
  function syncDrawers() {
    $("rightDrawer").classList.toggle("open", state.rightOpen);
    $("btnToggleRight").classList.toggle("active", state.rightOpen);
    $("edgePullTab").classList.toggle("hidden", state.rightOpen);
    $("leftRail").classList.toggle("collapsed", !state.leftOpen);
    $("typesRail").classList.toggle("collapsed", !state.typesLeftOpen);
  }

  $("btnToggleLeft").addEventListener("click", () => {
    if (state.activeTab === "types") {
      state.typesLeftOpen = !state.typesLeftOpen;
    } else {
      state.leftOpen = !state.leftOpen;
    }
    syncDrawers();
  });

  $("btnToggleRight").addEventListener("click", () => {
    state.rightOpen = !state.rightOpen;
    syncDrawers();
  });

  $("edgePullTab").addEventListener("click", () => {
    state.rightOpen = true;
    syncDrawers();
  });

  $("btnCloseRight").addEventListener("click", () => {
    state.rightOpen = false;
    syncDrawers();
  });

  document.querySelectorAll(".nav-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".nav-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const tab = btn.dataset.tab;
      state.activeTab = tab;
      $("tab-analyze").classList.toggle("active", tab === "analyze");
      $("tab-types").classList.toggle("active", tab === "types");
      $("tab-config").classList.toggle("active", tab === "config");
      $("btnToggleRight").classList.toggle("hidden", tab !== "analyze");
      $("btnToggleLeft").classList.toggle("hidden", tab === "config");
      if (tab === "types") loadTypes();
      if (tab === "config") loadSettings();
    });
  });

  document.querySelectorAll(".dtab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".dtab").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".dpanel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      $(`subtab-${btn.dataset.subtab}`).classList.add("active");
    });
  });

  async function loadSysInfo() {
    try {
      const info = await api("/api/info");
      const shortModel = String(info.model || "").split("/").pop();
      $("sysMeta").textContent = `${shortModel} · ${info.ocr_engine} · ${info.types_count} typy`;
    } catch {
      $("sysMeta").textContent = "Offline";
    }
  }

  // ==================== FLOATING POPOVER ====================
  function showPopoverForElement(el, field, value) {
    const pop = $("hlPopover");
    $("hlPopField").textContent = field;
    $("hlPopVal").textContent = value === null || value === undefined ? "null" : String(value);
    pop.classList.remove("hidden");

    const rect = el.getBoundingClientRect();
    const popRect = pop.getBoundingClientRect();
    let top = rect.top - popRect.height - 8;
    if (top < 56) top = rect.bottom + 8;
    let left = rect.left;
    if (left + popRect.width > window.innerWidth - 12) {
      left = window.innerWidth - popRect.width - 12;
    }
    if (left < 12) left = 12;
    pop.style.top = `${Math.round(top)}px`;
    pop.style.left = `${Math.round(left)}px`;
  }

  function hidePopover() {
    $("hlPopover").classList.add("hidden");
  }

  function applyFocusHighlight() {
    const active = state.hoveredField || state.selectedField;
    document.querySelectorAll(".overlay-layer").forEach((layer) => {
      layer.classList.toggle("has-focus", Boolean(active));
    });
    document.querySelectorAll(".hl-box").forEach((box) => {
      box.classList.toggle("active", Boolean(active) && box.dataset.field === active);
    });
    document.querySelectorAll(".field-row").forEach((row) => {
      row.classList.toggle("active", Boolean(active) && row.dataset.field === active);
    });
  }

  // ==================== DOCUMENTS ====================
  async function refreshDocs(selectId = null) {
    const res = await api("/api/docs");
    state.docs = res.documents || [];
    renderDocList();
    if (selectId) {
      await selectDoc(selectId);
    } else if (!state.activeDocId && state.docs.length) {
      await selectDoc(state.docs[0].doc_id);
    }
  }

  function renderDocList() {
    $("docCount").textContent = String(state.docs.length);
    $("btnExportZip").disabled = !state.docs.some((d) => d.status === "done");

    const list = $("docList");
    if (!state.docs.length) {
      list.innerHTML = `<div class="empty-note">Przeciągnij tutaj pliki PDF lub kliknij <strong>Wgraj PDF</strong>.</div>`;
      return;
    }

    list.innerHTML = state.docs
      .map((d) => {
        const active = d.doc_id === state.activeDocId ? "active" : "";
        const tag = d.doc_type
          ? `<span class="tag ${escapeHtml(d.doc_type)}">${escapeHtml(d.doc_type)}</span>`
          : d.status === "error"
          ? `<span class="tag error">błąd</span>`
          : "";
        const secs = d.seconds != null ? `<span>${d.seconds}s</span>` : "";
        return `
          <div class="doc-row ${active}" data-id="${d.doc_id}">
            <div class="doc-row-top">
              <div class="doc-row-name" title="${escapeHtml(d.rel_path)}">
                <span class="dot ${d.status}"></span>
                <span>${escapeHtml(d.filename)}</span>
              </div>
              <button class="doc-del" data-del="${d.doc_id}" title="Usuń">×</button>
            </div>
            <div class="doc-row-sub">
              <span>${d.n_pages} str.</span>
              ${tag}
              ${secs}
            </div>
          </div>
        `;
      })
      .join("");

    list.querySelectorAll(".doc-row").forEach((el) => {
      el.addEventListener("click", (e) => {
        if (e.target.dataset.del) return;
        selectDoc(el.dataset.id);
      });
    });

    list.querySelectorAll(".doc-del").forEach((btn) => {
      btn.addEventListener("click", async (e) => {
        e.stopPropagation();
        const id = btn.dataset.del;
        await api(`/api/docs/${id}`, { method: "DELETE" });
        if (state.activeDocId === id) {
          state.activeDocId = null;
          state.activeDoc = null;
          clearDetailView();
        }
        await refreshDocs();
      });
    });
  }

  function clearDetailView() {
    $("pdfTitle").textContent = "Brak wybranego dokumentu";
    $("pdfTypeChip").classList.add("hidden");
    $("pdfSkewChip").classList.add("hidden");
    $("btnAnalyzeOne").disabled = true;
    $("btnCopyJson").disabled = true;
    $("btnDownloadJson").disabled = true;
    $("hlBadge").textContent = "0";
    $("hlBadgeTop").textContent = "0";
    $("hlBadgeStrip").textContent = "0";
    hidePopover();
    $("pdfCanvas").innerHTML = `
      <div class="stage-empty">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#1282A2" stroke-width="1.4"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><path d="M14 2v6h6M8 13h8M8 17h5"/></svg>
        <h3>Podgląd dokumentu</h3>
        <p>Przeciągnij plik PDF w dowolne miejsce okna lub użyj <strong>Wgraj PDF</strong>.</p>
      </div>`;
    $("fieldsContainer").innerHTML = `<div class="empty-note">Uruchom analizę, aby wyświetlić wyekstrahowane pola.</div>`;
    $("jsonViewer").textContent = "{}";
    $("auditContainer").innerHTML = `<div class="empty-note">Ślad audytowy pojawi się po analizie dokumentu.</div>`;
  }

  async function selectDoc(docId) {
    state.activeDocId = docId;
    state.selectedField = null;
    state.hoveredField = null;
    hidePopover();
    renderDocList();
    state.activeDoc = await api(`/api/docs/${docId}`);
    renderActiveDoc();
  }

  async function uploadFiles(fileList) {
    let lastId = null;
    let count = 0;
    for (const file of fileList) {
      if (!file.name.toLowerCase().endsWith(".pdf")) continue;
      try {
        const buf = await file.arrayBuffer();
        const doc = await api(`/api/docs/upload?filename=${encodeURIComponent(file.name)}`, {
          method: "POST",
          headers: {
            "Content-Type": "application/pdf",
            "X-Filename": encodeURIComponent(file.name),
            "X-Relpath": encodeURIComponent(file.webkitRelativePath || file.name),
          },
          body: buf,
        });
        lastId = doc.doc_id;
        count += 1;
      } catch (err) {
        showToast(`Błąd wgrywania ${file.name}: ${err.message}`, 5000);
      }
    }
    if (lastId) {
      showToast(`Dodano ${count} plik(ów) PDF`);
      await refreshDocs(lastId);
    }
  }

  $("btnUpload").addEventListener("click", () => $("fileInput").click());
  $("fileInput").addEventListener("change", () => {
    if ($("fileInput").files.length) uploadFiles($("fileInput").files);
    $("fileInput").value = "";
  });

  // Global window drag & drop (no tacky dashed box needed)
  let dragDepth = 0;
  window.addEventListener("dragenter", (e) => {
    e.preventDefault();
    dragDepth += 1;
    $("dropOverlay").classList.remove("hidden");
  });
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("dragleave", () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) $("dropOverlay").classList.add("hidden");
  });
  window.addEventListener("drop", (e) => {
    e.preventDefault();
    dragDepth = 0;
    $("dropOverlay").classList.add("hidden");
    if (e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files);
  });

  $("btnLoadInput").addEventListener("click", async () => {
    const res = await api("/api/docs/load-input", { method: "POST" });
    showToast(`Wczytano ${res.added.length} nowych PDF-ów z ./input`);
    await refreshDocs(res.added[0]?.doc_id || null);
  });

  async function analyzeOne(docId) {
    const target = state.docs.find((d) => d.doc_id === docId);
    if (target) target.status = "processing";
    renderDocList();
    if (docId === state.activeDocId) {
      $("btnAnalyzeOne").disabled = true;
      $("btnAnalyzeOne").textContent = "Analizuję…";
    }
    try {
      const detail = await api(`/api/docs/${docId}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ force_ocr: $("chkForceOcr").checked }),
      });
      if (docId === state.activeDocId) {
        state.rightOpen = true;
        syncDrawers();
      }
      await refreshDocs(docId === state.activeDocId ? docId : null);
      if (detail.error) {
        showToast(`Błąd: ${detail.error}`);
      } else {
        showToast(`${detail.filename} → ${detail.doc_type} (${detail.seconds}s)`);
      }
    } catch (err) {
      showToast(`Błąd: ${err.message}`);
      await refreshDocs();
    } finally {
      if (docId === state.activeDocId) {
        $("btnAnalyzeOne").disabled = false;
        $("btnAnalyzeOne").textContent = "Analizuj";
      }
    }
  }

  $("btnAnalyzeOne").addEventListener("click", () => {
    if (state.activeDocId) analyzeOne(state.activeDocId);
  });

  $("btnAnalyzeAll").addEventListener("click", async () => {
    const queue = state.docs.filter((d) => d.status !== "done").map((d) => d.doc_id);
    const ids = queue.length ? queue : state.docs.map((d) => d.doc_id);
    if (!ids.length) return;
    $("btnAnalyzeAll").disabled = true;
    for (const id of ids) await analyzeOne(id);
    $("btnAnalyzeAll").disabled = false;
  });

  $("btnExportZip").addEventListener("click", () => {
    window.location.href = "/api/docs/export.zip";
  });

  // ==================== PDF STAGE + HIGHLIGHTS ====================
  function renderActiveDoc() {
    const doc = state.activeDoc;
    if (!doc) return;

    $("pdfTitle").textContent = doc.rel_path;
    $("btnAnalyzeOne").disabled = false;
    $("btnAnalyzeOne").textContent = doc.status === "done" ? "Analizuj ponownie" : "Analizuj";

    const chip = $("pdfTypeChip");
    if (doc.doc_type) {
      chip.textContent = doc.doc_type;
      chip.className = `tag ${doc.doc_type}`;
    } else {
      chip.classList.add("hidden");
    }

    const skewAngles = Object.values(doc.page_skew_angles || {});
    const skewEl = $("pdfSkewChip");
    if (skewAngles.length) {
      skewEl.textContent = `Prostowanie ${skewAngles[0] > 0 ? "+" : ""}${skewAngles[0]}°`;
      skewEl.classList.remove("hidden");
    } else {
      skewEl.classList.add("hidden");
    }

    renderPdfPages(doc);
    renderFieldsPanel(doc);
    renderJsonPanel(doc);
    renderAuditPanel(doc);
  }

  function pageWidthStyle() {
    if (state.zoom === null) return "width:100%;max-width:780px;";
    return `width:${Math.round(760 * state.zoom)}px;`;
  }

  function renderPdfPages(doc) {
    const canvas = $("pdfCanvas");
    const byPage = new Map();
    for (const hl of doc.highlights || []) {
      if (!byPage.has(hl.page)) byPage.set(hl.page, []);
      byPage.get(hl.page).push(hl);
    }

    const wStyle = pageWidthStyle();
    const rev = doc.render_rev || 0;
    const skewMap = doc.page_skew_angles || {};
    const layerClass = state.showHighlights ? "overlay-layer" : "overlay-layer hidden-boxes";

    const html = [];
    for (let p = 1; p <= doc.n_pages; p++) {
      const hls = byPage.get(p) || [];
      const boxes = hls
        .map((hl) => {
          const left = (hl.x0 * 100).toFixed(3);
          const top = (hl.y0 * 100).toFixed(3);
          const width = Math.max(0.7, (hl.x1 - hl.x0) * 100).toFixed(3);
          const height = Math.max(0.8, (hl.y1 - hl.y0) * 100).toFixed(3);
          return `<div class="hl-box" data-field="${escapeHtml(hl.field)}" data-val="${escapeHtml(hl.value)}" style="left:${left}%;top:${top}%;width:${width}%;height:${height}%;--hl-color:${hl.color};"></div>`;
        })
        .join("");

      html.push(`
        <div class="pdf-page-card" data-page="${p}" style="${wStyle}">
          <div class="pdf-page-meta">
            <span>Strona ${p} z ${doc.n_pages}</span>
            <span>${hls.length ? `${hls.length} pól` : ""}</span>
          </div>
          <div class="pdf-page-stage">
            <img src="/api/docs/${doc.doc_id}/page/${p}.png?rev=${rev}" alt="Strona ${p}" loading="lazy" />
            <div class="${layerClass}">${boxes}</div>
          </div>
        </div>
      `);
    }

    canvas.innerHTML = html.join("");
    canvas.querySelectorAll(".hl-box").forEach((box) => {
      box.addEventListener("mouseenter", () => {
        state.hoveredField = box.dataset.field;
        applyFocusHighlight();
        showPopoverForElement(box, box.dataset.field, box.dataset.val);
      });
      box.addEventListener("mouseleave", () => {
        state.hoveredField = null;
        applyFocusHighlight();
        hidePopover();
      });
      box.addEventListener("click", (e) => {
        e.stopPropagation();
        state.selectedField = box.dataset.field;
        state.rightOpen = true;
        syncDrawers();
        document.querySelector('.dtab[data-subtab="fields"]').click();
        applyFocusHighlight();
        const row = document.querySelector(`.field-row[data-field="${CSS.escape(box.dataset.field)}"]`);
        if (row) row.scrollIntoView({ behavior: "smooth", block: "center" });
      });
    });

    canvas.addEventListener("click", () => {
      state.selectedField = null;
      applyFocusHighlight();
      hidePopover();
    });
    canvas.addEventListener("scroll", hidePopover, { passive: true });
  }

  $("chkHighlights").addEventListener("change", (e) => {
    state.showHighlights = e.target.checked;
    document.querySelectorAll(".overlay-layer").forEach((l) => l.classList.toggle("hidden-boxes", !state.showHighlights));
    if (!state.showHighlights) hidePopover();
  });

  function updateZoom(val) {
    state.zoom = val === null ? null : Math.max(0.6, Math.min(2.2, Math.round(val * 100) / 100));
    $("btnZoomReset").textContent = state.zoom === null ? "Dopasuj" : `${Math.round(state.zoom * 100)}%`;
    const style = pageWidthStyle();
    document.querySelectorAll(".pdf-page-card").forEach((c) => c.setAttribute("style", style));
    hidePopover();
  }
  $("btnZoomIn").addEventListener("click", () => updateZoom((state.zoom ?? 1.0) + 0.15));
  $("btnZoomOut").addEventListener("click", () => updateZoom((state.zoom ?? 1.0) - 0.15));
  $("btnZoomReset").addEventListener("click", () => updateZoom(null));

  // ==================== RIGHT DRAWER: FIELDS / JSON / AUDIT ====================
  function renderFieldsPanel(doc) {
    const container = $("fieldsContainer");
    const hlMap = new Map();
    for (const hl of doc.highlights || []) {
      if (!hlMap.has(hl.field)) hlMap.set(hl.field, hl);
    }
    $("hlBadge").textContent = String(hlMap.size);
    $("hlBadgeTop").textContent = String(hlMap.size);
    $("hlBadgeStrip").textContent = String(hlMap.size);

    if (doc.error) {
      container.innerHTML = `<div class="error-banner"><strong>Błąd:</strong> ${escapeHtml(doc.error)}</div>`;
      return;
    }
    if (!doc.data) {
      container.innerHTML = `<div class="empty-note">Kliknij <strong>Analizuj</strong>, aby wyekstrahować dane.</div>`;
      return;
    }
    if (!Object.keys(doc.data).length) {
      container.innerHTML = `<div class="empty-note">Dokument rozpoznany jako <strong>other</strong> — zwrócono pusty obiekt <code>{}</code>.</div>`;
      return;
    }

    const q = ($("fieldSearch").value || "").trim().toLowerCase();
    const leaves = flattenLeaves(doc.data).filter((l) => {
      if (!q) return true;
      return l.field.toLowerCase().includes(q) || String(l.value ?? "").toLowerCase().includes(q);
    });

    const grouped = new Map();
    for (const leaf of leaves) {
      if (!grouped.has(leaf.section)) grouped.set(leaf.section, []);
      grouped.get(leaf.section).push(leaf);
    }

    container.innerHTML = [...grouped.entries()]
      .map(([sec, rows]) => {
        const secColor = rows.map((r) => hlMap.get(r.field)?.color).find(Boolean) || "#034078";
        const rowsHtml = rows
          .map((r) => {
            const hl = hlMap.get(r.field);
            const pill = hl
              ? `<span class="page-pill" style="background:${hl.color};">Str. ${hl.page}</span>`
              : r.value === null
              ? `<span class="page-pill muted">brak</span>`
              : `<span class="page-pill muted">wniosek</span>`;
            const valHtml =
              r.value === null
                ? `<div class="field-val null-val">null</div>`
                : `<div class="field-val">${escapeHtml(r.value)}</div>`;
            return `
              <div class="field-row" data-field="${escapeHtml(r.field)}">
                <div class="field-row-top">
                  <span class="field-key">${escapeHtml(r.field)}</span>
                  ${pill}
                </div>
                ${valHtml}
              </div>
            `;
          })
          .join("");
        return `
          <div class="section-card">
            <div class="section-head">
              <span class="sec-dot" style="background:${secColor};"></span>
              <span>${escapeHtml(sec)}</span>
            </div>
            ${rowsHtml}
          </div>
        `;
      })
      .join("");

    container.querySelectorAll(".field-row").forEach((row) => {
      const f = row.dataset.field;
      row.addEventListener("mouseenter", () => {
        state.hoveredField = f;
        applyFocusHighlight();
        const box = document.querySelector(`.hl-box[data-field="${CSS.escape(f)}"]`);
        if (box) showPopoverForElement(box, f, box.dataset.val);
      });
      row.addEventListener("mouseleave", () => {
        state.hoveredField = null;
        applyFocusHighlight();
        hidePopover();
      });
      row.addEventListener("click", () => {
        state.selectedField = state.selectedField === f ? null : f;
        applyFocusHighlight();
        const box = document.querySelector(`.hl-box[data-field="${CSS.escape(f)}"]`);
        if (box && state.selectedField) {
          box.scrollIntoView({ behavior: "smooth", block: "center" });
          setTimeout(() => showPopoverForElement(box, f, box.dataset.val), 250);
        } else {
          hidePopover();
        }
      });
    });
  }

  $("fieldSearch").addEventListener("input", () => {
    if (state.activeDoc) renderFieldsPanel(state.activeDoc);
  });

  function renderJsonPanel(doc) {
    const hasData = doc.data !== null && doc.data !== undefined;
    $("btnCopyJson").disabled = !hasData;
    $("btnDownloadJson").disabled = !hasData;
    $("jsonViewer").innerHTML = hasData ? highlightJson(doc.data) : "{}";
  }

  $("btnCopyJson").addEventListener("click", async () => {
    if (!state.activeDoc?.data) return;
    await navigator.clipboard.writeText(JSON.stringify(state.activeDoc.data, null, 2));
    showToast("Skopiowano JSON");
  });

  $("btnDownloadJson").addEventListener("click", () => {
    if (state.activeDocId) window.location.href = `/api/docs/${state.activeDocId}/json`;
  });

  function renderAuditPanel(doc) {
    const el = $("auditContainer");
    if (!doc.classification && !doc.strategy) {
      el.innerHTML = `<div class="empty-note">Ślad audytowy pojawi się po analizie dokumentu.</div>`;
      return;
    }
    const cls = doc.classification || {};
    const probs = cls.probabilities || {};
    const probRows = Object.entries(probs)
      .sort((a, b) => b[1] - a[1])
      .map(([k, v]) => {
        const pct = Math.round(v * 100);
        return `
          <div class="prob-row">
            <strong>${escapeHtml(k)}</strong>
            <div class="prob-track"><div class="prob-fill ${k === doc.doc_type ? "winner" : ""}" style="width:${pct}%"></div></div>
            <span>${(v * 100).toFixed(1)}%</span>
          </div>
        `;
      })
      .join("");

    el.innerHTML = `
      <div class="audit-card">
        <h4>Odczyt dokumentu</h4>
        <div style="font-size:12px;line-height:1.55;">
          <strong>Strategia:</strong> ${escapeHtml(doc.strategy || "-")} · ${escapeHtml(doc.strategy_reason || "")}<br/>
          <strong>Czas:</strong> ${doc.seconds ?? "-"} s · <strong>Zaznaczenia na PDF:</strong> ${(doc.highlights || []).length}
        </div>
      </div>
      <div class="audit-card">
        <h4>Klasyfikacja</h4>
        <div style="font-size:11.5px;margin-bottom:8px;color:#526071;"><strong>Metoda:</strong> ${escapeHtml(cls.method || "-")} · ${escapeHtml(cls.reason || "")}</div>
        ${probRows || `<div style="font-size:12px;">Pewność: <strong>${cls.probability != null ? (cls.probability * 100).toFixed(1) + "%" : "-"}</strong></div>`}
      </div>
      <div class="audit-card">
        <h4>Surowa odpowiedź LLM</h4>
        <pre class="audit-pre">${escapeHtml(doc.trace?.raw_output || "Brak danych")}</pre>
      </div>
      <div class="audit-card">
        <h4>Tekst przekazany do modelu</h4>
        <pre class="audit-pre">${escapeHtml((doc.trace?.input_text || "").slice(0, 5000))}</pre>
      </div>
    `;
  }

  // ==================== TAB 2: TYPY DOKUMENTÓW ====================
  let cmEditor = null;

  function ensureCodeMirror() {
    if (cmEditor || typeof window.CodeMirror !== "function") return cmEditor;
    cmEditor = window.CodeMirror.fromTextArea($("typeJsonEditor"), {
      mode: { name: "javascript", json: true },
      lineNumbers: true,
      tabSize: 2,
      indentUnit: 2,
      lineWrapping: false,
      extraKeys: {
        Tab: (cm) => cm.replaceSelection("  ", "end"),
      },
    });
    return cmEditor;
  }

  function getJsonEditorValue() {
    return cmEditor ? cmEditor.getValue() : $("typeJsonEditor").value;
  }

  function setJsonEditorValue(text) {
    const cm = ensureCodeMirror();
    if (cm) {
      cm.setValue(text);
    } else {
      $("typeJsonEditor").value = text;
    }
  }

  function setSlSelectValue(selectEl, value) {
    if (!selectEl) return;
    const val = value || "";
    selectEl.value = val;
    selectEl.setAttribute("value", val);
    if (window.customElements?.whenDefined) {
      window.customElements.whenDefined("sl-select").then(() => {
        selectEl.value = val;
      });
    }
  }

  function setTypeMode(mode) {
    const sw = $("typeModeSwitch");
    if (sw && sw.value !== mode) {
      sw.value = mode;
      sw.setAttribute("value", mode);
    }
    document.querySelectorAll(".tmode-panel").forEach((p) => p.classList.toggle("active", p.id === `tmode-${mode}`));
    if (mode === "json") {
      const cm = ensureCodeMirror();
      if (cm) requestAnimationFrame(() => cm.refresh());
    }
  }

  $("typeModeSwitch").addEventListener("sl-change", (e) => {
    const mode = e.target.value || "preview";
    if (mode === "preview") {
      try {
        renderSchemaOverview(JSON.parse(getJsonEditorValue()));
      } catch {
        // Keep current preview if JSON is mid-edit
      }
    }
    setTypeMode(mode);
  });

  function formatRuleItem(note) {
    const text = String(note || "").trim();
    const colonIdx = text.indexOf(":");
    if (colonIdx > 1 && colonIdx <= 48) {
      const lead = text.slice(0, colonIdx + 1);
      const rest = text.slice(colonIdx + 1);
      return `<li><span class="note-lead">${escapeHtml(lead)}</span>${escapeHtml(rest)}</li>`;
    }
    return `<li>${escapeHtml(text)}</li>`;
  }

  function collectSchemaRows(node, root, prefix = "", out = []) {
    if (!node || typeof node !== "object") return out;
    if (node.$ref && node.$ref.startsWith("#/")) {
      let target = root;
      for (const part of node.$ref.slice(2).split("/")) target = target?.[part];
      if (target) node = { ...target, ...node, $ref: undefined };
    }
    if (node.properties) {
      const reqSet = new Set(node.required || []);
      for (const [k, sub] of Object.entries(node.properties)) {
        const path = prefix ? `${prefix}.${k}` : k;
        const resolved = sub?.$ref?.startsWith("#/")
          ? { ...sub.$ref.slice(2).split("/").reduce((a, p) => a?.[p], root), ...sub }
          : sub;
        if (resolved?.properties) {
          collectSchemaRows(resolved, root, path, out);
        } else if (resolved?.type === "array" && resolved?.items?.properties) {
          collectSchemaRows(resolved.items, root, `${path}[]`, out);
        } else {
          const t = Array.isArray(resolved?.type) ? resolved.type.join(" | ") : resolved?.type || "any";
          const norm = resolved?.["x-normalizer"] || resolved?.format || "";
          out.push({
            path,
            group: path.split(".")[0].replace(/\[\]$/, ""),
            leaf: path.includes(".") ? path.slice(path.indexOf(".") + 1) : path,
            type: t,
            norm,
            required: reqSet.has(k),
            desc: resolved?.description || "",
          });
        }
      }
    }
    return out;
  }

  function renderSchemaOverview(rawObj) {
    const el = $("schemaOverview");
    const rows = collectSchemaRows(rawObj.schema, rawObj.schema);
    const notes = rawObj.extraction?.uwagi || [];
    const pages = rawObj.extraction?.pages ?? "1-3";
    const prio = rawObj.classification?.priority ?? "-";
    const lookups = Object.keys(rawObj.extraction || {}).filter(
      (k) => !["pages", "uwagi", "lookup_bindings", "consistency_rules"].includes(k)
    );

    const groups = new Set(rows.map((r) => r.group));
    const useGroups = groups.size > 1;
    let lastGroup = null;
    const tableRows = rows
      .map((r) => {
        const groupHeader =
          useGroups && r.group !== lastGroup
            ? `<tr class="group-row"><td colspan="4">${escapeHtml(r.group)}</td></tr>`
            : "";
        lastGroup = r.group;
        return `${groupHeader}
          <tr>
            <td class="mono-cell">${escapeHtml(useGroups ? r.leaf : r.path)}</td>
            <td><sl-tag size="small" variant="neutral">${escapeHtml(r.type)}</sl-tag></td>
            <td>${r.norm ? `<sl-tag size="small" variant="primary">${escapeHtml(r.norm)}</sl-tag>` : "—"}</td>
            <td class="desc-cell">${escapeHtml(r.desc)}</td>
          </tr>`;
      })
      .join("");

    const totalNoteChars = notes.reduce((acc, n) => acc + String(n).length, 0);
    const openByDefault = notes.length <= 4 && totalNoteChars <= 600;
    const rulesCountLabel = `${notes.length} ${notes.length === 1 ? "reguła" : notes.length < 5 ? "reguły" : "reguł"}`;
    const notesHtml = notes.length
      ? `
        <sl-details class="rules-details" summary="Reguły ekstrakcji — ${rulesCountLabel}" ${openByDefault ? "open" : ""}>
          <ul class="notes-list">${notes.map(formatRuleItem).join("")}</ul>
        </sl-details>
      `
      : "";

    el.innerHTML = `
      <div class="overview-summary">
        <h3>${escapeHtml(rawObj.nazwa || rawObj.type_id || "Nowy typ")}</h3>
        <p>${escapeHtml(rawObj.opis || "")}</p>
        <div class="overview-stats">
          <span>Priorytet: <strong>${escapeHtml(prio)}</strong></span>
          <span>Strony OCR: <strong>${escapeHtml(pages)}</strong></span>
          <span>Pola schematu: <strong>${rows.length}</strong></span>
          <span>Reguły: <strong>${notes.length}</strong></span>
          <span>Słowniki: <strong>${lookups.length ? escapeHtml(lookups.join(", ")) : "brak"}</strong></span>
        </div>
      </div>

      ${notesHtml}

      <div class="schema-table-wrap">
        <div class="schema-table-head">
          <span>Struktura pól JSON</span>
          <span class="muted-num">${rows.length}</span>
        </div>
        <table class="schema-table">
          <thead>
            <tr><th>Pole</th><th>Typ</th><th>Normalizacja</th><th>Opis</th></tr>
          </thead>
          <tbody>${tableRows || '<tr><td colspan="4" class="empty-note">Brak zdefiniowanych pól</td></tr>'}</tbody>
        </table>
      </div>
    `;
  }

  function openTypeInEditor(rawObj, isExisting) {
    state.selectedTypeId = rawObj.type_id || null;
    $("editorTitle").textContent = rawObj.nazwa || rawObj.type_id || "Nowy typ dokumentu";
    const tag = $("editorTypeTag");
    if (rawObj.type_id) {
      tag.textContent = rawObj.type_id;
      tag.className = `tag ${rawObj.type_id}`;
    } else {
      tag.classList.add("hidden");
    }
    $("editorErrors").classList.add("hidden");
    $("btnDeleteType").classList.toggle("hidden", !isExisting);
    setJsonEditorValue(JSON.stringify(rawObj, null, 2));
    renderSchemaOverview(rawObj);
    document.querySelectorAll("#typesList .type-card").forEach((c) => {
      c.classList.toggle("active", c.dataset.tid === state.selectedTypeId);
    });
    if ($("optTypeSelect") && isExisting && state.selectedTypeId) {
      setSlSelectValue($("optTypeSelect"), state.selectedTypeId);
    }
  }

  $("btnPickInferFile").addEventListener("click", () => $("inferFile").click());
  $("inferFile").addEventListener("change", () => {
    const f = $("inferFile").files[0];
    $("inferFileName").textContent = f ? f.name : "Nie wybrano pliku";
  });

  $("btnPickOptFiles").addEventListener("click", () => $("optFiles").click());
  $("optFiles").addEventListener("change", () => {
    const files = Array.from($("optFiles").files || []);
    if (!files.length) {
      $("optFilesName").textContent = "Nie wybrano plików";
      return;
    }
    const pdfs = files.filter((f) => f.name.toLowerCase().endsWith(".pdf")).length;
    const jsons = files.filter((f) => f.name.toLowerCase().endsWith(".json")).length;
    $("optFilesName").textContent = `Wybrano ${files.length} plików (${pdfs} PDF, ${jsons} JSON)`;
  });

  async function fileToBase64(file) {
    const bytes = new Uint8Array(await file.arrayBuffer());
    let bin = "";
    const chunk = 0x8000;
    for (let i = 0; i < bytes.length; i += chunk) {
      bin += String.fromCharCode(...bytes.subarray(i, i + chunk));
    }
    return btoa(bin);
  }

  async function loadTypes() {
    const data = await api("/api/types");
    state.typesData = data;
    const pill = $("typesValidPill");
    pill.classList.toggle("hidden", Boolean(data.valid));

    $("typesList").innerHTML = data.types
      .map(
        (t) => `
        <div class="type-card ${state.selectedTypeId === t.type_id ? "active" : ""}" data-tid="${escapeHtml(t.type_id)}">
          <div class="type-card-top">
            <span class="type-card-title">[${t.priority}] ${escapeHtml(t.name)}</span>
            <span class="tag ${escapeHtml(t.type_id)}">${escapeHtml(t.type_id)}</span>
          </div>
          <div class="type-card-desc">${escapeHtml(t.description)}</div>
        </div>
      `
      )
      .join("");

    $("typesList").querySelectorAll(".type-card").forEach((c) => {
      c.addEventListener("click", () => {
        const t = data.types.find((x) => x.type_id === c.dataset.tid);
        if (t) openTypeInEditor(t.raw, true);
      });
    });

    $("examplesList").innerHTML =
      (data.examples || [])
        .map(
          (ex) => `
        <div class="type-card" data-ex="${escapeHtml(ex.type_id)}">
          <div class="type-card-top">
            <span class="type-card-title">${escapeHtml(ex.name)}</span>
            <span class="valid-tag ${ex.active ? "ok" : ""}">${ex.active ? "Aktywny" : "Przykład"}</span>
          </div>
          <div class="type-card-desc">${escapeHtml(ex.description)}</div>
        </div>
      `
        )
        .join("") || `<div class="empty-note">Brak przykładów.</div>`;

    $("examplesList").querySelectorAll(".type-card").forEach((c) => {
      c.addEventListener("click", () => {
        const ex = data.examples.find((x) => x.type_id === c.dataset.ex);
        if (ex) {
          openTypeInEditor(ex.raw, ex.active);
          showToast(`Załadowano przykład '${ex.type_id}'`);
        }
      });
    });

    const optSelect = $("optTypeSelect");
    const prevSel = optSelect.value || state.selectedTypeId || (data.types[0]?.type_id ?? "");
    optSelect.innerHTML = data.types
      .map((t) => `<sl-option value="${escapeHtml(t.type_id)}">${escapeHtml(t.name)}</sl-option>`)
      .join("");
    if (prevSel && data.types.some((t) => t.type_id === prevSel)) {
      setSlSelectValue(optSelect, prevSel);
    }

    if (!state.selectedTypeId && data.types.length) {
      openTypeInEditor(data.types[0].raw, true);
    }
  }

  $("btnNewType").addEventListener("click", () => {
    openTypeInEditor(
      {
        type_id: "nowy_dokument",
        nazwa: "Nowy dokument",
        opis: "Krótki opis nowego typu dokumentu.",
        classification: { priority: 10, include: "Kiedy dokument JEST tego typu.", exclude: "Czym dokument NIE jest." },
        extraction: { pages: "1-3,last", uwagi: ["Dodaj tutaj instrukcje ekstrakcji."] },
        schema: {
          $schema: "https://json-schema.org/draft/2020-12/schema",
          $id: "nowy_dokument",
          title: "Nowy dokument",
          type: "object",
          additionalProperties: false,
          required: ["Dokument"],
          properties: {
            Dokument: {
              type: "object",
              additionalProperties: false,
              required: ["data", "podmiot"],
              properties: {
                data: { type: ["string", "null"], format: "date", pattern: "^\\d{4}-\\d{2}-\\d{2}$", description: "Data w formacie YYYY-MM-DD." },
                podmiot: { type: ["string", "null"], "x-normalizer": "entity_name", description: "Nazwa podmiotu." },
              },
            },
          },
        },
      },
      false
    );
    setTypeMode("json");
  });

  $("btnSaveType").addEventListener("click", async () => {
    const errBox = $("editorErrors");
    errBox.classList.add("hidden");
    let parsed;
    try {
      parsed = JSON.parse(getJsonEditorValue());
    } catch (exc) {
      errBox.textContent = `Błąd JSON: ${exc.message}`;
      errBox.classList.remove("hidden");
      return;
    }
    const btn = $("btnSaveType");
    btn.loading = true;
    try {
      const res = await api("/api/types/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(parsed),
      });
      showToast(`Zapisano i aktywowano '${res.type_id}'`);
      state.selectedTypeId = res.type_id;
      await Promise.all([loadTypes(), loadSysInfo()]);
    } catch (exc) {
      errBox.textContent = exc.message;
      errBox.classList.remove("hidden");
    } finally {
      btn.loading = false;
    }
  });

  $("btnDeleteType").addEventListener("click", async () => {
    if (!state.selectedTypeId || !confirm(`Usunąć typ '${state.selectedTypeId}'?`)) return;
    await api(`/api/types/${state.selectedTypeId}`, { method: "DELETE" });
    showToast(`Usunięto '${state.selectedTypeId}'`);
    state.selectedTypeId = null;
    await Promise.all([loadTypes(), loadSysInfo()]);
  });

  $("btnInferType").addEventListener("click", async () => {
    const file = $("inferFile").files[0];
    const typeId = ($("inferTypeId").value || "").trim().toLowerCase();
    const name = ($("inferName").value || "").trim();
    const status = $("inferStatus");
    if (!file || !typeId) {
      showToast("Podaj identyfikator typu i wybierz plik .json lub .pdf");
      return;
    }
    status.textContent = "Generowanie konfiguracji…";
    status.classList.remove("hidden");
    const btn = $("btnInferType");
    btn.loading = true;
    try {
      const contentB64 = await fileToBase64(file);
      const res = await api("/api/types/infer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: file.name, content_base64: contentB64, type_id: typeId, name: name || undefined }),
      });
      openTypeInEditor(res.draft, false);
      setTypeMode("preview");
      status.textContent = res.valid ? "Wygenerowano — przejrzyj i kliknij Zapisz i aktywuj." : `Szkic ma ${res.issues.length} uwag walidacji.`;
    } catch (exc) {
      status.textContent = `Błąd: ${exc.message}`;
    } finally {
      btn.loading = false;
    }
  });

  function renderOptimizationResult(job) {
    const box = $("optResultBox");
    const r = job.result;
    if (!r) {
      box.classList.add("hidden");
      return;
    }
    const before = ((r.seed_val?.f1 || 0) * 100).toFixed(2);
    const after = ((r.best_val?.f1 || 0) * 100).toFixed(2);
    const delta = (((r.best_val?.f1 || 0) - (r.seed_val?.f1 || 0)) * 100).toFixed(2);
    const deltaSign = Number(delta) >= 0 ? "+" : "";
    const appliedMsg = r.applied_to
      ? `<sl-tag size="small" variant="success" pill>Zapisano automatycznie</sl-tag>`
      : `<sl-button variant="primary" size="small" id="btnLoadOptNotes">Wczytaj nowe reguły do edytora</sl-button>`;

    box.innerHTML = `
      <div class="opt-metrics">
        <span>Model: <strong>${escapeHtml(job.model)}</strong></span>
        <span>Walidacja F1: <strong>${before}%</strong> → <span class="opt-f1-gain">${after}%</span> (${deltaSign}${delta} pp)</span>
        <span>Próbek: <strong>${r.train_count}</strong> train / <strong>${r.val_count}</strong> val</span>
        <span>Czas: <strong>${r.elapsed_seconds} s</strong></span>
      </div>
      <div>${appliedMsg}</div>
    `;
    box.classList.remove("hidden");

    const loadBtn = $("btnLoadOptNotes");
    if (loadBtn) {
      loadBtn.addEventListener("click", () => {
        const typeObj = (state.typesData?.types || []).find((t) => t.type_id === r.type_id);
        if (!typeObj) return;
        const clone = JSON.parse(JSON.stringify(typeObj.raw));
        clone.extraction = clone.extraction || {};
        clone.extraction.uwagi = r.best_notes || [];
        openTypeInEditor(clone, true);
        setTypeMode("preview");
        showToast("Załadowano zoptymalizowane reguły — kliknij Zapisz i aktywuj");
      });
    }
  }

  $("btnRunOptimize").addEventListener("click", async () => {
    const typeId = $("optTypeSelect").value;
    const files = Array.from($("optFiles").files || []);
    const dataDir = ($("optDataDir").value || "").trim();
    const budget = Number($("optBudget").value) || 20;
    const apply = $("optApply").checked;
    const status = $("optStatus");

    if (!typeId) {
      showToast("Wybierz typ dokumentu do optymalizacji");
      return;
    }
    if (!files.length && !dataDir) {
      showToast("Wybierz pliki .pdf + .json lub podaj ścieżkę katalogu");
      return;
    }

    clearInterval(state.optPollTimer);
    $("optResultBox").classList.add("hidden");
    const btn = $("btnRunOptimize");
    btn.loading = true;
    status.textContent = "Przygotowanie danych…";
    status.classList.remove("hidden");

    try {
      const payload = { type_id: typeId, max_metric_calls: budget, apply };
      if (files.length) {
        payload.samples = await Promise.all(
          files.map(async (f) => ({ filename: f.name, content_base64: await fileToBase64(f) }))
        );
      } else {
        payload.data_dir = dataDir;
      }

      const job = await api("/api/types/optimize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      status.textContent = `Optymalizacja w toku (${job.model})… 0 s`;

      state.optPollTimer = setInterval(async () => {
        try {
          const snap = await api(`/api/types/optimize/${job.job_id}`);
          if (snap.status === "running") {
            status.textContent = `Optymalizacja w toku (${snap.model})… ${snap.elapsed_seconds} s`;
            return;
          }
          clearInterval(state.optPollTimer);
          btn.loading = false;
          if (snap.status === "done") {
            status.textContent = "Optymalizacja zakończona.";
            renderOptimizationResult(snap);
            if (snap.result?.applied_to) {
              await loadTypes();
              showToast(`Zastosowano zoptymalizowane reguły dla '${typeId}'`);
            }
          } else {
            status.textContent = `Błąd: ${snap.error || "nieznany błąd"}`;
          }
        } catch (pollErr) {
          clearInterval(state.optPollTimer);
          btn.loading = false;
          status.textContent = `Błąd odczytu statusu: ${pollErr.message}`;
        }
      }, 2000);
    } catch (exc) {
      btn.loading = false;
      status.textContent = `Błąd: ${exc.message}`;
    }
  });

  // ==================== TAB 3: KONFIGURACJA ====================
  function renderKeyBadge(el, inputEl, isSet, source = null) {
    if (source === "default") {
      el.textContent = "Domyślny";
      el.setAttribute("variant", "success");
      inputEl.value = "";
      inputEl.placeholder = "Aktywny klucz domyślny — wpisz inny, aby nadpisać";
      return;
    }
    el.textContent = isSet ? "Zapisany" : "Brak";
    el.setAttribute("variant", isSet ? "success" : "neutral");
    inputEl.value = "";
    inputEl.placeholder = isSet ? "Klucz zapisany — wpisz nowy, aby zmienić" : "Wpisz klucz API…";
  }

  function syncDeciderVisibility() {
    const isDecider = ($("cfgClsMethod").value || "decider") === "decider";
    $("deciderFieldsGroup").classList.toggle("hidden", !isDecider);
  }
  $("cfgClsMethod").addEventListener("sl-change", syncDeciderVisibility);
  $("cfgClsMethod").addEventListener("change", syncDeciderVisibility);

  async function loadSettings() {
    try {
      const s = await api("/api/settings");
      $("cfgLlmUrl").value = s.llm_base_url || "";
      $("cfgLlmModel").value = s.llm_model || "";
      $("cfgLlmTimeout").value = String(s.llm_timeout ?? 300);
      renderKeyBadge($("llmKeyStatus"), $("cfgLlmKey"), Boolean(s.llm_api_key_set), s.llm_api_key_source);

      setSlSelectValue($("cfgClsMethod"), s.classification_method || "decider");
      $("cfgFallbackRules").checked = Boolean(s.fallback_to_rules);
      $("cfgDeciderUrl").value = s.decider_url || "";
      $("cfgDeciderModel").value = s.decider_model || "";
      renderKeyBadge($("orKeyStatus"), $("cfgOrKey"), Boolean(s.openrouter_api_key_set));

      $("cfgDeskew").checked = Boolean(s.ocr_deskew);
      $("cfgRenderWidth").value = String(s.ocr_render_width_px ?? 1654);
      $("cfgWorkers").value = String(s.default_workers ?? 4);
      syncDeciderVisibility();
    } catch (err) {
      showToast(`Błąd wczytywania konfiguracji: ${err.message}`);
    }
  }

  document.querySelectorAll("[data-preset]").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (btn.dataset.preset === "regolo") {
        $("cfgLlmUrl").value = "https://api.regolo.ai/v1";
        $("cfgLlmModel").value = "qwen3.5-9b";
        $("cfgDeciderUrl").value = "https://api.regolo.ai/v1/systemone";
        $("cfgDeciderModel").value = "clef-flash";
        setSlSelectValue($("cfgClsMethod"), "decider");
        syncDeciderVisibility();
        showToast("Wypełniono preset Regolo.ai (UE) — wpisz klucz API i zapisz");
      } else if (btn.dataset.preset === "ovh") {
        $("cfgLlmUrl").value = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1";
        $("cfgLlmModel").value = "Qwen3.5-9B";
        setSlSelectValue($("cfgClsMethod"), "llm");
        syncDeciderVisibility();
        showToast("Wypełniono preset OVHcloud EU — wpisz klucz API i zapisz");
      } else if (btn.dataset.preset === "openrouter") {
        $("cfgLlmUrl").value = "https://openrouter.ai/api/v1";
        $("cfgLlmModel").value = "qwen/qwen3.5-9b";
        setSlSelectValue($("cfgClsMethod"), "decider");
        syncDeciderVisibility();
        showToast("Wypełniono preset OpenRouter — kliknij Zapisz zmiany");
      }
    });
  });

  $("btnClearLlmKey").addEventListener("click", async () => {
    const res = await api("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clear_llm_api_key: true }),
    });
    renderKeyBadge($("llmKeyStatus"), $("cfgLlmKey"), Boolean(res.settings.llm_api_key_set), res.settings.llm_api_key_source);
    showToast("Usunięto zapisany klucz API LLM");
  });

  $("btnClearOrKey").addEventListener("click", async () => {
    const res = await api("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clear_openrouter_api_key: true }),
    });
    renderKeyBadge($("llmKeyStatus"), $("cfgLlmKey"), Boolean(res.settings.llm_api_key_set), res.settings.llm_api_key_source);
    renderKeyBadge($("orKeyStatus"), $("cfgOrKey"), Boolean(res.settings.openrouter_api_key_set));
    showToast("Usunięto zapisany klucz API OpenRouter");
  });

  $("btnSaveConfig").addEventListener("click", async () => {
    const btn = $("btnSaveConfig");
    const payload = {
      llm_base_url: ($("cfgLlmUrl").value || "").trim(),
      llm_model: ($("cfgLlmModel").value || "").trim(),
      llm_timeout: Number($("cfgLlmTimeout").value) || 300,
      classification_method: $("cfgClsMethod").value || "decider",
      fallback_to_rules: $("cfgFallbackRules").checked,
      decider_url: ($("cfgDeciderUrl").value || "").trim(),
      decider_model: ($("cfgDeciderModel").value || "").trim(),
      ocr_deskew: $("cfgDeskew").checked,
      ocr_render_width_px: Number($("cfgRenderWidth").value) || 1654,
      default_workers: Number($("cfgWorkers").value) || 4,
    };
    const llmKey = ($("cfgLlmKey").value || "").trim();
    const orKey = ($("cfgOrKey").value || "").trim();
    if (llmKey) payload.llm_api_key = llmKey;
    if (orKey) payload.openrouter_api_key = orKey;

    btn.loading = true;
    try {
      const res = await api("/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      // Immediately clear password inputs so keys are never visible after save
      renderKeyBadge($("llmKeyStatus"), $("cfgLlmKey"), Boolean(res.settings.llm_api_key_set), res.settings.llm_api_key_source);
      renderKeyBadge($("orKeyStatus"), $("cfgOrKey"), Boolean(res.settings.openrouter_api_key_set));
      await loadSysInfo();
      showToast("Konfiguracja zapisana i aktywna");
    } catch (err) {
      showToast(`Błąd zapisu: ${err.message}`);
    } finally {
      btn.loading = false;
    }
  });

  syncDrawers();
  loadSysInfo();
  refreshDocs();
})();
