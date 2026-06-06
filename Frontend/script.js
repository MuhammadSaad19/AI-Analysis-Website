 // ── CONFIG ──────────────────────────────────────────────────────────────────
  let API_BASE = localStorage.getItem('api_base') || 'http://localhost:8000';
  document.getElementById('api-url-input').value = API_BASE;

  function toggleApiPanel() {
    document.getElementById('api-panel').classList.toggle('open');
  }

  function saveApiUrl() {
    API_BASE = document.getElementById('api-url-input').value.trim().replace(/\/$/, '');
    localStorage.setItem('api_base', API_BASE);
    document.getElementById('api-panel').classList.remove('open');
    showToast('API URL saved ✓');
  }

  // ── TABS ────────────────────────────────────────────────────────────────────
  function switchTab(tab) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    document.querySelector(`.tab-btn[data-tab="${tab}"]`).classList.add('active');
    document.getElementById(`panel-${tab}`).classList.add('active');
  }

  // ── IMAGE CLASSIFIER ────────────────────────────────────────────────────────
  let selectedFile = null;

  function handleImageSelect(e) {
    const file = e.target.files[0];
    if (file) loadImageFile(file);
  }

  function handleDrag(e) {
    e.preventDefault();
    document.getElementById('upload-zone').classList.add('dragging');
  }

  function handleDragLeave() {
    document.getElementById('upload-zone').classList.remove('dragging');
  }

  function handleDrop(e) {
    e.preventDefault();
    document.getElementById('upload-zone').classList.remove('dragging');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) loadImageFile(file);
  }

  function loadImageFile(file) {
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = ev => {
      const preview = document.getElementById('image-preview');
      preview.src = ev.target.result;
      preview.style.display = 'block';
    };
    reader.readAsDataURL(file);
    document.getElementById('classify-btn').disabled = false;
    hideResult('img');
  }

  async function classifyImage() {
    if (!selectedFile) return;
    showLoader('img'); hideError('img'); hideResult('img');
    document.getElementById('classify-btn').disabled = true;

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const res = await fetch(`${API_BASE}/predict/image`, { method: 'POST', body: formData });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Server error');

      document.getElementById('img-pred-label').textContent = data.prediction;
      document.getElementById('img-pred-conf').textContent = `Confidence: ${data.confidence}%`;
      renderBars('img-bars', data.all_scores, 'purple');
      showResult('img');
    } catch (err) {
      showError('img', err.message);
    } finally {
      hideLoader('img');
      document.getElementById('classify-btn').disabled = false;
    }
  }

  // ── NLP MODEL ───────────────────────────────────────────────────────────────
  async function runNLP() {
    const text = document.getElementById('nlp-input').value.trim();
    if (!text) return;
    showLoader('nlp'); hideError('nlp'); hideResult('nlp');
    document.getElementById('nlp-btn').disabled = true;

    try {
      const res = await fetch(`${API_BASE}/predict/nlp`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Server error');

      document.getElementById('nlp-pred-label').textContent = data.prediction;
      document.getElementById('nlp-pred-conf').textContent = `Confidence: ${data.confidence}%`;
      renderBars('nlp-bars', data.all_scores, 'cyan');
      showResult('nlp');
    } catch (err) {
      showError('nlp', err.message);
    } finally {
      hideLoader('nlp');
      document.getElementById('nlp-btn').disabled = false;
    }
  }

  // ── GEMINI AGENT ─────────────────────────────────────────────────────────────
  function handleChatKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendToAgent();
    }
  }

  async function sendToAgent() {
    const input = document.getElementById('agent-input');
    const message = input.value.trim();
    if (!message) return;

    appendMsg('user', message);
    input.value = '';
    input.style.height = 'auto';
    showLoader('agent');
    hideError('agent');
    document.getElementById('agent-btn').disabled = true;

    try {
      const res = await fetch(`${API_BASE}/chat/gemini`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Server error');
      appendMsg('ai', data.reply);
    } catch (err) {
      showError('agent', err.message);
    } finally {
      hideLoader('agent');
      document.getElementById('agent-btn').disabled = false;
    }
  }

  async function resetChat() {
    try {
      await fetch(`${API_BASE}/chat/gemini/reset`, { method: 'POST' });
    } catch (_) {}
    const win = document.getElementById('chat-window');
    win.innerHTML = `
      <div class="chat-empty" id="chat-empty">
        <div class="emoji">🤖</div>
        <div>Say something to start the conversation</div>
      </div>`;
  }

  function appendMsg(role, text) {
    const win = document.getElementById('chat-window');
    const empty = document.getElementById('chat-empty');
    if (empty) empty.remove();

    const avatar = role === 'user' ? '🧑' : '✨';
    const div = document.createElement('div');
    div.className = `msg ${role}`;
    div.innerHTML = `
      <div class="msg-avatar">${avatar}</div>
      <div class="msg-bubble">${escapeHtml(text).replace(/\n/g, '<br>')}</div>`;
    win.appendChild(div);
    win.scrollTop = win.scrollHeight;
  }

  function escapeHtml(s) {
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }

  // ── BARS ────────────────────────────────────────────────────────────────────
  function renderBars(containerId, scores, color) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    Object.entries(scores).forEach(([label, val]) => {
      const row = document.createElement('div');
      row.className = 'bar-row';
      row.innerHTML = `
        <div class="bar-label-row"><span>${label}</span><span>${val}%</span></div>
        <div class="bar-track"><div class="bar-fill ${color}" style="width:0%"></div></div>`;
      container.appendChild(row);
      setTimeout(() => {
        row.querySelector('.bar-fill').style.width = `${val}%`;
      }, 50);
    });
  }

  // ── HELPERS ─────────────────────────────────────────────────────────────────
  function showLoader(id)  { document.getElementById(`${id}-loader`).classList.add('visible'); }
  function hideLoader(id)  { document.getElementById(`${id}-loader`).classList.remove('visible'); }
  function showResult(id)  { document.getElementById(`${id}-result`).classList.add('visible'); }
  function hideResult(id)  { const el = document.getElementById(`${id}-result`); if(el) el.classList.remove('visible'); }
  function showError(id, msg) {
    const el = document.getElementById(`${id}-error`);
    el.textContent = `Error: ${msg}`;
    el.classList.add('visible');
  }
  function hideError(id) { document.getElementById(`${id}-error`).classList.remove('visible'); }

  function showToast(msg) {
    const t = document.createElement('div');
    t.style.cssText = 'position:fixed;bottom:80px;right:20px;background:#10b981;color:#fff;padding:10px 18px;border-radius:8px;font-size:13px;font-family:DM Mono,monospace;z-index:999;animation:fadeUp 0.3s ease';
    t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 2500);
  }

  // Auto-resize chat textarea
  document.getElementById('agent-input').addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 120) + 'px';
  });