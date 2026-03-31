const API_URL = 'http://localhost:5000/api';

/* ── CANVAS PARTICLE BG ── */
(function initCanvas() {
  const canvas = document.getElementById('bg-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let particles = [];
  function resize() { canvas.width = window.innerWidth; canvas.height = window.innerHeight; }
  window.addEventListener('resize', resize);
  resize();
  for (let i = 0; i < 60; i++) {
    particles.push({ x: Math.random()*canvas.width, y: Math.random()*canvas.height, r: Math.random()*2+0.5, dx: (Math.random()-0.5)*0.3, dy: (Math.random()-0.5)*0.3, alpha: Math.random()*0.5+0.1 });
  }
  function draw() {
    ctx.clearRect(0,0,canvas.width,canvas.height);
    particles.forEach(p => { ctx.beginPath(); ctx.arc(p.x,p.y,p.r,0,Math.PI*2); ctx.fillStyle=`rgba(91,33,182,${p.alpha})`; ctx.fill(); p.x+=p.dx; p.y+=p.dy; if(p.x<0||p.x>canvas.width) p.dx*=-1; if(p.y<0||p.y>canvas.height) p.dy*=-1; });
    requestAnimationFrame(draw);
  }
  draw();
})();

/* ── STICKY HEADER ── */
const header = document.getElementById('site-header');
if (header) window.addEventListener('scroll', () => header.classList.toggle('scrolled', window.scrollY > 50));

/* ── HAMBURGER ── */
const hamburger = document.getElementById('hamburger');
const mainNav = document.getElementById('main-nav');
if (hamburger) hamburger.addEventListener('click', () => {
  mainNav.classList.toggle('open');
  hamburger.setAttribute('aria-expanded', mainNav.classList.contains('open'));
});

/* ── ACTIVE NAV LINK ── */
const sections = document.querySelectorAll('section[id]');
const navLinks = document.querySelectorAll('.nav-link');
const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) navLinks.forEach(link => link.classList.toggle('active', link.getAttribute('href') === '#' + entry.target.id));
  });
}, { threshold: 0.4 });
sections.forEach(s => observer.observe(s));

/* ── REVEAL ON SCROLL ── */
const revealEls = document.querySelectorAll('.skin-card, .about-card, .disease-card, .team-card');
revealEls.forEach((el, i) => { el.classList.add('reveal'); el.style.transitionDelay = `${(i%4)*0.08}s`; });
const revealObs = new IntersectionObserver(entries => {
  entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('visible'); });
}, { threshold: 0.1 });
document.querySelectorAll('.reveal').forEach(el => revealObs.observe(el));

/* ── TABS ── */
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
  });
});

/* ── SCANNER ── */
window.addEventListener('load', () => {
  const fileInput     = document.getElementById('file-input');
  const chooseFileBtn = document.getElementById('choose-file-btn');
  const changeFileBtn = document.getElementById('change-file-btn');
  const uploadZone    = document.getElementById('upload-zone');
  const previewZone   = document.getElementById('preview-zone');
  const loadingZone   = document.getElementById('loading-zone');
  const resultsZone   = document.getElementById('results-zone');
  const previewImg    = document.getElementById('preview-img');
  const fileNameDisplay = document.getElementById('file-name-display');
  const analyzeBtn    = document.getElementById('analyze-btn');
  const loadingText   = document.getElementById('loading-text');

  let selectedFile = null;

  const loadingMessages = ['Preprocessing image…','Running CNN inference…','Classifying skin condition…','Generating recommendations…'];

  function showZone(zone) {
    [uploadZone, previewZone, loadingZone, resultsZone].forEach(z => { if (z) z.style.display = 'none'; });
    if (zone) {
      zone.style.display = zone.id === 'results-zone' ? 'flex' : 'block';
      zone.style.flexDirection = 'column';
      zone.style.gap = '12px';
    }
  }

  function handleFile(file) {
    if (!file) return;
    const allowed = ['image/png','image/jpeg','image/jpg'];
    if (!allowed.includes(file.type)) { showToast('Please choose a PNG or JPEG image.','error'); return; }
    if (file.size > 16*1024*1024) { showToast('File must be under 16 MB.','error'); return; }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = e => {
      previewImg.src = e.target.result;
      fileNameDisplay.textContent = file.name;
      showZone(previewZone);
    };
    reader.readAsDataURL(file);
  }

  chooseFileBtn.addEventListener('click', (e) => { e.stopPropagation(); fileInput.click(); });
  changeFileBtn.addEventListener('click', (e) => { e.stopPropagation(); fileInput.click(); });
  uploadZone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', (e) => { handleFile(e.target.files[0]); });

  uploadZone.addEventListener('dragover', e => { e.preventDefault(); });
  uploadZone.addEventListener('drop', e => { e.preventDefault(); handleFile(e.dataTransfer.files[0]); });

  analyzeBtn.onclick = async function() {
  if (!selectedFile) return;
  showZone(loadingZone);
  try {
    const fd = new FormData();
    fd.append('file', selectedFile);
    const res = await fetch('http://localhost:5000/api/predict', {
      method: 'POST',
      body: fd
    });
    const data = await res.json();
    if (data.success) displayResults(data);
    else showZone(previewZone);
  } catch(e) {
    console.error(e);
    showZone(previewZone);
  }
};

  const severityColors = {
    Melanoma:             { color:'#DC2626', icon:'fas fa-radiation', level:'CRITICAL' },
    Actinic_keratosis:    { color:'#D97706', icon:'fas fa-sun',       level:'HIGH' },
    Basal_cell_carcinoma: { color:'#D97706', icon:'fas fa-disease',   level:'HIGH' },
    Melanocytic_nevus:    { color:'#2563EB', icon:'fas fa-circle',    level:'MODERATE' },
    Benign_keratosis:     { color:'#16A34A', icon:'fas fa-leaf',      level:'LOW' },
    Dermatofibroma:       { color:'#16A34A', icon:'fas fa-dot-circle',level:'LOW' },
    Vascular_lesion:      { color:'#16A34A', icon:'fas fa-tint',      level:'LOW' },
  };

  function displayResults(data) {
    const meta = severityColors[data.prediction] || { color:'#7C3AED', icon:'fas fa-search', level:'UNKNOWN' };
    const rec = data.recommendations || {};
    const tips = (rec.care_tips || []).map(t => `<li>${t}</li>`).join('');
    resultsZone.innerHTML = `
      <div class="result-header">
        <div class="result-icon"><i class="${meta.icon}" style="color:${meta.color}"></i></div>
        <div>
          <div class="result-condition">${data.prediction.replace(/_/g,' ')}</div>
          <div class="result-confidence">Confidence: <strong>${data.confidence}</strong> · Severity: ${meta.level}</div>
        </div>
      </div>
      <div class="result-block"><h4><i class="fas fa-file-medical"></i> Description</h4><p>${rec.description||'—'}</p></div>
      <div class="result-block"><h4><i class="fas fa-notes-medical"></i> Care Tips</h4><ul>${tips}</ul></div>
      <div class="result-doctor"><i class="fas fa-stethoscope"></i><div><p>When to see a doctor</p><p>${rec.when_to_see_doctor||'Consult a dermatologist.'}</p></div></div>
      <button class="result-retry-btn" id="retry-btn"><i class="fas fa-redo"></i> Analyze Another Image</button>
    `;
    showZone(resultsZone);
    resultsZone.scrollIntoView({ behavior:'smooth', block:'nearest' });
    document.getElementById('retry-btn').addEventListener('click', () => {
      selectedFile = null;
      fileInput.value = '';
      showZone(uploadZone);
    });
  }

  /* ── CONTACT FORM ── */
  const contactForm = document.getElementById('contact-form');
  if (contactForm) {
    contactForm.addEventListener('submit', async e => {
      e.preventDefault();
      const submitBtn = contactForm.querySelector('button[type="submit"]');
      const originalHTML = submitBtn.innerHTML;
      const inputs = contactForm.querySelectorAll('input, textarea');
      const name = inputs[0].value.trim(), email = inputs[1].value.trim(), phone = inputs[2].value.trim(), message = inputs[3].value.trim();
      if (!name||!email||!message) { showToast('Please fill in name, email and message.','error'); return; }
      submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Sending…';
      submitBtn.disabled = true;
      try {
        const res = await fetch(`${API_URL}/contact`, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({name,email,phone,message}) });
        const data = await res.json();
        if (res.ok && data.success) { showToast(`Thanks, ${name}! Message sent! 📧`,'success'); contactForm.reset(); }
        else throw new Error(data.error||'Something went wrong.');
      } catch (err) {
        showToast('Error: ' + err.message,'error');
      } finally {
        submitBtn.innerHTML = originalHTML;
        submitBtn.disabled = false;
      }
    });
  }

  /* ── BACKEND HEALTH CHECK ── */
  fetch(API_URL.replace('/api','') + '/')
    .then(r => r.json())
    .then(d => { console.log('✅ Backend connected:', d.message); if (d.tensorflow_available) console.log('✅ TensorFlow model loaded!'); })
    .catch(() => console.warn('⚠️ Backend not reachable.'));
});

/* ── TOAST ── */
function showToast(msg, type='info') {
  const existing = document.querySelector('.toast');
  if (existing) existing.remove();
  const toast = document.createElement('div');
  toast.className = 'toast toast-' + type;
  const icon = type==='success' ? 'fa-check-circle' : type==='error' ? 'fa-exclamation-circle' : 'fa-info-circle';
  toast.innerHTML = `<i class="fas ${icon}"></i><span>${msg}</span>`;
  Object.assign(toast.style, { position:'fixed', bottom:'24px', right:'24px', zIndex:'9999', display:'flex', alignItems:'flex-start', gap:'10px', background: type==='success'?'#064E3B':type==='error'?'#450A0A':'#1E1144', color:'#fff', padding:'14px 20px', borderRadius:'12px', boxShadow:'0 8px 32px rgba(0,0,0,0.25)', maxWidth:'340px', fontSize:'0.88rem', lineHeight:'1.5', borderLeft:`3px solid ${type==='success'?'#34D399':type==='error'?'#F87171':'#A78BFA'}` });
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}