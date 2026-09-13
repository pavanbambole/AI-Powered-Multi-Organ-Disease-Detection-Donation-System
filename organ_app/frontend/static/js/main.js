/**
 * MultiOrganAI - Main UI Interactivity & Helper Scripts
 */

// Toast notification helper
window.showToast = function(message, type = 'info') {
  let container = document.getElementById('toastContainer');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toastContainer';
    container.style.cssText = `
      position: fixed;
      bottom: 24px;
      right: 24px;
      z-index: 9999;
      display: flex;
      flex-direction: column;
      gap: 10px;
    `;
    document.body.appendChild(container);
  }

  const toast = document.createElement('div');
  let bg = '#0f172a';
  let border = '#10b981';
  let icon = '✓';

  if (type === 'danger' || type === 'error') {
    bg = '#991b1b';
    border = '#ef4444';
    icon = '✕';
  } else if (type === 'warning') {
    bg = '#92400e';
    border = '#f59e0b';
    icon = '⚠';
  }

  toast.style.cssText = `
    background: ${bg};
    color: #ffffff;
    border-left: 4px solid ${border};
    padding: 12px 20px;
    border-radius: 8px;
    font-size: 13.5px;
    font-weight: 500;
    box-shadow: 0 10px 25px rgba(0,0,0,0.25);
    display: flex;
    align-items: center;
    gap: 10px;
    transform: translateX(120%);
    transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1);
  `;
  toast.innerHTML = `<span>${icon}</span> <span>${message}</span>`;
  container.appendChild(toast);

  // Slide in
  requestAnimationFrame(() => {
    toast.style.transform = 'translateX(0)';
  });

  // Slide out and remove
  setTimeout(() => {
    toast.style.transform = 'translateX(120%)';
    setTimeout(() => toast.remove(), 350);
  }, 4000);
};

document.addEventListener('DOMContentLoaded', () => {
  // Mobile menu toggle
  const mobileBtn = document.getElementById('mobileMenuBtn');
  const navLinks = document.querySelector('.nav-links');
  if (mobileBtn && navLinks) {
    mobileBtn.addEventListener('click', () => {
      const isVisible = navLinks.style.display === 'flex';
      navLinks.style.display = isVisible ? 'none' : 'flex';
      if (!isVisible) {
        navLinks.style.flexDirection = 'column';
        navLinks.style.position = 'absolute';
        navLinks.style.top = '74px';
        navLinks.style.left = '0';
        navLinks.style.width = '100%';
        navLinks.style.background = 'var(--bg-surface)';
        navLinks.style.padding = '20px';
        navLinks.style.borderBottom = '1px solid var(--border-color)';
        navLinks.style.boxShadow = 'var(--shadow-lg)';
      }
    });
  }

  // Smooth scroll for in-page anchors
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
      const targetId = this.getAttribute('href');
      if (targetId && targetId !== '#') {
        const target = document.querySelector(targetId);
        if (target) {
          e.preventDefault();
          target.scrollIntoView({ behavior: 'smooth' });
        }
      }
    });
  });

  // Donor registration form handler
  const donorForm = document.getElementById('donorRegistrationForm');
  if (donorForm) {
    donorForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const formData = new FormData(donorForm);
      const data = {};
      formData.forEach((val, key) => data[key] = val);

      try {
        const res = await fetch('/api/donor/register', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data)
        });
        const result = await res.json();
        if (res.ok) {
          window.showToast(result.message, 'success');
          donorForm.reset();
          setTimeout(() => location.reload(), 1500);
        } else {
          alert(result.error || 'Failed to register donor pledge.');
        }
      } catch (err) {
        alert('Network error while registering donor.');
      }
    });
  }

  // Recipient request form handler
  const recipientForm = document.getElementById('recipientRegistrationForm');
  if (recipientForm) {
    recipientForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const formData = new FormData(recipientForm);
      const data = {};
      formData.forEach((val, key) => data[key] = val);

      try {
        const res = await fetch('/api/recipient/register', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data)
        });
        const result = await res.json();
        if (res.ok) {
          window.showToast(result.message, 'success');
          recipientForm.reset();
          setTimeout(() => location.reload(), 1500);
        } else {
          alert(result.error || 'Failed to submit organ request.');
        }
      } catch (err) {
        alert('Network error while submitting organ request.');
      }
    });
  }
});

/**
 * Handle "Generate Sample PDF" with loading feedback, error handling, and auto-preview
 */
window.handleGenerateSamplePdf = async function(e, btn) {
  if (e) e.preventDefault();
  if (!btn) btn = document.getElementById('generateSamplePdfBtn');
  if (!btn) return;
  if (btn.dataset.loading === 'true') return;

  const originalContent = btn.innerHTML;
  btn.dataset.loading = 'true';
  btn.style.pointerEvents = 'none';
  btn.innerHTML = `<span style="display:inline-block; width:13px; height:13px; border:2px solid currentColor; border-right-color:transparent; border-radius:50%; animation:spin 0.8s linear infinite; margin-right:6px; vertical-align:middle;"></span> Generating PDF...`;

  try {
    const res = await fetch('/api/generate-sample-pdf?format=json', {
      headers: { 'Accept': 'application/json' }
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.error || `Server responded with status ${res.status}`);
    }

    const data = await res.json();
    if (data.success && (data.pdf_url || data.download_url)) {
      const targetUrl = data.pdf_url || data.download_url;
      const opened = window.open(targetUrl, '_blank');
      if (!opened) {
        // Fallback if browser popup blocker triggers
        window.location.href = targetUrl;
      }
      if (typeof window.showToast === 'function') {
        window.showToast('Clinical sample PDF report generated successfully!', 'success');
      }
    } else {
      throw new Error(data.error || 'Failed to generate clinical PDF report.');
    }
  } catch (err) {
    console.error('Sample PDF generation error:', err);
    if (typeof window.showToast === 'function') {
      window.showToast(err.message || 'Unable to generate sample PDF. Please try again.', 'danger');
    } else {
      alert(err.message || 'Unable to generate sample PDF. Please try again.');
    }
  } finally {
    btn.innerHTML = originalContent;
    btn.dataset.loading = 'false';
    btn.style.pointerEvents = 'auto';
  }
};
