/**
 * MultiOrganAI - Interactive Organ Selector, Form Generator & AI Predictor
 */

const ORGAN_DEFINITIONS = {
  kidney: {
    title: 'Kidney Health Assessment',
    description: 'Evaluate glomerular filtration, electrolytes, blood urea, and proteinuria indicators.',
    icon: '/static/img/icons/kidney.svg',
    color: '#0284c7',
    fields: [
      { id: 'age', label: 'Patient Age', unit: 'years', placeholder: 'e.g. 48', default: 45 },
      { id: 'bp', label: 'Diastolic BP', unit: 'mmHg', placeholder: '60 - 120', default: 80 },
      { id: 'sc', label: 'Serum Creatinine', unit: 'mg/dL', placeholder: '0.6 - 5.0', default: 0.9 },
      { id: 'bu', label: 'Blood Urea', unit: 'mg/dL', placeholder: '15 - 120', default: 32 },
      { id: 'al', label: 'Urine Albumin', unit: 'grade (0-4)', placeholder: '0 (Normal) to 4', default: 0 },
      { id: 'su', label: 'Urine Sugar', unit: 'grade (0-4)', placeholder: '0 (Normal) to 4', default: 0 },
      { id: 'bgr', label: 'Random Blood Glucose', unit: 'mg/dL', placeholder: '70 - 300', default: 110 },
      { id: 'hemo', label: 'Hemoglobin', unit: 'g/dL', placeholder: '6.0 - 18.0', default: 14.2 },
      { id: 'sod', label: 'Serum Sodium', unit: 'mEq/L', placeholder: '125 - 150', default: 138 },
      { id: 'pot', label: 'Serum Potassium', unit: 'mEq/L', placeholder: '3.0 - 6.5', default: 4.3 },
      { id: 'sg', label: 'Specific Gravity', unit: '', placeholder: '1.005 - 1.025', default: 1.020 },
      { id: 'wbcc', label: 'WBC Count', unit: '/uL', placeholder: '4000 - 18000', default: 7200 }
    ]
  },
  liver: {
    title: 'Hepatic Function & Liver Disease Profiling',
    description: 'Analyze hepatic enzymes (ALT, AST, ALP), bilirubin metabolism, and protein synthesis.',
    icon: '/static/img/icons/liver.svg',
    color: '#d97706',
    fields: [
      { id: 'age', label: 'Patient Age', unit: 'years', placeholder: 'e.g. 42', default: 42 },
      { id: 'gender', label: 'Gender', unit: '1=M, 0=F', type: 'select', options: [{val: 1, text: 'Male'}, {val: 0, text: 'Female'}], default: 1 },
      { id: 'total_bilirubin', label: 'Total Bilirubin', unit: 'mg/dL', placeholder: '0.3 - 15.0', default: 0.8 },
      { id: 'direct_bilirubin', label: 'Direct Bilirubin', unit: 'mg/dL', placeholder: '0.1 - 8.0', default: 0.2 },
      { id: 'alkaline_phosphotase', label: 'Alkaline Phosphatase (ALP)', unit: 'IU/L', placeholder: '60 - 600', default: 185 },
      { id: 'alamine_aminotransferase', label: 'ALT (SGPT)', unit: 'IU/L', placeholder: '10 - 300', default: 28 },
      { id: 'aspartate_aminotransferase', label: 'AST (SGOT)', unit: 'IU/L', placeholder: '10 - 350', default: 30 },
      { id: 'total_proteins', label: 'Total Proteins', unit: 'g/dL', placeholder: '4.0 - 9.0', default: 7.1 },
      { id: 'albumin', label: 'Serum Albumin', unit: 'g/dL', placeholder: '1.5 - 5.5', default: 4.0 },
      { id: 'albumin_and_globulin_ratio', label: 'A/G Ratio', unit: '', placeholder: '0.4 - 2.5', default: 1.25 }
    ]
  }
};

let currentOrgan = 'kidney';

function initOrganSelector() {
  const container = document.getElementById('organTesterFields');
  if (!container) return;

  // Tab click listeners
  const tabs = document.querySelectorAll('.organ-tab-btn');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      currentOrgan = tab.dataset.organ;
      renderFields(currentOrgan);
    });
  });

  // Sample data button listeners
  const btnNormal = document.getElementById('btnLoadNormal');
  const btnAbnormal = document.getElementById('btnLoadAbnormal');

  if (btnNormal) {
    btnNormal.addEventListener('click', () => loadSampleData(currentOrgan, 'normal'));
  }
  if (btnAbnormal) {
    btnAbnormal.addEventListener('click', () => loadSampleData(currentOrgan, 'abnormal'));
  }

  // Form submission listener
  const form = document.getElementById('organAnalysisForm');
  if (form) {
    form.addEventListener('submit', handleFormSubmit);
  }

  // OCR Upload setup
  initOcrUploader();

  // Initial render
  renderFields(currentOrgan);
}

function renderFields(organ) {
  const container = document.getElementById('organTesterFields');
  const organConfig = ORGAN_DEFINITIONS[organ];
  if (!container || !organConfig) return;

  let html = '';
  organConfig.fields.forEach(field => {
    html += `<div class="form-group">
      <label for="${field.id}">
        ${field.label}
        ${field.unit ? `<span>(${field.unit})</span>` : ''}
      </label>`;

    if (field.type === 'select') {
      html += `<select id="${field.id}" name="${field.id}" class="form-select">`;
      field.options.forEach(opt => {
        const selected = (opt.val === field.default) ? 'selected' : '';
        html += `<option value="${opt.val}" ${selected}>${opt.text}</option>`;
      });
      html += `</select>`;
    } else {
      html += `<input type="number" step="any" id="${field.id}" name="${field.id}" class="form-input" placeholder="${field.placeholder || ''}" value="${field.default || ''}">`;
    }

    html += `</div>`;
  });

  container.innerHTML = html;

  // Hide previous results
  const resultsCard = document.getElementById('analysisResultsCard');
  if (resultsCard) resultsCard.style.display = 'none';
}

async function loadSampleData(organ, caseType) {
  try {
    const res = await fetch(`/api/sample-data/${organ}/${caseType}`);
    const json = await res.json();
    if (json.data) {
      for (const [key, val] of Object.entries(json.data)) {
        const el = document.getElementById(key);
        if (el) el.value = val;
      }
      if (window.showToast) {
        window.showToast(`Loaded ${caseType.toUpperCase()} sample for ${organ.toUpperCase()}`, 'info');
      }
    }
  } catch (err) {
    console.error('Error loading sample data:', err);
  }
}

async function handleFormSubmit(e) {
  e.preventDefault();
  const submitBtn = document.getElementById('btnSubmitAnalysis');
  const originalText = submitBtn ? submitBtn.innerHTML : 'Run AI Analysis';

  if (submitBtn) {
    submitBtn.innerHTML = `
      <svg class="spinner" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="10" stroke-dasharray="32" stroke-dashoffset="12"/>
      </svg>
      Analyzing Biomarkers...
    `;
    submitBtn.disabled = true;
  }

  const formData = new FormData(e.target);
  const data = {};
  formData.forEach((value, key) => { data[key] = value; });

  try {
    const res = await fetch(`/api/predict/${currentOrgan}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });

    const result = await res.json();

    if (!res.ok) {
      alert(result.error || 'Analysis failed. Please verify parameters.');
      return;
    }

    displayResults(result);

    if (window.showToast) {
      window.showToast(`AI Analysis completed for ${currentOrgan.toUpperCase()}!`, 'success');
    }
  } catch (err) {
    console.error('Prediction error:', err);
    alert('Failed to connect to the prediction server.');
  } finally {
    if (submitBtn) {
      submitBtn.innerHTML = originalText;
      submitBtn.disabled = false;
    }
  }
}

function displayResults(data) {
  const card = document.getElementById('analysisResultsCard');
  if (!card) return;

  const pred = data.prediction;
  const analysis = data.analysis;

  // Gauge colors and values
  const score = pred.risk_score;
  const riskLevel = pred.risk_level;

  let gaugeBg = 'radial-gradient(circle, #ecfdf5 40%, #d1fae5 100%)';
  let gaugeColor = '#059669';
  let badgeClass = 'badge-normal';

  if (riskLevel === 'High') {
    gaugeBg = 'radial-gradient(circle, #fef2f2 40%, #fee2e2 100%)';
    gaugeColor = '#dc2626';
    badgeClass = 'badge-critical';
  } else if (riskLevel === 'Moderate') {
    gaugeBg = 'radial-gradient(circle, #fffbeb 40%, #fef3c7 100%)';
    gaugeColor = '#d97706';
    badgeClass = 'badge-warning';
  }

  const gauge = document.getElementById('resultsRiskGauge');
  if (gauge) {
    gauge.style.background = gaugeBg;
    gauge.style.border = `3px solid ${gaugeColor}`;
    gauge.innerHTML = `
      <div class="gauge-score-num" style="color: ${gaugeColor};">${score}%</div>
      <div class="gauge-score-label" style="color: ${gaugeColor};">${riskLevel} Risk</div>
    `;
  }

  // Status & Label
  const labelEl = document.getElementById('resultsLabel');
  if (labelEl) labelEl.textContent = pred.label;

  const confidenceEl = document.getElementById('resultsConfidence');
  if (confidenceEl) confidenceEl.textContent = `Confidence: ${pred.confidence}%`;

  const summaryEl = document.getElementById('resultsSummary');
  if (summaryEl) summaryEl.textContent = analysis.summary;

  // Biomarker table
  const tableBody = document.getElementById('resultsBiomarkersTable');
  if (tableBody) {
    let rowsHtml = '';
    analysis.parameters.forEach(p => {
      rowsHtml += `<tr>
        <td><strong>${p.name}</strong></td>
        <td>${p.value} ${p.unit}</td>
        <td>${p.reference_range}</td>
        <td><span class="badge ${p.css_class}">${p.status}</span></td>
      </tr>`;
    });
    tableBody.innerHTML = rowsHtml;
  }

  // Recommendations checklist
  const recsList = document.getElementById('resultsRecommendationsList');
  if (recsList) {
    let recsHtml = '';
    analysis.recommendations.forEach(r => {
      recsHtml += `<li>${r}</li>`;
    });
    recsList.innerHTML = recsHtml;
  }

  // PDF download button
  const pdfBtn = document.getElementById('resultsPdfBtn');
  if (pdfBtn && data.pdf_url) {
    pdfBtn.href = data.pdf_url;
    pdfBtn.style.display = 'inline-flex';
  }

  card.style.display = 'block';
  card.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function initOcrUploader() {
  const dropzone = document.getElementById('ocrDropzone');
  const fileInput = document.getElementById('ocrFileInput');
  if (!dropzone || !fileInput) return;

  dropzone.addEventListener('click', () => fileInput.click());

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
      fileInput.files = e.dataTransfer.files;
      uploadMedicalReport(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) {
      uploadMedicalReport(fileInput.files[0]);
    }
  });
}

async function uploadMedicalReport(file) {
  const dropzone = document.getElementById('ocrDropzone');
  const originalHtml = dropzone ? dropzone.innerHTML : '';

  if (dropzone) {
    dropzone.innerHTML = `
      <div style="padding: 20px;">
        <div style="font-size: 24px; margin-bottom: 8px;">⏳</div>
        <strong>Extracting medical biomarkers with OCR...</strong>
        <p style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">Analyzing lab parameters for Kidney, Liver, and Heart</p>
      </div>
    `;
  }

  const formData = new FormData();
  formData.append('report_file', file);

  try {
    const res = await fetch('/api/upload-report', {
      method: 'POST',
      body: formData
    });

    const result = await res.json();

    if (!res.ok) {
      alert(result.error || 'Failed to parse lab report.');
      if (dropzone) dropzone.innerHTML = originalHtml;
      return;
    }

    // Switch tab to detected organ
    currentOrgan = result.detected_organ;
    const tabs = document.querySelectorAll('.organ-tab-btn');
    tabs.forEach(t => {
      if (t.dataset.organ === currentOrgan) t.classList.add('active');
      else t.classList.remove('active');
    });

    renderFields(currentOrgan);

    // Populate parsed parameters into fields
    if (result.parsed_parameters) {
      for (const [key, val] of Object.entries(result.parsed_parameters)) {
        const el = document.getElementById(key);
        if (el) el.value = val;
      }
    }

    displayResults(result);

    if (window.showToast) {
      window.showToast(`Extracted report parameters for ${result.detected_organ.toUpperCase()}`, 'success');
    }

    if (dropzone) {
      dropzone.innerHTML = `
        <div style="padding: 10px; color: var(--primary);">
          ✓ Report parsed: <strong>${file.name}</strong> (${result.detected_organ.toUpperCase()})
          <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">Click to upload another report image or PDF</div>
        </div>
      `;
    }
  } catch (err) {
    console.error('OCR Upload Error:', err);
    alert('Error uploading medical report.');
    if (dropzone) dropzone.innerHTML = originalHtml;
  }
}

document.addEventListener('DOMContentLoaded', initOrganSelector);
