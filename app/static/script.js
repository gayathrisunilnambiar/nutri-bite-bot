/**
 * Nutri-Bite Bot - Frontend JavaScript
 * Handles form submission, image upload, and results display
 */

document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const analyzeBtn = document.getElementById('analyzeBtn');
    const btnText = analyzeBtn.querySelector('.btn-text');
    const btnLoader = analyzeBtn.querySelector('.btn-loader');
    const results = document.getElementById('results');

    // Image upload elements
    const dropZone = document.getElementById('dropZone');
    const imageInput = document.getElementById('imageInput');
    const imagePreview = document.getElementById('imagePreview');
    const previewImg = document.getElementById('previewImg');
    const removeImage = document.getElementById('removeImage');

    let uploadedImageBase64 = null;

    // ===== Image Upload Handling =====

    dropZone.addEventListener('click', () => imageInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleImageFile(files[0]);
        }
    });

    imageInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleImageFile(e.target.files[0]);
        }
    });

    removeImage.addEventListener('click', () => {
        uploadedImageBase64 = null;
        imagePreview.classList.add('hidden');
        dropZone.classList.remove('hidden');
        imageInput.value = '';
    });

    function handleImageFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            // Store base64 (remove data URL prefix)
            const base64 = e.target.result.split(',')[1];
            uploadedImageBase64 = base64;

            // Show preview
            previewImg.src = e.target.result;
            imagePreview.classList.remove('hidden');
            dropZone.classList.add('hidden');
        };
        reader.readAsDataURL(file);
    }

    // ===== Form Submission =====

    analyzeBtn.addEventListener('click', async () => {
        // Gather form data
        const labValues = {
            egfr: parseFloatOrNull(document.getElementById('egfr').value),
            creatinine: parseFloatOrNull(document.getElementById('creatinine').value),
            potassium: parseFloatOrNull(document.getElementById('potassium').value),
            sodium: parseFloatOrNull(document.getElementById('sodium').value),
            glucose: parseFloatOrNull(document.getElementById('glucose').value),
            hba1c: parseFloatOrNull(document.getElementById('hba1c').value)
        };

        const conditions = {
            diabetes_t1: document.getElementById('diabetes_t1').checked,
            hypertension: document.getElementById('hypertension').checked,
            ckd: document.getElementById('ckd').checked
        };

        const ingredientsText = document.getElementById('ingredientsText').value.trim();

        // Validation
        if (!labValues.egfr && !labValues.potassium) {
            alert('Please enter at least eGFR or Potassium lab values');
            return;
        }

        if (!uploadedImageBase64 && !ingredientsText) {
            alert('Please upload a pantry image or enter ingredients');
            return;
        }

        // Build request
        const requestBody = {
            lab_values: labValues,
            conditions: conditions
        };

        if (uploadedImageBase64) {
            requestBody.pantry_image = uploadedImageBase64;
        }
        if (ingredientsText) {
            requestBody.ingredients_text = ingredientsText;
        }

        // Show loading state
        btnText.classList.add('hidden');
        btnLoader.classList.remove('hidden');
        analyzeBtn.disabled = true;

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestBody)
            });

            const data = await response.json();

            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }

            // Display results
            displayResults(data);

        } catch (error) {
            console.error('Error:', error);
            alert('Failed to analyze. Please try again.');
        } finally {
            // Reset button
            btnText.classList.remove('hidden');
            btnLoader.classList.add('hidden');
            analyzeBtn.disabled = false;
        }
    });

    // ===== Results Display =====

    function displayResults(data) {
        results.classList.remove('hidden');

        // Scroll to results
        results.scrollIntoView({ behavior: 'smooth' });

        // Display CBC summary
        displayCBCSummary(data.cbc_analysis);

        // Display daily limits
        displayDailyLimits(data.daily_limits);

        // Display recommendations
        displayRecommendations(data.recommendations);
    }

    function displayCBCSummary(analysis) {
        const cbcSummary = document.getElementById('cbcSummary');
        const alertsDiv = document.getElementById('alerts');

        // Lab values grid
        let labHtml = '<div class="lab-grid">';

        for (const [key, value] of Object.entries(analysis.lab_values || {})) {
            const statusClass = value.status.toLowerCase();
            labHtml += `
                <div class="lab-item">
                    <div class="name">${formatLabName(key)}</div>
                    <div class="value">${value.value} ${value.unit}</div>
                    <span class="status ${statusClass}">${value.status}</span>
                </div>
            `;
        }

        labHtml += '</div>';

        // Add CKD stage if present
        if (analysis.ckd_stage) {
            labHtml += `
                <div style="margin-top: 1rem; padding: 0.75rem; background: var(--bg-input); border-radius: 8px;">
                    <strong>CKD Stage:</strong> ${analysis.ckd_stage}
                </div>
            `;
        }

        cbcSummary.innerHTML = labHtml;

        // Alerts
        let alertsHtml = '';

        for (const alert of (analysis.alerts || [])) {
            const levelClass = alert.level === 'critical' ? 'alert-critical' :
                alert.level === 'high' ? 'alert-high' : 'alert-moderate';
            const icon = alert.level === 'critical' ? '🚨' :
                alert.level === 'high' ? '⚠️' : 'ℹ️';

            alertsHtml += `
                <div class="alert ${levelClass}">
                    <span class="alert-icon">${icon}</span>
                    <div class="alert-content">
                        <div class="alert-message">${alert.message}</div>
                        <div class="alert-action">→ ${alert.action}</div>
                    </div>
                </div>
            `;
        }

        alertsDiv.innerHTML = alertsHtml;
    }

    function displayDailyLimits(limits) {
        const limitsDiv = document.getElementById('dailyLimits');

        limitsDiv.innerHTML = `
            <div class="limit-item">
                <div class="value">${limits.potassium_mg}</div>
                <div class="label">Potassium (mg/day)</div>
            </div>
            <div class="limit-item">
                <div class="value">${limits.sodium_mg}</div>
                <div class="label">Sodium (mg/day)</div>
            </div>
            <div class="limit-item">
                <div class="value">${limits.phosphorus_mg}</div>
                <div class="label">Phosphorus (mg/day)</div>
            </div>
        `;
    }

    function displayRecommendations(recommendations) {
        const listDiv = document.getElementById('recommendationsList');

        if (!recommendations || recommendations.length === 0) {
            listDiv.innerHTML = '<p>No ingredients to analyze.</p>';
            return;
        }

        let html = '';

        for (const rec of recommendations) {
            const statusIcon = rec.status === 'prohibited' ? '⛔' :
                rec.status === 'limited' ? '⚠️' : '✅';
            const statusClass = rec.status;

            let details = '';
            if (rec.nutrients_per_100g) {
                const n = rec.nutrients_per_100g;
                details = `K+: ${n.potassium_mg}mg | Na+: ${n.sodium_mg}mg per 100g`;
            }
            if (rec.warning) {
                details = rec.warning;
            }

            html += `
                <div class="rec-item">
                    <span class="rec-status">${statusIcon}</span>
                    <div class="rec-info">
                        <div class="rec-name">${rec.name}</div>
                        <div class="rec-details">${details}</div>
                    </div>
                    <div class="rec-quantity">
                        <div class="max ${statusClass}">${rec.max_allowed_g.toFixed(0)}</div>
                        <div class="unit">g max</div>
                    </div>
                </div>
            `;
        }

        listDiv.innerHTML = html;
    }

    // ===== Utility Functions =====

    function parseFloatOrNull(value) {
        const parsed = parseFloat(value);
        return isNaN(parsed) ? null : parsed;
    }

    function formatLabName(key) {
        const names = {
            'egfr': 'eGFR',
            'creatinine': 'Creatinine',
            'potassium': 'Potassium',
            'sodium': 'Sodium',
            'glucose': 'Glucose',
            'hba1c': 'HbA1c'
        };
        return names[key] || key;
    }
});
