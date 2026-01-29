// Use relative URL or same origin for API calls
const API_URL = window.location.hostname === 'localhost'
    ? 'http://localhost:8000'
    : window.location.origin;

// DOM Elements
const tabs = document.querySelectorAll('.tab');
const generatePanel = document.getElementById('generate-panel');
const editPanel = document.getElementById('edit-panel');
const generateBtn = document.getElementById('generate-btn');
const editBtn = document.getElementById('edit-btn');
const resultSection = document.getElementById('result-section');
const loading = document.getElementById('loading');
const errorDiv = document.getElementById('error');
const healthStatus = document.getElementById('health-status');

let currentImageData = null;

// Tab switching
tabs.forEach(tab => {
    tab.addEventListener('click', () => {
        tabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        
        const tabName = tab.dataset.tab;
        generatePanel.classList.toggle('active', tabName === 'generate');
        editPanel.classList.toggle('active', tabName === 'edit');
    });
});

// Health check
async function checkHealth() {
    try {
        const res = await fetch(`${API_URL}/health`);
        const data = await res.json();
        healthStatus.textContent = data.cuda_available 
            ? `✅ GPU: ${data.gpu_name}` 
            : '⚠️ CPU Only';
        healthStatus.className = 'health-badge ' + (data.cuda_available ? 'healthy' : 'warning');
    } catch (e) {
        healthStatus.textContent = '❌ API Offline';
        healthStatus.className = 'health-badge error';
    }
}

// Generate image
generateBtn.addEventListener('click', async () => {
    const prompt = document.getElementById('prompt').value;
    if (!prompt.trim()) {
        showError('Please enter a prompt');
        return;
    }

    showLoading(true);
    hideError();

    try {
        const res = await fetch(`${API_URL}/v1/images/create`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt,
                model: document.getElementById('model').value,
                steps: parseInt(document.getElementById('steps').value),
                width: parseInt(document.getElementById('width').value),
                height: parseInt(document.getElementById('height').value)
            })
        });

        const data = await res.json();
        if (data.success) {
            displayResult(data);
        } else {
            showError(data.detail || 'Generation failed');
        }
    } catch (e) {
        showError(`Error: ${e.message}`);
    } finally {
        showLoading(false);
    }
});

// Edit image
editBtn.addEventListener('click', async () => {
    const fileInput = document.getElementById('edit-image');
    if (!fileInput.files[0]) {
        showError('Please upload an image');
        return;
    }

    const operations = getOperations();
    if (operations.length === 0) {
        showError('Please add at least one operation');
        return;
    }

    showLoading(true);
    hideError();

    try {
        const formData = new FormData();
        formData.append('image', fileInput.files[0]);
        formData.append('operations', JSON.stringify(operations));
        formData.append('prompt', document.getElementById('edit-prompt').value);
        formData.append('model', 'qwen-2512');
        formData.append('steps', '8');

        const res = await fetch(`${API_URL}/v1/images/edit`, {
            method: 'POST',
            body: formData
        });

        const data = await res.json();
        if (data.success) {
            displayResult(data);
        } else {
            showError(data.detail || 'Edit failed');
        }
    } catch (e) {
        showError(`Error: ${e.message}`);
    } finally {
        showLoading(false);
    }
});

// Operations management
document.getElementById('add-operation').addEventListener('click', () => {
    const list = document.getElementById('operations-list');
    const div = document.createElement('div');
    div.className = 'operation-item';
    div.innerHTML = `
        <select class="op-type">
            <option value="resize">Resize</option>
            <option value="crop">Crop</option>
            <option value="rotate">Rotate</option>
            <option value="flip">Flip</option>
            <option value="filter">Filter</option>
            <option value="adjust">Adjust</option>
        </select>
        <input type="text" class="op-params" placeholder='{"width": 512, "height": 512}'>
        <button class="btn-remove" onclick="this.parentElement.remove()">×</button>
    `;
    list.appendChild(div);
});

function getOperations() {
    const items = document.querySelectorAll('.operation-item');
    return Array.from(items).map(item => {
        const type = item.querySelector('.op-type').value;
        let params = {};
        try {
            params = JSON.parse(item.querySelector('.op-params').value || '{}');
        } catch (e) {}
        return { type, params };
    });
}

// Image preview
document.getElementById('edit-image').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            document.getElementById('preview-container').innerHTML = 
                `<img src="${e.target.result}" alt="Preview">`;
        };
        reader.readAsDataURL(file);
    }
});

// Display result
function displayResult(data) {
    const image = data.data.images[0];
    currentImageData = image.data;
    
    document.getElementById('result-metadata').innerHTML = `
        <span>Size: ${image.metadata.width}×${image.metadata.height}</span>
        <span>Model: ${image.metadata.model_used}</span>
        <span>Time: ${data.metadata.processing_time_ms}ms</span>
    `;
    
    document.getElementById('result-image-container').innerHTML = 
        `<img src="data:image/png;base64,${image.data}" alt="Result">`;
    
    resultSection.classList.remove('hidden');
}

// Download
document.getElementById('download-btn').addEventListener('click', () => {
    if (!currentImageData) return;
    
    const link = document.createElement('a');
    link.href = `data:image/png;base64,${currentImageData}`;
    link.download = `generated-${Date.now()}.png`;
    link.click();
});

// Helpers
function showLoading(show) {
    loading.classList.toggle('hidden', !show);
}

function showError(msg) {
    errorDiv.textContent = msg;
    errorDiv.classList.remove('hidden');
}

function hideError() {
    errorDiv.classList.add('hidden');
}

// Init
checkHealth();
setInterval(checkHealth, 30000);
