// Global variables
let trainData = null;
let testData = null;
let preprocessedTrainData = null;
let preprocessedTestData = null;
let model = null;
let trainingHistory = null;
let validationData = null;
let validationLabels = null;
let validationPredictions = null;
let testPredictions = null;

// Schema configuration - change these for different datasets
const TARGET_FEATURE = 'Survived';       // Binary classification target
const ID_FEATURE = 'PassengerId';        // Identifier to exclude from features
const NUMERICAL_FEATURES = ['Age', 'Fare', 'SibSp', 'Parch'];
const CATEGORICAL_FEATURES = ['Pclass', 'Sex', 'Embarked'];

// Cached stats (computed once from training data)
let AGE_MEDIAN = 0;
let AGE_STD = 1;
let FARE_MEDIAN = 0;
let FARE_STD = 1;
let EMBARKED_MODE = 'S';

// -------------------- Data Load --------------------

async function loadData() {
    const trainFile = document.getElementById('train-file').files[0];
    const testFile = document.getElementById('test-file').files[0];
    if (!trainFile || !testFile) {
        alert('Please upload both training and test CSV files.');
        return;
    }

    const statusDiv = document.getElementById('data-status');
    statusDiv.textContent = 'Loading data...';

    try {
        const trainText = await readFile(trainFile);
        const testText  = await readFile(testFile);

        // Robust CSV parsing (handles commas in quotes, escaped quotes, CRLF, BOM)
        const { rows: trainRows } = parseCSV(trainText);
        const { rows: testRows }  = parseCSV(testText);

        trainData = trainRows;
        testData  = testRows;

        statusDiv.textContent =
            `Data loaded successfully! Training: ${trainData.length} rows, Test: ${testData.length} rows`;

        document.getElementById('inspect-btn').disabled = false;
    } catch (error) {
        statusDiv.textContent = `Error loading data: ${error.message}`;
        console.error(error);
    }
}

// Read file as text
function readFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = e => resolve(e.target.result);
        reader.onerror = () => reject(new Error('Failed to read file'));
        reader.readAsText(file);
    });
}

/**
 * Robust CSV parser for common cases:
 * - Commas inside quoted fields
 * - Escaped quotes as ""
 * - CRLF/Unix newlines
 * - Strips UTF-8 BOM
 * Returns { headers: string[], rows: object[] }
 */
function parseCSV(text) {
    // Strip BOM if present and normalize line endings
    let s = text.replace(/^\uFEFF/, '').replace(/\r\n/g, '\n').replace(/\r/g, '\n');

    const rows = [];
    let row = [];
    let field = '';
    let inQuotes = false;

    const pushField = () => { row.push(field); field = ''; };
    const pushRow = (headers) => {
        // Pad/truncate to headers length
        const vals = row.length >= headers.length
            ? row.slice(0, headers.length)
            : row.concat(Array(headers.length - row.length).fill(''));
        const obj = {};
        headers.forEach((h, i) => {
            const v = vals[i] === '' ? null : vals[i];
            // Try numeric parse when appropriate
            if (v !== null && !isNaN(v) && v.trim() !== '') obj[h] = parseFloat(v);
            else obj[h] = v;
        });
        rows.push(obj);
        row = [];
    };

    // First, read header line using the same state machine
    let i = 0;
    const headers = [];
    let headerField = '';
    let headerInQuotes = false;
    for (; i < s.length; i++) {
        const c = s[i];
        if (headerInQuotes) {
            if (c === '"') {
                if (s[i + 1] === '"') { headerField += '"'; i++; } // escaped quote
                else { headerInQuotes = false; }
            } else headerField += c;
        } else {
            if (c === '"') headerInQuotes = true;
            else if (c === ',') { headers.push(headerField.trim()); headerField = ''; }
            else if (c === '\n') { headers.push(headerField.trim()); i++; break; }
            else headerField += c;
        }
    }
    if (headerField.length && headers.length === 0) {
        // No newline found; single-line CSV
        headers.push(...headerField.split(',').map(h => h.trim()));
        return { headers, rows: [] };
    }

    // Now parse the remaining characters as data rows
    for (; i < s.length; i++) {
        const c = s[i];
        if (inQuotes) {
            if (c === '"') {
                if (s[i + 1] === '"') { field += '"'; i++; } // escaped quote
                else { inQuotes = false; }
            } else field += c;
        } else {
            if (c === '"') inQuotes = true;
            else if (c === ',') pushField();
            else if (c === '\n') { pushField(); pushRow(headers); }
            else field += c;
        }
    }
    // Flush last field/row if file doesn't end with newline
    if (field.length > 0 || inQuotes || row.length > 0) { pushField(); pushRow(headers); }

    return { headers, rows };
}

// -------------------- Inspect --------------------

function inspectData() {
    if (!trainData || trainData.length === 0) {
        alert('Please load data first.');
        return;
    }

    const previewDiv = document.getElementById('data-preview');
    previewDiv.innerHTML = '<h3>Data Preview (First 10 Rows)</h3>';
    previewDiv.appendChild(createPreviewTable(trainData.slice(0, 10)));

    const statsDiv = document.getElementById('data-stats');
    statsDiv.innerHTML = '<h3>Data Statistics</h3>';

    const shapeInfo = `Dataset shape: ${trainData.length} rows × ${Object.keys(trainData[0]).length} columns`;
    const survivalCount = trainData.filter(r => r[TARGET_FEATURE] === 1).length;
    const survivalRate = ((survivalCount / trainData.length) * 100).toFixed(2);
    const targetInfo = `Survival rate: ${survivalCount}/${trainData.length} (${survivalRate}%)`;

    // Missing %
    let missingInfo = '<h4>Missing Values Percentage:</h4><ul>';
    Object.keys(trainData[0]).forEach(feature => {
        const missingCount = trainData.filter(row => row[feature] === null || row[feature] === undefined).length;
        missingInfo += `<li>${feature}: ${(missingCount / trainData.length * 100).toFixed(2)}%</li>`;
    });
    missingInfo += '</ul>';

    statsDiv.innerHTML += `<p>${shapeInfo}</p><p>${targetInfo}</p>${missingInfo}`;

    createVisualizations();
    document.getElementById('preprocess-btn').disabled = false;
}

// Preview table
function createPreviewTable(data) {
    const table = document.createElement('table');
    const headerRow = document.createElement('tr');
    Object.keys(data[0]).forEach(key => {
        const th = document.createElement('th');
        th.textContent = key;
        headerRow.appendChild(th);
    });
    table.appendChild(headerRow);

    data.forEach(row => {
        const tr = document.createElement('tr');
        Object.keys(data[0]).forEach(key => {
            const td = document.createElement('td');
            const v = row[key];
            td.textContent = v === null || v === undefined ? 'NULL' : v;
            tr.appendChild(td);
        });
        table.appendChild(tr);
    });

    return table;
}

// Simple tfjs-vis charts
function createVisualizations() {
    // Survival by Sex
    const survivalBySex = {};
    trainData.forEach(row => {
        if (row.Sex != null && row.Survived != null) {
            if (!survivalBySex[row.Sex]) survivalBySex[row.Sex] = { survived: 0, total: 0 };
            survivalBySex[row.Sex].total++;
            if (row.Survived === 1) survivalBySex[row.Sex].survived++;
        }
    });
    const sexValues = Object.entries(survivalBySex).map(([sex, s]) => ({
        x: sex, y: (s.survived / s.total) * 100
    }));
    const sexContainer = document.getElementById('chart-sex');
    sexContainer.innerHTML = ''; // clear if re-render
    tfvis.render.barchart(sexContainer, sexValues, {
        xLabel: 'Sex', yLabel: 'Survival Rate (%)', width: 520, height: 320
    });

    // Survival by Pclass
    const byPclass = {};
    trainData.forEach(row => {
        if (row.Pclass != null && row.Survived != null) {
            if (!byPclass[row.Pclass]) byPclass[row.Pclass] = { survived: 0, total: 0 };
            byPclass[row.Pclass].total++;
            if (row.Survived === 1) byPclass[row.Pclass].survived++;
        }
    });
    const pclassValues = Object.entries(byPclass).map(([pc, s]) => ({
        x: `Class ${pc}`, y: (s.survived / s.total) * 100
    }));
    const pclassContainer = document.getElementById('chart-pclass');
    pclassContainer.innerHTML = '';
    tfvis.render.barchart(pclassContainer, pclassValues, {
        xLabel: 'Passenger Class', yLabel: 'Survival Rate (%)', width: 520, height: 320
    });
}


// -------------------- Preprocessing --------------------

function preprocessData() {
    if (!trainData || !testData) {
        alert('Please load data first.');
        return;
    }

    const outputDiv = document.getElementById('preprocessing-output');
    outputDiv.textContent = 'Preprocessing data...';

    try {
        // Compute stats from training data
        const ageVals  = trainData.map(r => r.Age).filter(v => v != null);
        const fareVals = trainData.map(r => r.Fare).filter(v => v != null);
        AGE_MEDIAN  = calculateMedian(ageVals);
        FARE_MEDIAN = calculateMedian(fareVals);
        AGE_STD     = calculateStdDev(ageVals) || 1;
        FARE_STD    = calculateStdDev(fareVals) || 1;
        EMBARKED_MODE = calculateMode(trainData.map(r => r.Embarked).filter(v => v != null)) ?? 'S';

        // Build tensors
        const X = [];
        const y = [];
        for (const row of trainData) {
            X.push(extractFeatures(row));
            y.push(row[TARGET_FEATURE]);
        }

        const Xtest = [];
        const ids   = [];
        for (const row of testData) {
            Xtest.push(extractFeatures(row));
            ids.push(row[ID_FEATURE]);
        }

        preprocessedTrainData = {
            features: tf.tensor2d(X),
            labels: tf.tensor1d(y, 'float32')
        };
        preprocessedTestData = {
            features: Xtest, // keep raw array; will tensorize at predict() time
            passengerIds: ids
        };

        outputDiv.innerHTML = `
            <p>Preprocessing completed!</p>
            <p>Training features shape: ${preprocessedTrainData.features.shape}</p>
            <p>Training labels shape: ${preprocessedTrainData.labels.shape}</p>
            <p>Test features shape: [${Xtest.length}, ${Xtest[0] ? Xtest[0].length : 0}]</p>
        `;

        document.getElementById('create-model-btn').disabled = false;
    } catch (error) {
        outputDiv.textContent = `Error during preprocessing: ${error.message}`;
        console.error(error);
    }
}

// Build one feature vector for a row
function extractFeatures(row) {
    const age = (row.Age != null ? row.Age : AGE_MEDIAN);
    const fare = (row.Fare != null ? row.Fare : FARE_MEDIAN);
    const embarked = (row.Embarked != null ? row.Embarked : EMBARKED_MODE);

    // Standardize with training stats
    const standardizedAge  = (age - AGE_MEDIAN) / (AGE_STD || 1);
    const standardizedFare = (fare - FARE_MEDIAN) / (FARE_STD || 1);

    const pclassOneHot   = oneHotEncode(row.Pclass, [1, 2, 3]);
    const sexOneHot      = oneHotEncode(row.Sex, ['male', 'female']);
    const embarkedOneHot = oneHotEncode(embarked, ['C', 'Q', 'S']);

    let features = [
        standardizedAge,
        standardizedFare,
        row.SibSp || 0,
        row.Parch || 0
    ];
    features = features.concat(pclassOneHot, sexOneHot, embarkedOneHot);

    if (document.getElementById('add-family-features').checked) {
        const familySize = (row.SibSp || 0) + (row.Parch || 0) + 1;
        const isAlone = familySize === 1 ? 1 : 0;
        features.push(familySize, isAlone);
    }
    // Replace any NaN/undefined with 0
    return features.map(v => (Number.isFinite(v) ? v : 0));
}

// Helpers
function calculateMedian(values) {
    if (!values.length) return 0;
    const arr = [...values].sort((a, b) => a - b);
    const m = Math.floor(arr.length / 2);
    return arr.length % 2 ? arr[m] : (arr[m - 1] + arr[m]) / 2;
}
function calculateMode(values) {
    if (!values.length) return null;
    const freq = {};
    let best = null, bestCnt = 0;
    for (const v of values) {
        freq[v] = (freq[v] || 0) + 1;
        if (freq[v] > bestCnt) { bestCnt = freq[v]; best = v; }
    }
    return best;
}
function calculateStdDev(values) {
    if (!values.length) return 0;
    const mean = values.reduce((s, v) => s + v, 0) / values.length;
    const varSum = values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length;
    return Math.sqrt(varSum);
}
function oneHotEncode(value, categories) {
    const enc = new Array(categories.length).fill(0);
    const idx = categories.indexOf(value);
    if (idx !== -1) enc[idx] = 1;
    return enc;
}

// -------------------- Model --------------------

function createModel() {
    if (!preprocessedTrainData) {
        alert('Please preprocess data first.');
        return;
    }
    const inputShape = preprocessedTrainData.features.shape[1];

    model = tf.sequential();
    model.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [inputShape] }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    model.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    const summaryDiv = document.getElementById('model-summary');
    summaryDiv.innerHTML = '<h3>Model Summary</h3>';
    let summaryText = '<ul>';
    model.layers.forEach((layer, i) => {
        summaryText += `<li>Layer ${i + 1}: ${layer.getClassName()} - Output Shape: ${JSON.stringify(layer.outputShape)}</li>`;
    });
    summaryText += '</ul>';
    summaryText += `<p>Total parameters: ${model.countParams()}</p>`;
    summaryDiv.innerHTML += summaryText;

    document.getElementById('train-btn').disabled = false;
}

// -------------------- Training --------------------

async function trainModel() {
    if (!model || !preprocessedTrainData) {
        alert('Please create model first.');
        return;
    }
    const statusDiv = document.getElementById('training-status');
    statusDiv.textContent = 'Training model...';

    try {
        const n = preprocessedTrainData.features.shape[0];
        const splitIndex = Math.floor(n * 0.8);

        const trainFeatures = preprocessedTrainData.features.slice(0, splitIndex);
        const trainLabels   = preprocessedTrainData.labels.slice(0, splitIndex);
        const valFeatures   = preprocessedTrainData.features.slice(splitIndex);
        const valLabels     = preprocessedTrainData.labels.slice(splitIndex);

        validationData = valFeatures;
        validationLabels = valLabels;

        const visCallbacks = tfvis.show.fitCallbacks(
            { name: 'Training Performance', tab: 'Training' },
            ['loss', 'acc', 'val_loss', 'val_acc'], // 'acc' works for accuracy
            { callbacks: ['onEpochEnd'] }
        );

        // Combine tfjs-vis callbacks with a custom progress logger
        const progressLogger = {
            onEpochEnd: async (epoch, logs) => {
                const { loss, acc, val_loss, val_acc } = logs;
                statusDiv.textContent =
                    `Epoch ${epoch + 1}/50 - loss: ${num(loss)}  acc: ${num(acc)}  val_loss: ${num(val_loss)}  val_acc: ${num(val_acc)}`;
                // Also forward to vis
                if (visCallbacks.onEpochEnd) await visCallbacks.onEpochEnd(epoch, logs);
            },
            onTrainEnd: async () => {
                statusDiv.innerHTML += '<p>Training completed!</p>';
            },
            onTrainBegin: async (logs) => {
                if (visCallbacks.onTrainBegin) await visCallbacks.onTrainBegin(logs);
            }
        };

        trainingHistory = await model.fit(trainFeatures, trainLabels, {
            epochs: 50,
            batchSize: 32,
            validationData: [valFeatures, valLabels],
            callbacks: [progressLogger]
        });

        // Predictions for evaluation (keep tensor for interactions)
        validationPredictions = model.predict(validationData);

        // Enable evaluation controls and compute initial metrics
        const slider = document.getElementById('threshold-slider');
        slider.disabled = false;
        slider.removeEventListener('input', updateMetrics);
        slider.addEventListener('input', updateMetrics);
        await updateMetrics();

        document.getElementById('predict-btn').disabled = false;
    } catch (error) {
        statusDiv.textContent = `Error during training: ${error.message}`;
        console.error(error);
    }
}

function num(v) {
    return (typeof v === 'number' && isFinite(v)) ? v.toFixed(4) : '—';
}

// -------------------- Metrics / Evaluation --------------------

async function updateMetrics() {
    if (!validationPredictions || !validationLabels) return;

    const threshold = parseFloat(document.getElementById('threshold-slider').value);
    document.getElementById('threshold-value').textContent = threshold.toFixed(2);

    // Flatten predictions [[p], [p], ...] -> [p, p, ...]
    const predVals = validationPredictions.arraySync()
        .map(v => Array.isArray(v) ? v[0] : v);
    const trueVals = validationLabels.arraySync();

    let tp = 0, tn = 0, fp = 0, fn = 0;
    for (let i = 0; i < predVals.length; i++) {
        const p = predVals[i] >= threshold ? 1 : 0;
        const a = trueVals[i] >= 0.5 ? 1 : 0;
        if (p === 1 && a === 1) tp++;
        else if (p === 0 && a === 0) tn++;
        else if (p === 1 && a === 0) fp++;
        else if (p === 0 && a === 1) fn++;
    }

    const cmDiv = document.getElementById('confusion-matrix');
    cmDiv.innerHTML = `
        <table>
            <tr><th></th><th>Predicted Positive</th><th>Predicted Negative</th></tr>
            <tr><th>Actual Positive</th><td>${tp}</td><td>${fn}</td></tr>
            <tr><th>Actual Negative</th><td>${fp}</td><td>${tn}</td></tr>
        </table>
    `;

    const precision = tp / (tp + fp || 1);
    const recall    = tp / (tp + fn || 1);
    const f1        = 2 * (precision * recall) / ((precision + recall) || 1);
    const accuracy  = (tp + tn) / (tp + tn + fp + fn || 1);

    const metricsDiv = document.getElementById('performance-metrics');
    metricsDiv.innerHTML = `
        <p>Accuracy: ${(accuracy * 100).toFixed(2)}%</p>
        <p>Precision: ${precision.toFixed(4)}</p>
        <p>Recall: ${recall.toFixed(4)}</p>
        <p>F1 Score: ${f1.toFixed(4)}</p>
    `;

    await plotROC(trueVals, predVals);
}

async function plotROC(trueLabels, predictions) {
    const thresholds = Array.from({ length: 101 }, (_, i) => i / 100);
    const points = [];

    for (const t of thresholds) {
        let tp = 0, fn = 0, fp = 0, tn = 0;
        for (let i = 0; i < predictions.length; i++) {
            const pred = predictions[i] >= t ? 1 : 0;
            const actual = trueLabels[i] >= 0.5 ? 1 : 0;
            if (actual === 1) { if (pred === 1) tp++; else fn++; }
            else { if (pred === 1) fp++; else tn++; }
        }
        const tpr = tp / (tp + fn || 1);
        const fpr = fp / (fp + tn || 1);
        points.push({ x: fpr, y: tpr });
    }

    // AUC via trapezoidal rule (sorted by FPR ascending)
    const sorted = points.slice().sort((a, b) => a.x - b.x);
    let auc = 0;
    for (let i = 1; i < sorted.length; i++) {
        const dx = sorted[i].x - sorted[i - 1].x;
        const avgY = (sorted[i].y + sorted[i - 1].y) / 2;
        auc += dx * avgY;
    }

    const rocEl = document.getElementById('roc-chart');
if (rocEl) {
    rocEl.innerHTML = ''; // clear previous render
    tfvis.render.linechart(
        rocEl,
        { values: [sorted], series: ['ROC'] },
        { xLabel: 'False Positive Rate', yLabel: 'True Positive Rate', width: 520, height: 360 }
    );
}


    const metricsDiv = document.getElementById('performance-metrics');
    // Avoid duplicate AUC lines by replacing/adding it once
    const aucP = document.createElement('p');
    aucP.textContent = `AUC: ${auc.toFixed(4)}`;
    metricsDiv.appendChild(aucP);
}

// -------------------- Inference & Export --------------------

async function predict() {
    if (!model || !preprocessedTestData) {
        alert('Please train model first.');
        return;
    }

    const outputDiv = document.getElementById('prediction-output');
    outputDiv.textContent = 'Making predictions...';

    try {
        const testFeatures = tf.tensor2d(preprocessedTestData.features);
        testPredictions = model.predict(testFeatures);
        const predValues = testPredictions.arraySync().map(v => Array.isArray(v) ? v[0] : v);

        const results = preprocessedTestData.passengerIds.map((id, i) => ({
            PassengerId: id,
            Survived: predValues[i] >= 0.5 ? 1 : 0,
            Probability: predValues[i]
        }));

        outputDiv.innerHTML = '<h3>Prediction Results (First 10 Rows)</h3>';
        outputDiv.appendChild(createPredictionTable(results.slice(0, 10)));
        outputDiv.innerHTML += `<p>Predictions completed! Total: ${results.length} rows</p>`;

        document.getElementById('export-btn').disabled = false;
    } catch (error) {
        outputDiv.textContent = `Error during prediction: ${error.message}`;
        console.error(error);
    }
}

function createPredictionTable(data) {
    const table = document.createElement('table');
    const headerRow = document.createElement('tr');
    ['PassengerId', 'Survived', 'Probability'].forEach(h => {
        const th = document.createElement('th'); th.textContent = h; headerRow.appendChild(th);
    });
    table.appendChild(headerRow);

    data.forEach(row => {
        const tr = document.createElement('tr');
        ['PassengerId', 'Survived', 'Probability'].forEach(k => {
            const td = document.createElement('td');
            td.textContent = k === 'Probability' ? Number(row[k]).toFixed(4) : row[k];
            tr.appendChild(td);
        });
        table.appendChild(tr);
    });
    return table;
}

async function exportResults() {
    if (!testPredictions || !preprocessedTestData) {
        alert('Please make predictions first.');
        return;
    }
    const statusDiv = document.getElementById('export-status');
    statusDiv.textContent = 'Exporting results...';

    try {
        const predValues = testPredictions.arraySync().map(v => Array.isArray(v) ? v[0] : v);

        let submissionCSV = 'PassengerId,Survived\n';
        preprocessedTestData.passengerIds.forEach((id, i) => {
            submissionCSV += `${id},${predValues[i] >= 0.5 ? 1 : 0}\n`;
        });

        let probabilitiesCSV = 'PassengerId,Probability\n';
        preprocessedTestData.passengerIds.forEach((id, i) => {
            probabilitiesCSV += `${id},${Number(predValues[i]).toFixed(6)}\n`;
        });

        const submissionLink = document.createElement('a');
        submissionLink.href = URL.createObjectURL(new Blob([submissionCSV], { type: 'text/csv' }));
        submissionLink.download = 'submission.csv';

        const probabilitiesLink = document.createElement('a');
        probabilitiesLink.href = URL.createObjectURL(new Blob([probabilitiesCSV], { type: 'text/csv' }));
        probabilitiesLink.download = 'probabilities.csv';

        submissionLink.click();
        probabilitiesLink.click();

        await model.save('downloads://titanic-tfjs-model');

        statusDiv.innerHTML = `
            <p>Export completed!</p>
            <p>Downloaded: <code>submission.csv</code></p>
            <p>Downloaded: <code>probabilities.csv</code></p>
            <p>Model saved to browser downloads</p>
        `;
    } catch (error) {
        statusDiv.textContent = `Error during export: ${error.message}`;
        console.error(error);
    }
}
