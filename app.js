/* ==========================================================================
   Employee Earnings Predictor - Machine Learning & UI Engine
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    // --- Currency Conversion Configuration ---
    const CURRENCY_CONFIG = {
        USD: { symbol: '$', rate: 1.0, precision: 0 },
        EUR: { symbol: '€', rate: 0.92, precision: 0 },
        GBP: { symbol: '£', rate: 0.79, precision: 0 },
        INR: { symbol: '₹', rate: 83.5, precision: 0 }
    };
    let currentCurrency = 'USD';

    // --- Chart Instances ---
    let waterfallChart = null;
    let trajectoryChart = null;
    let radarChart = null;

    // --- DOM Elements ---
    const ageRange = document.getElementById('age-range');
    const ageVal = document.getElementById('age-val');
    const expRange = document.getElementById('exp-range');
    const expVal = document.getElementById('exp-val');
    const hoursRange = document.getElementById('hours-range');
    const hoursVal = document.getElementById('hours-val');
    const educationSelect = document.getElementById('education');
    const occupationSelect = document.getElementById('occupation');
    const workModeSelect = document.getElementById('work-mode');
    const companySizeSelect = document.getElementById('company-size');
    const locationTierSelect = document.getElementById('location-tier');
    const genderSelect = document.getElementById('gender');
    const currencySelect = document.getElementById('currency-select');
    const resetBtn = document.getElementById('reset-btn');

    // Outputs
    const predictedSalaryEl = document.getElementById('predicted-salary');
    const minSalaryEl = document.getElementById('min-salary');
    const maxSalaryEl = document.getElementById('max-salary');
    const monthlySalaryEl = document.getElementById('monthly-salary');
    const hourlySalaryEl = document.getElementById('hourly-salary');
    const percentileRankEl = document.getElementById('percentile-rank');
    const meterFillEl = document.getElementById('meter-fill');
    const topDriversListEl = document.getElementById('top-drivers-list');
    const optimizerTipsEl = document.getElementById('optimizer-tips');

    // --- Event Listeners ---
    [ageRange, expRange, hoursRange].forEach(input => {
        input.addEventListener('input', updateInputDisplay);
    });

    [ageRange, expRange, hoursRange, educationSelect, occupationSelect, workModeSelect, companySizeSelect, locationTierSelect, genderSelect].forEach(element => {
        element.addEventListener('change', runPrediction);
        element.addEventListener('input', runPrediction);
    });

    currencySelect.addEventListener('change', (e) => {
        currentCurrency = e.target.value;
        document.querySelectorAll('.currency-symbol').forEach(el => {
            el.textContent = CURRENCY_CONFIG[currentCurrency].symbol;
        });
        runPrediction();
        updateBatchDisplay();
    });

    resetBtn.addEventListener('click', resetForm);

    // Initial Trigger
    updateInputDisplay();
    runPrediction();
    try {
        initCharts();
        runPrediction(); // Update charts after init
    } catch (err) {
        console.warn('Charts init warning:', err);
    }
    setupBatchProcessor();

    // --- Input Display Updater ---
    function updateInputDisplay() {
        ageVal.textContent = ageRange.value;
        expVal.textContent = expRange.value;
        hoursVal.textContent = hoursRange.value;
        
        // Enforce experience <= age - 16
        const maxExp = Math.max(0, parseInt(ageRange.value) - 16);
        if (parseInt(expRange.value) > maxExp) {
            expRange.value = maxExp;
            expVal.textContent = maxExp;
        }
        expRange.max = maxExp;
    }

    // --- Form Reset ---
    function resetForm() {
        ageRange.value = 34;
        expRange.value = 8;
        hoursRange.value = 40;
        educationSelect.value = 'bachelors';
        occupationSelect.value = 'swe';
        workModeSelect.value = 'hybrid';
        companySizeSelect.value = 'mid';
        locationTierSelect.value = 'tier1';
        genderSelect.value = 'female';
        updateInputDisplay();
        runPrediction();
    }

    // ==========================================================================
    // Core Machine Learning Inference Engine (XGBoost / Tree Simulator)
    // ==========================================================================
    function calculateEarnings(inputs) {
        const BASE_MARKET_SALARY = 24000;
        
        // Feature Weights & Multipliers (Scaled down for realistic salary calibration)
        const eduWeights = {
            highschool: 0,
            associate: 4500,
            bachelors: 14000,
            masters: 26000,
            doctorate: 40000,
            professional: 48000
        };

        const occData = {
            swe: { base: 22000, expSlope: 1800 },
            ds_ai: { base: 25000, expSlope: 2000 },
            mgmt: { base: 28000, expSlope: 1900 },
            finance: { base: 24000, expSlope: 2100 },
            health: { base: 20000, expSlope: 1500 },
            design: { base: 14000, expSlope: 1300 },
            sales: { base: 15000, expSlope: 1500 },
            edu: { base: 8000, expSlope: 1000 },
            trades: { base: 10000, expSlope: 1100 }
        };

        const locMultipliers = {
            tier1: 1.18,
            tier2: 1.02,
            tier3: 0.88,
            global: 1.10
        };

        const sizeMultipliers = {
            startup: 0.95,
            mid: 1.00,
            enterprise: 1.12
        };

        const modeBonus = {
            remote: 1500,
            hybrid: 800,
            onsite: 0
        };

        // Component Calculations
        const occConfig = occData[inputs.occupation] || occData.swe;
        const occEffect = occConfig.base;
        
        // Experience non-linear logarithmic curve
        const expCurveMultiplier = Math.pow(0.975, Math.max(0, inputs.experience - 10));
        const expEffect = inputs.experience * occConfig.expSlope * expCurveMultiplier;

        // Education impact
        const eduEffect = eduWeights[inputs.education] || 0;

        // Hours worked impact (linear over 40hrs, penalty under 35hrs)
        let hoursEffect = 0;
        if (inputs.hours > 40) {
            hoursEffect = (inputs.hours - 40) * 1450;
        } else if (inputs.hours < 35) {
            hoursEffect = (inputs.hours - 40) * 950;
        }

        // Age parabolic factor (peaks at ~50 yrs)
        const ageEffect = Math.sin((inputs.age - 18) / 55 * Math.PI) * 8500;

        // Sum Raw Subtotal
        const rawSubtotal = BASE_MARKET_SALARY + occEffect + expEffect + eduEffect + hoursEffect + ageEffect + modeBonus[inputs.workMode];

        // Apply Location & Company Scale Multipliers
        const locMultiplier = locMultipliers[inputs.locationTier] || 1.0;
        const sizeMultiplier = sizeMultipliers[inputs.companySize] || 1.0;

        const locEffect = rawSubtotal * (locMultiplier - 1);
        const sizeEffect = rawSubtotal * (sizeMultiplier - 1);

        const totalSalaryUSD = (rawSubtotal + locEffect + sizeEffect);

        // Feature Attribution Breakdown for SHAP Waterfall
        const attributions = [
            { name: 'Base Market Salary', val: BASE_MARKET_SALARY },
            { name: 'Occupation Level', val: occEffect },
            { name: 'Education Level', val: eduEffect },
            { name: 'Work Experience', val: expEffect },
            { name: 'Weekly Hours', val: hoursEffect },
            { name: 'Age & Lifecycle', val: ageEffect },
            { name: 'Regional Market', val: locEffect },
            { name: 'Company Scale', val: sizeEffect },
            { name: 'Work Arrangement', val: modeBonus[inputs.workMode] }
        ];

        return {
            totalUSD: totalSalaryUSD,
            attributions: attributions,
            occConfig: occConfig,
            rawInputs: inputs
        };
    }

    // ==========================================================================
    // Run Main Prediction Pipeline
    // ==========================================================================
    function runPrediction() {
        const inputs = {
            age: parseInt(ageRange.value),
            experience: parseInt(expRange.value),
            hours: parseInt(hoursRange.value),
            education: educationSelect.value,
            occupation: occupationSelect.value,
            workMode: workModeSelect.value,
            companySize: companySizeSelect.value,
            locationTier: locationTierSelect.value,
            gender: genderSelect.value
        };

        const result = calculateEarnings(inputs);
        const curr = CURRENCY_CONFIG[currentCurrency];
        
        // Converted Values
        const convertedSalary = Math.round(result.totalUSD * curr.rate);
        const minSalary = Math.round(convertedSalary * 0.82);
        const maxSalary = Math.round(convertedSalary * 1.22);
        const monthly = Math.round(convertedSalary / 12);
        const hourly = (convertedSalary / (inputs.hours * 52)).toFixed(2);

        // Animate Salary Counter
        animateCounter(predictedSalaryEl, convertedSalary);
        minSalaryEl.textContent = `${curr.symbol}${minSalary.toLocaleString()}`;
        maxSalaryEl.textContent = `${curr.symbol}${maxSalary.toLocaleString()}`;
        monthlySalaryEl.textContent = `${curr.symbol}${monthly.toLocaleString()}`;
        hourlySalaryEl.textContent = `${curr.symbol}${hourly} / hr`;

        // Percentile & Meter Fill
        const percentile = Math.min(99, Math.max(5, Math.round((result.totalUSD - 20000) / 180000 * 100)));
        percentileRankEl.textContent = `Top ${100 - percentile}%`;
        meterFillEl.style.width = `${percentile}%`;

        // Populate Top Drivers List
        populateDrivers(result.attributions, curr);

        // Update Charts & Optimizer Tips
        try {
            updateWaterfallChart(result.attributions, curr);
            updateTrajectoryChart(inputs, curr);
            updateRadarChart(inputs);
        } catch (chartErr) {
            console.warn('Chart update notice:', chartErr);
        }
        updateOptimizerTips(inputs, result, curr);
    }

    // --- Animate Number Counter ---
    function animateCounter(element, target) {
        const duration = 400;
        const start = parseInt(element.textContent.replace(/,/g, '')) || 0;
        const range = target - start;
        let startTime = null;

        function step(timestamp) {
            if (!startTime) startTime = timestamp;
            const progress = Math.min((timestamp - startTime) / duration, 1);
            const current = Math.floor(start + range * progress);
            element.textContent = current.toLocaleString();
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        }
        window.requestAnimationFrame(step);
    }

    // --- Populate Top Drivers ---
    function populateDrivers(attributions, curr) {
        topDriversListEl.innerHTML = '';
        
        // Filter out base salary & sort by magnitude
        const sorted = [...attributions]
            .filter(a => a.name !== 'Base Market Salary')
            .sort((a, b) => Math.abs(b.val) - Math.abs(a.val))
            .slice(0, 4);

        sorted.forEach(item => {
            const li = document.createElement('li');
            li.className = 'driver-item';
            
            const converted = Math.round(item.val * curr.rate);
            const isPos = converted >= 0;
            const formattedVal = `${isPos ? '+' : ''}${curr.symbol}${converted.toLocaleString()}`;

            li.innerHTML = `
                <span class="driver-name">${item.name}</span>
                <span class="driver-val ${isPos ? 'pos' : 'neg'}">${formattedVal}</span>
            `;
            topDriversListEl.appendChild(li);
        });
    }

    // ==========================================================================
    // Visualizations Suite (Chart.js Integrations)
    // ==========================================================================
    function initCharts() {
        Chart.defaults.color = '#94a3b8';
        Chart.defaults.font.family = "'Plus Jakarta Sans', sans-serif";

        // 1. SHAP Waterfall / Bar Chart
        const ctxW = document.getElementById('waterfallChart').getContext('2d');
        waterfallChart = new Chart(ctxW, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Feature Contribution',
                    data: [],
                    backgroundColor: [],
                    borderRadius: 6
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(ctx) {
                                const curr = CURRENCY_CONFIG[currentCurrency];
                                return `Impact: ${ctx.raw >= 0 ? '+' : ''}${curr.symbol}${Math.round(ctx.raw).toLocaleString()}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' }
                    },
                    y: {
                        grid: { display: false }
                    }
                }
            }
        });

        // 2. Trajectory Chart
        const ctxT = document.getElementById('trajectoryChart').getContext('2d');
        trajectoryChart = new Chart(ctxT, {
            type: 'line',
            data: {
                labels: ['0 yr', '5 yrs', '10 yrs', '15 yrs', '20 yrs', '25 yrs', '30 yrs'],
                datasets: [{
                    label: 'Predicted Growth Trajectory',
                    data: [],
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 5,
                    pointBackgroundColor: '#8b5cf6'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: { grid: { color: 'rgba(255, 255, 255, 0.05)' } },
                    y: { grid: { color: 'rgba(255, 255, 255, 0.05)' } }
                }
            }
        });

        // 3. Radar Benchmark Chart
        const ctxR = document.getElementById('radarChart').getContext('2d');
        radarChart = new Chart(ctxR, {
            type: 'radar',
            data: {
                labels: ['Base Competency', 'Edu Impact', 'Role Multiplier', 'Market Loc', 'Experience Acceleration'],
                datasets: [{
                    label: 'Selected Profile',
                    data: [70, 80, 75, 90, 60],
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.2)',
                    pointBackgroundColor: '#10b981'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    r: {
                        angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        pointLabels: { color: '#94a3b8', font: { size: 10 } },
                        ticks: { display: false }
                    }
                }
            }
        });
    }

    function updateWaterfallChart(attributions, curr) {
        if (!waterfallChart) return;

        const labels = attributions.map(a => a.name);
        const data = attributions.map(a => Math.round(a.val * curr.rate));
        const colors = data.map(v => v >= 0 ? 'rgba(16, 185, 129, 0.8)' : 'rgba(236, 72, 153, 0.8)');

        waterfallChart.data.labels = labels;
        waterfallChart.data.datasets[0].data = data;
        waterfallChart.data.datasets[0].backgroundColor = colors;
        waterfallChart.update();
    }

    function updateTrajectoryChart(inputs, curr) {
        if (!trajectoryChart) return;

        const expMilestones = [0, 5, 10, 15, 20, 25, 30];
        const trajectoryData = expMilestones.map(exp => {
            const testInputs = { ...inputs, experience: exp };
            const res = calculateEarnings(testInputs);
            return Math.round(res.totalUSD * curr.rate);
        });

        trajectoryChart.data.datasets[0].data = trajectoryData;
        trajectoryChart.update();
    }

    function updateRadarChart(inputs) {
        if (!radarChart) return;

        const eduScore = { highschool: 30, associate: 45, bachelors: 70, masters: 88, doctorate: 98, professional: 100 }[inputs.education] || 50;
        const expScore = Math.min(100, inputs.experience * 3.5 + 20);
        const locScore = { tier1: 95, tier2: 70, tier3: 45, global: 85 }[inputs.locationTier] || 60;
        const hoursScore = Math.min(100, inputs.hours * 1.5 + 10);
        const sizeScore = { startup: 60, mid: 75, enterprise: 95 }[inputs.companySize] || 70;

        radarChart.data.datasets[0].data = [expScore, eduScore, sizeScore, locScore, hoursScore];
        radarChart.update();
    }

    // ==========================================================================
    // Optimizer & What-If Recommendations
    // ==========================================================================
    function updateOptimizerTips(inputs, result, curr) {
        optimizerTipsEl.innerHTML = '';

        const tips = [];

        // Tip 1: Education Upgrade
        if (inputs.education !== 'masters' && inputs.education !== 'doctorate' && inputs.education !== 'professional') {
            const masterInputs = { ...inputs, education: 'masters' };
            const mRes = calculateEarnings(masterInputs);
            const diff = Math.round((mRes.totalUSD - result.totalUSD) * curr.rate);
            tips.push({
                icon: 'fa-graduation-cap',
                title: 'Higher Degree Acceleration',
                desc: `Obtaining a Master's degree increases predicted annual salary by +${curr.symbol}${diff.toLocaleString()} / year.`,
                highlight: true
            });
        }

        // Tip 2: Tier 1 Location / Remote Benchmark
        if (inputs.locationTier !== 'tier1') {
            const t1Inputs = { ...inputs, locationTier: 'tier1' };
            const tRes = calculateEarnings(t1Inputs);
            const diff = Math.round((tRes.totalUSD - result.totalUSD) * curr.rate);
            tips.push({
                icon: 'fa-location-dot',
                title: 'Market Relocation / Remote Tier 1',
                desc: `Targeting Tier 1 market benchmarks adds approximately +${curr.symbol}${diff.toLocaleString()} to annual compensation.`,
                highlight: false
            });
        }

        // Tip 3: Enterprise Scale Shift
        if (inputs.companySize !== 'enterprise') {
            const entInputs = { ...inputs, companySize: 'enterprise' };
            const eRes = calculateEarnings(entInputs);
            const diff = Math.round((eRes.totalUSD - result.totalUSD) * curr.rate);
            tips.push({
                icon: 'fa-building',
                title: 'Enterprise Scale Advantage',
                desc: `Moving to an Enterprise-scale organization (500+ employees) yields +${curr.symbol}${diff.toLocaleString()} premium.`,
                highlight: false
            });
        }

        // Tip 4: Hours optimization
        if (inputs.hours < 40) {
            const hInputs = { ...inputs, hours: 40 };
            const hRes = calculateEarnings(hInputs);
            const diff = Math.round((hRes.totalUSD - result.totalUSD) * curr.rate);
            tips.push({
                icon: 'fa-clock',
                title: 'Full-Time Hours (40 hrs/wk)',
                desc: `Increasing work hours to standard 40 hrs/week recovers +${curr.symbol}${diff.toLocaleString()} annually.`,
                highlight: true
            });
        }

        tips.forEach(t => {
            const div = document.createElement('div');
            div.className = `tip-box ${t.highlight ? 'highlight' : ''}`;
            div.innerHTML = `
                <i class="fa-solid ${t.icon} tip-icon"></i>
                <div class="tip-content">
                    <span class="tip-title">${t.title}</span>
                    <span class="tip-desc">${t.desc}</span>
                </div>
            `;
            optimizerTipsEl.appendChild(div);
        });
    }

    // ==========================================================================
    // Batch CSV Prediction Processor
    // ==========================================================================
    let batchDataStore = [];

    function setupBatchProcessor() {
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('csv-file-input');
        const sampleBtn = document.getElementById('load-sample-btn');
        const downloadBtn = document.getElementById('download-csv-btn');

        // Drag & drop handlers
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropZone.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropZone.classList.remove('dragover');
            });
        });

        dropZone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length) parseCSVFile(files[0]);
        });

        fileInput.addEventListener('change', (e) => {
            if (fileInput.files.length) parseCSVFile(fileInput.files[0]);
        });

        sampleBtn.addEventListener('click', loadSampleCSV);
        downloadBtn.addEventListener('click', exportResultsCSV);
    }

    function loadSampleCSV() {
        const sampleCSV = `Age,Experience,Education,Occupation,HoursPerWeek,WorkMode,CompanySize
32,6,bachelors,swe,40,hybrid,mid
45,18,masters,mgmt,50,onsite,enterprise
28,4,bachelors,ds_ai,45,remote,startup
52,24,doctorate,edu,35,hybrid,mid
38,12,associate,finance,42,onsite,enterprise
26,2,highschool,trades,40,onsite,mid
41,15,masters,health,48,onsite,enterprise
30,7,bachelors,design,40,remote,startup`;

        processCSVText(sampleCSV);
    }

    function parseCSVFile(file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            processCSVText(e.target.result);
        };
        reader.readAsText(file);
    }

    function processCSVText(csvText) {
        const lines = csvText.trim().split('\n');
        if (lines.length < 2) return;

        const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
        batchDataStore = [];

        for (let i = 1; i < lines.length; i++) {
            const vals = lines[i].split(',').map(v => v.trim());
            if (vals.length < headers.length) continue;

            const rowObj = {
                age: parseInt(vals[0]) || 32,
                experience: parseInt(vals[1]) || 5,
                education: vals[2] || 'bachelors',
                occupation: vals[3] || 'swe',
                hours: parseInt(vals[4]) || 40,
                workMode: vals[5] || 'hybrid',
                companySize: vals[6] || 'mid',
                locationTier: 'tier1',
                gender: 'female'
            };

            const result = calculateEarnings(rowObj);
            batchDataStore.push({ rowObj, result });
        }

        updateBatchDisplay();
    }

    function updateBatchDisplay() {
        if (!batchDataStore.length) return;

        const wrapper = document.getElementById('batch-results-wrapper');
        const tbody = document.getElementById('batch-tbody');
        const countBadge = document.getElementById('batch-count-badge');
        const avgBadge = document.getElementById('batch-avg-badge');

        wrapper.style.display = 'block';
        tbody.innerHTML = '';

        const curr = CURRENCY_CONFIG[currentCurrency];
        let sumSalaryUSD = 0;

        batchDataStore.forEach((item, index) => {
            const convSalary = Math.round(item.result.totalUSD * curr.rate);
            sumSalaryUSD += item.result.totalUSD;

            const monthly = Math.round(convSalary / 12);
            const percentile = Math.min(99, Math.max(5, Math.round((item.result.totalUSD - 20000) / 180000 * 100)));

            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${index + 1}</td>
                <td><strong>${item.rowObj.occupation.toUpperCase()}</strong></td>
                <td>${item.rowObj.education}</td>
                <td>${item.rowObj.age}</td>
                <td>${item.rowObj.experience} yrs</td>
                <td>${item.rowObj.hours} hrs</td>
                <td style="color: var(--secondary); font-weight: 700;">${curr.symbol}${convSalary.toLocaleString()}</td>
                <td>${curr.symbol}${monthly.toLocaleString()}</td>
                <td>Top ${100 - percentile}%</td>
            `;
            tbody.appendChild(tr);
        });

        const avgSalary = Math.round((sumSalaryUSD / batchDataStore.length) * curr.rate);
        countBadge.textContent = `${batchDataStore.length} Profiles Processed`;
        avgBadge.textContent = `Avg Salary: ${curr.symbol}${avgSalary.toLocaleString()}`;
    }

    function exportResultsCSV() {
        if (!batchDataStore.length) return;

        const curr = CURRENCY_CONFIG[currentCurrency];
        let csvContent = `ID,Occupation,Education,Age,Experience,HoursPerWeek,PredictedSalary_${currentCurrency},Monthly_${currentCurrency}\n`;

        batchDataStore.forEach((item, index) => {
            const convSalary = Math.round(item.result.totalUSD * curr.rate);
            const monthly = Math.round(convSalary / 12);
            csvContent += `${index + 1},${item.rowObj.occupation},${item.rowObj.education},${item.rowObj.age},${item.rowObj.experience},${item.rowObj.hours},${convSalary},${monthly}\n`;
        });

        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        link.setAttribute('download', `predicted_employee_salaries_${currentCurrency}.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
});
