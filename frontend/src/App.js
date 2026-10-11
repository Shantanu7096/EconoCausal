import React, { useState } from 'react';

function App() {
const [budget, setBudget] = useState(5000);
const [file, setFile] = useState(null);

const handleUpload = (e) => {
    e.preventDefault();
    if (!file) return alert("Please select a CSV file.");
    alert(`File "${file.name}" uploaded successfully with budget: $${budget}`);
};

return (
    <div style={{ padding: '2rem', fontFamily: 'sans-serif' }}>
    <h2>EconoCausal: Dynamic Pricing Engine</h2>
    <p>Welcome to the EconoCausal Dynamic Pricing Engine!</p>
    <form onSubmit={handleUpload} style={{ maxWidth: '400px', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <div>
        <label><strong>1. Upload Campaign Data (CSV):</strong></label><br />
        <input type="file" accept=".csv" onChange={(e) => setFile(e.target.files[0])} />
        </div>
        <div>
        <label><strong>2. Total Promotional Budget ($):</strong></label><br />
        <input
            type="number"
            value={budget}
            onChange={(e) => setBudget(e.target.value)}
            style={{ width: '100%', padding: '8px' }}
        />
        </div>
        <button type="submit" style={{ padding: '10px', background: '#0066cc', color: 'white', border: 'none', borderRadius: '4px' }}>
        Initialize Causal Uplift Pipeline
        </button>
    </form>
    </div>
);
}

export default App;