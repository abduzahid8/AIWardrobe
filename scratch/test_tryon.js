
const SUPABASE_URL = 'https://fyqpifmrsftsfqibhwhy.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ5cXBpZm1yc2Z0c2ZxaWJod2h5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQzMDYyOTgsImV4cCI6MjA3OTg4MjI5OH0.dydnFn3lqub7qMo9uFfn5yUyY4Wr_eQPnsbvHWHwMTk';

async function testEndpoint(name, payload) {
    console.log(`\n--- Testing ${name} ---`);
    try {
        const response = await fetch(`${SUPABASE_URL}/functions/v1/mannequin-tryon`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'apikey': SUPABASE_ANON_KEY,
                'Authorization': `Bearer ${SUPABASE_ANON_KEY}`
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();
        console.log('Status:', response.status);
        console.log('Response:', JSON.stringify(data, null, 2));
    } catch (error) {
        console.error('Fetch error:', error);
    }
}

// ... existing constants ...

async function testAuthWithLLM() {
    console.log('\n--- Verifying API Key with LLM (Authentication Check) ---');
    // Using the same URL and model as analyze-outfit/index.ts
    const NVIDIA_LLM_URL = 'https://integrate.api.nvidia.com/v1/chat/completions';
    
    // We need to fetch the token from the DB or use the one we know?
    // Actually, I'll tell the edge function to run a diagnostic.
    
    const response = await fetch(`${SUPABASE_URL}/functions/v1/mannequin-tryon`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'apikey': SUPABASE_ANON_KEY,
            'Authorization': `Bearer ${SUPABASE_ANON_KEY}`
        },
        body: JSON.stringify({ diagnostic: true })
    });

    const data = await response.json();
    console.log('Diagnostic Result:', JSON.stringify(data, null, 2));
}

runAllTests().then(() => testAuthWithLLM());
