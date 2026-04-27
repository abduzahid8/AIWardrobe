
const SUPABASE_URL = 'https://fyqpifmrsftsfqibhwhy.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ5cXBpZm1yc2Z0c2ZxaWJod2h5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQzMDYyOTgsImV4cCI6MjA3OTg4MjI5OH0.dydnFn3lqub7qMo9uFfn5yUyY4Wr_eQPnsbvHWHwMTk';

async function verifyNvidiaKey() {
    console.log('Verifying NVIDIA API Key via Diagnostic Mode...');
    try {
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
        console.log('Result:', JSON.stringify(data, null, 2));
    } catch (error) {
        console.error('Test failed:', error);
    }
}

verifyNvidiaKey();
