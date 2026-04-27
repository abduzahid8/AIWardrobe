const SUPABASE_URL = 'https://fyqpifmrsftsfqibhwhy.supabase.co';
const ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ5cXBpZm1yc2Z0c2ZxaWJod2h5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQzMDYyOTgsImV4cCI6MjA3OTg4MjI5OH0.dydnFn3lqub7qMo9uFfn5yUyY4Wr_eQPnsbvHWHwMTk';
const FUNCTION_URL = `${SUPABASE_URL}/functions/v1/try-on`; // Note: testing try-on NOT mannequin-tryon

const payload = {
    person_image: "https://image.uniqlo.com/UQ/ST3/us/imagesgoods/456630/item/usgoods_64_456630_3x4.jpg",
    garment_image: "https://image.uniqlo.com/UQ/ST3/us/imagesgoods/456630/item/usgoods_64_456630_3x4.jpg",
    garment_type: "upper_body"
};

async function testTryOn() {
    const response = await fetch(FUNCTION_URL, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${ANON_KEY}`
        },
        body: JSON.stringify(payload)
    });
    console.log("TRY-ON STATUS:", response.status);
    console.log("TRY-ON RESPONSE:", (await response.text()).slice(0, 500));
}

testTryOn();
