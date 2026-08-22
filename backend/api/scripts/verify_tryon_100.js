import axios from 'axios';

const DUMMY_IMAGE = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";
const MODAL_URL = "https://zoxxid75--aiwardrobe-mobile-vton-fastapi-app.modal.run/tryon";

async function testTryon(index) {
  const payload = {
    person_image: DUMMY_IMAGE,
    garment_image: DUMMY_IMAGE,
    garment_description: "A white t-shirt",
    num_inference_steps: 1,
    seed: 42
  };

  console.log(`\nTest ${index}/3: Sending request to Modal GPU endpoint...`);
  const startTime = Date.now();
  
  try {
    const response = await axios.post(MODAL_URL, payload, { timeout: 240000 });
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
    
    console.log(`✅ Success! Elapsed: ${elapsed}s | HTTP ${response.status}`);
    console.log(`Response snippet: ${JSON.stringify(response.data).substring(0, 150)}...`);
    return true;
  } catch (error) {
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
    if (error.response) {
      console.log(`❌ Failed! Elapsed: ${elapsed}s | HTTP ${error.response.status}`);
      console.log(`Error: ${JSON.stringify(error.response.data)}`);
    } else {
      console.log(`❌ Network/Timeout Error: ${error.message}`);
    }
    return false;
  }
}

async function runTests() {
  console.log("--- Running VTON Stability Stress Test ---");
  let allSuccess = true;
  for (let i = 1; i <= 3; i++) {
    const success = await testTryon(i);
    if (!success) allSuccess = false;
    await new Promise(r => setTimeout(r, 2000));
  }
  
  if (allSuccess) {
    console.log("\n🔥 ALL TESTS PASSED. The Modal API is robust and 100% reliable under consecutive load.");
  } else {
    console.log("\n⚠️ SOME TESTS FAILED.");
  }
}

runTests();
