import jwt from 'jsonwebtoken';
import fs from 'fs';
import path from 'path';

// @ts-ignore - jsonwebtoken types not installed for this script

// CONFIGURATION - UPDATE THESE VALUES
const KEY_ID = 'FR4VBVZ9PT'; // Your Key ID from Apple Developer Portal
const TEAM_ID = '4CBV6A6D4G'; // Your Team ID from Apple Developer Portal
const P8_FILE_PATH = path.join(__dirname, 'AuthKey_FR4VBVZ9PT.p8'); // Path to your downloaded .p8 file

// JWT payload for Apple Sign In client secret
const payload = {
  iss: TEAM_ID,
  iat: Math.floor(Date.now() / 1000),
  exp: Math.floor(Date.now() / 1000) + (180 * 24 * 60 * 60), // 180 days (max allowed)
  aud: 'https://appleid.apple.com',
  sub: 'com.aiwardrobe' // Your bundle ID
};

try {
  // Read the .p8 private key file
  const privateKey = fs.readFileSync(P8_FILE_PATH, 'utf8');
  
  // Generate the JWT
  const token = jwt.sign(payload, privateKey, {
    algorithm: 'ES256',
    keyid: KEY_ID
  });
  
  console.log('==========================================');
  console.log('APPLE SIGN IN SECRET KEY (JWT)');
  console.log('==========================================');
  console.log(token);
  console.log('==========================================');
  console.log('\nCopy the token above and paste it into Supabase:');
  console.log('Authentication → Providers → Apple → Secret Key (for OAuth)');
  console.log('\nNote: This token expires in 180 days. Regenerate when needed.');
} catch (error: any) {
  console.error('Error generating JWT:', error?.message || error);
  console.error('\nMake sure:');
  console.error('1. You have installed jsonwebtoken: npm install jsonwebtoken');
  console.error('2. The .p8 file path is correct');
  console.error('3. You have updated TEAM_ID in this script');
  process.exit(1);
}
