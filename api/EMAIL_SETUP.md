# Email Ingestion Environment Variables

Add these environment variables to `/api/.env`:

```env
# Gmail OAuth Configuration
GMAIL_CLIENT_ID=your_google_client_id
GMAIL_CLIENT_SECRET=your_google_client_secret
GMAIL_REDIRECT_URI=http://localhost:3000/api/email/callback

# For production, use your deployed API URL:
# GMAIL_REDIRECT_URI=https://your-api.herokuapp.com/api/email/callback
```

## Setup Instructions

### 1. Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable the **Gmail API**:
   - Go to "APIs & Services" → "Enable APIs and Services"
   - Search for "Gmail API"
   - Click "Enable"

### 2. Create OAuth 2.0 Credentials

1. Go to "APIs & Services" → "Credentials"
2. Click "+ CREATE CREDENTIALS" → "OAuth client ID"
3. Application type: **Web application**
4. Name: "AIWardrobe Email Ingestion"
5. Authorized redirect URIs:
   - `http://localhost:3000/api/email/callback` (for local development)
   - `https://your-api.vercel.app/api/email/callback` (for production)
6. Click "Create"
7. Copy the **Client ID** and **Client Secret**

### 3. Update .env File

Paste the credentials into `/api/.env`:

```env
GMAIL_CLIENT_ID=123456789-abcdefghij.apps.googleusercontent.com
GMAIL_CLIENT_SECRET=GOCSPX-aBcDeFgHiJkLmNoPqRsTuVwXyZ
GMAIL_REDIRECT_URI=http://localhost:3000/api/email/callback
```

### 4. Test the Integration

1. Restart API server: `cd api && npm start`
2. In the app, navigate to EmailOnboarding screen
3. Click "Connect Gmail"
4. Authorize the app in browser
5. Return to app and click "Scan My Receipts"

## Privacy & Security

- We only request READ permission to Gmail
- Scope: `https://www.googleapis.com/auth/gmail.readonly`
- Refresh tokens are stored encrypted in MongoDB
- Users can disconnect anytime via Settings

## Supported Retailers

The receipt parser currently supports:

### International
- Zara, H&M, Uniqlo, Gap, Nordstrom
- Amazon, ASOS, SHEIN
- YOOX, Farfetch, Net-A-Porter

### Russian/CIS
- Wildberries, Ozon, Lamoda
- (More can be added easily)

## Troubleshooting

### "OAuth callback error"
- Check that redirect URI matches exactly in Google Console and .env
- Ensure Gmail API is enabled

### "No receipts found"
- Try "maxAge: '2y'" to scan longer history
- Check that user has clothing purchase emails

### "Rate limit exceeded"
- Google has quota limits (10,000 requests/day free tier)
- Consider implementing caching for parsed receipts
