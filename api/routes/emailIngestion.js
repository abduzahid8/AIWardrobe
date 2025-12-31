/**
 * Email Ingestion Routes
 * Handles Gmail OAuth and receipt parsing for automatic wardrobe population
 */

import express from 'express';
import { google } from 'googleapis';
import ClothingItem from '../models/ClothingItem.js';
import User from '../models/user.js';

const router = express.Router();

// Note: Receipt parser services may not exist yet
const parseReceipt = (data) => [];
const isClothingReceipt = (data) => false;
const batchFetchImages = async (items) => items.map(() => null);

// Gmail OAuth2 Configuration
const oauth2Client = new google.auth.OAuth2(
    process.env.GMAIL_CLIENT_ID,
    process.env.GMAIL_CLIENT_SECRET,
    process.env.GMAIL_REDIRECT_URI || 'http://localhost:3000/api/email/callback'
);

const SCOPES = ['https://www.googleapis.com/auth/gmail.readonly'];

/**
 * GET /api/email/auth-url
 * Generate Gmail OAuth authorization URL
 */
router.get('/auth-url', (req, res) => {
    try {
        const userId = req.query.userId;

        if (!userId) {
            return res.status(400).json({ error: 'userId required' });
        }

        const authUrl = oauth2Client.generateAuthUrl({
            access_type: 'offline',
            scope: SCOPES,
            state: userId, // Pass userId to callback
            prompt: 'consent'
        });

        res.json({ authUrl });
    } catch (error) {
        console.error('Error generating auth URL:', error);
        res.status(500).json({ error: error.message });
    }
});

/**
 * GET /api/email/callback
 * OAuth callback handler
 */
router.get('/callback', async (req, res) => {
    const { code, state: userId } = req.query;

    try {
        // Exchange code for tokens
        const { tokens } = await oauth2Client.getToken(code);

        // Save tokens to user record
        await User.findByIdAndUpdate(userId, {
            gmailRefreshToken: tokens.refresh_token,
            gmailAccessToken: tokens.access_token,
            gmailTokenExpiry: tokens.expiry_date
        });

        // Redirect to success page (frontend will handle)
        res.redirect(`/email-connected?success=true`);
    } catch (error) {
        console.error('OAuth callback error:', error);
        res.redirect(`/email-connected?success=false&error=${encodeURIComponent(error.message)}`);
    }
});

/**
 * POST /api/email/scan-receipts
 * Scan Gmail for clothing purchase receipts
 */
router.post('/scan-receipts', async (req, res) => {
    const { userId, maxResults = 100, maxAge = '1y' } = req.body;

    try {
        console.log(`📧 Scanning receipts for user ${userId}...`);

        // Get user's Gmail tokens
        const user = await User.findById(userId);
        if (!user || !user.gmailRefreshToken) {
            return res.status(400).json({
                error: 'Gmail not connected. Please authorize first.'
            });
        }

        // Set OAuth credentials
        oauth2Client.setCredentials({
            refresh_token: user.gmailRefreshToken,
            access_token: user.gmailAccessToken
        });

        const gmail = google.gmail({ version: 'v1', auth: oauth2Client });

        // Search for receipt emails
        // Common receipt subjects/senders
        const queries = [
            'subject:(order OR receipt OR purchase OR confirmation) clothing',
            'from:(zara OR hm.com OR uniqlo OR gap OR asos OR shein)',
            'from:(wildberries OR ozon OR lamoda)', // Russian retailers
            'subject:заказ', // Russian "order"
        ];

        const allMessages = [];

        for (const query of queries) {
            const response = await gmail.users.messages.list({
                userId: 'me',
                q: `${query} newer_than:${maxAge}`,
                maxResults: Math.floor(maxResults / queries.length)
            });

            if (response.data.messages) {
                allMessages.push(...response.data.messages);
            }
        }

        console.log(`Found ${allMessages.length} potential receipt emails`);

        // Fetch full email content and parse
        const parsedReceipts = [];

        for (const message of allMessages.slice(0, maxResults)) {
            try {
                const email = await gmail.users.messages.get({
                    userId: 'me',
                    id: message.id,
                    format: 'full'
                });

                const emailData = parseEmailData(email.data);

                // Check if it's a clothing receipt
                if (isClothingReceipt(emailData)) {
                    const items = parseReceipt(emailData);
                    if (items && items.length > 0) {
                        parsedReceipts.push({
                            emailId: message.id,
                            items
                        });
                    }
                }
            } catch (err) {
                console.error(`Error processing message ${message.id}:`, err.message);
            }
        }

        console.log(`Parsed ${parsedReceipts.length} clothing receipts`);

        // Flatten all items
        const allItems = parsedReceipts.flatMap(r => r.items);

        // Fetch product images
        console.log(`Fetching images for ${allItems.length} items...`);
        const images = await batchFetchImages(allItems);

        // Attach images to items
        allItems.forEach((item, index) => {
            item.imageUrl = images[index];
        });

        res.json({
            success: true,
            receiptsScanned: allMessages.length,
            receiptsFound: parsedReceipts.length,
            itemsDetected: allItems.length,
            items: allItems
        });

    } catch (error) {
        console.error('Error scanning receipts:', error);
        res.status(500).json({ error: error.message });
    }
});

/**
 * POST /api/email/import-items
 * Import scanned items into user's wardrobe
 */
router.post('/import-items', async (req, res) => {
    const { userId, items } = req.body;

    try {
        console.log(`Importing ${items.length} items for user ${userId}`);

        const savedItems = [];

        for (const item of items) {
            const clothingItem = new ClothingItem({
                userId,
                itemType: item.itemType,
                color: item.color,
                style: item.description,
                material: item.material,
                brand: item.retailer,
                size: item.size,
                price: item.price,
                purchaseDate: item.purchaseDate,
                imageUrl: item.imageUrl,
                description: item.description,
                source: 'email_receipt',
                sourceMetadata: {
                    rawText: item.rawText,
                    retailer: item.retailer
                }
            });

            await clothingItem.save();
            savedItems.push(clothingItem);
        }

        console.log(`✅ Imported ${savedItems.length} items successfully`);

        res.json({
            success: true,
            itemsImported: savedItems.length,
            items: savedItems
        });

    } catch (error) {
        console.error('Error importing items:', error);
        res.status(500).json({ error: error.message });
    }
});

/**
 * GET /api/email/status
 * Check if user has Gmail connected
 */
router.get('/status', async (req, res) => {
    const { userId } = req.query;

    try {
        const user = await User.findById(userId);
        const connected = !!(user && user.gmailRefreshToken);

        res.json({
            connected,
            email: connected ? user.email : null
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * DELETE /api/email/disconnect
 * Revoke Gmail access
 */
router.delete('/disconnect', async (req, res) => {
    const { userId } = req.body;

    try {
        await User.findByIdAndUpdate(userId, {
            $unset: {
                gmailRefreshToken: '',
                gmailAccessToken: '',
                gmailTokenExpiry: ''
            }
        });

        res.json({ success: true, message: 'Gmail disconnected' });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * Helper: Parse Gmail API response into usable format
 */
function parseEmailData(emailResponse) {
    const headers = emailResponse.payload.headers;
    const subject = headers.find(h => h.name.toLowerCase() === 'subject')?.value || '';
    const from = headers.find(h => h.name.toLowerCase() === 'from')?.value || '';
    const date = headers.find(h => h.name.toLowerCase() === 'date')?.value || '';

    // Extract body
    let body = '';
    let html = '';

    if (emailResponse.payload.parts) {
        for (const part of emailResponse.payload.parts) {
            if (part.mimeType === 'text/plain' && part.body.data) {
                body += Buffer.from(part.body.data, 'base64').toString('utf-8');
            }
            if (part.mimeType === 'text/html' && part.body.data) {
                html += Buffer.from(part.body.data, 'base64').toString('utf-8');
            }
        }
    } else if (emailResponse.payload.body.data) {
        body = Buffer.from(emailResponse.payload.body.data, 'base64').toString('utf-8');
    }

    return { subject, from, date, body, html };
}

export default router;
