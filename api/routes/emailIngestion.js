/**
 * Email Ingestion Routes
 * Handles Gmail OAuth and receipt parsing for automatic wardrobe population
 */

import express from 'express';
import { google } from 'googleapis';
import { supabase } from '../lib/supabase.js';
import { authenticateToken } from '../middleware/auth.js';
import logger from '../utils/logger.js';

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
router.get('/auth-url', authenticateToken, (req, res) => {
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
        logger.error('Error generating auth URL:', error);
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
        await supabase
            .from('profiles')
            .update({
                gmail_refresh_token: tokens.refresh_token,
                gmail_access_token: tokens.access_token,
                gmail_token_expiry: new Date(tokens.expiry_date).toISOString()
            })
            .eq('id', userId);

        // Redirect to success page (frontend will handle)
        res.redirect(`/email-connected?success=true`);
    } catch (error) {
        logger.error('OAuth callback error:', error);
        res.redirect(`/email-connected?success=false&error=${encodeURIComponent(error.message)}`);
    }
});

/**
 * POST /api/email/scan-receipts
 * Scan Gmail for clothing purchase receipts
 */
router.post('/scan-receipts', authenticateToken, async (req, res) => {
    const { userId, maxResults = 100, maxAge = '1y' } = req.body;

    try {
        logger.info(`Scanning receipts for user ${userId}...`, null, 'email');

        // Get user's Gmail tokens
        const { data: user } = await supabase
            .from('profiles')
            .select('gmail_refresh_token, gmail_access_token')
            .eq('id', userId)
            .single();

        if (!user || !user.gmail_refresh_token) {
            return res.status(400).json({
                error: 'Gmail not connected. Please authorize first.'
            });
        }

        // Set OAuth credentials
        oauth2Client.setCredentials({
            refresh_token: user.gmail_refresh_token,
            access_token: user.gmail_access_token
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

        logger.info(`Found ${allMessages.length} potential receipt emails`, null, 'email');

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
                logger.error(`Error processing message ${message.id}:`, err.message);
            }
        }

        logger.info(`Parsed ${parsedReceipts.length} clothing receipts`, null, 'email');

        // Flatten all items
        const allItems = parsedReceipts.flatMap(r => r.items);

        // Fetch product images
        logger.info(`Fetching images for ${allItems.length} items...`, null, 'email');
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
        logger.error('Error scanning receipts:', error);
        res.status(500).json({ error: error.message });
    }
});

/**
 * POST /api/email/import-items
 * Import scanned items into user's wardrobe
 */
router.post('/import-items', authenticateToken, async (req, res) => {
    const { userId, items } = req.body;

    try {
        logger.info(`Importing ${items.length} items for user ${userId}`, null, 'email');

        const savedItems = [];

        const itemData = {
            user_id: userId,
            type: item.itemType,
            category: "tops", // Fallback
            color: item.color,
            style: item.description || item.style,
            material: item.material,
            brand: item.retailer,
            size: item.size,
            price: item.price,
            purchase_date: item.purchaseDate,
            image_url: item.imageUrl,
            description: item.description,
            source_metadata: {
                rawText: item.rawText,
                retailer: item.retailer,
                source: 'email_receipt'
            }
        };

        const { data: savedItem, error } = await supabase
            .from('clothing_items')
            .insert([itemData])
            .select()
            .single();

        if (!error && savedItem) {
            savedItems.push(savedItem);
        }

        logger.info(`Imported ${savedItems.length} items successfully`, null, 'email');

        res.json({
            success: true,
            itemsImported: savedItems.length,
            items: savedItems
        });

    } catch (error) {
        logger.error('Error importing items:', error);
        res.status(500).json({ error: error.message });
    }
});

/**
 * GET /api/email/status
 * Check if user has Gmail connected
 */
router.get('/status', authenticateToken, async (req, res) => {
    const { userId } = req.query;

    try {
        const { data: user } = await supabase
            .from('profiles')
            .select('gmail_refresh_token, email')
            .eq('id', userId)
            .single();

        const connected = !!(user && user.gmail_refresh_token);

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
router.delete('/disconnect', authenticateToken, async (req, res) => {
    const { userId } = req.body;

    try {
        await supabase
            .from('profiles')
            .update({
                gmail_refresh_token: null,
                gmail_access_token: null,
                gmail_token_expiry: null
            })
            .eq('id', userId);

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
