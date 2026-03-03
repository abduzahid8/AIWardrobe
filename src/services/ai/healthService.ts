/**
 * Health Service — Server and AliceVision health checks.
 */

import axios from 'axios';
import Config from '../../config/env';

const API_URL = Config.api.url;
const ALICEVISION_URL = Config.api.alicevisionUrl;

export async function checkServerHealth(): Promise<{ healthy: boolean; message: string }> {
    try {
        await axios.get(`${API_URL}/health`, { timeout: 5000 });
        return { healthy: true, message: 'Server is running' };
    } catch {
        return { healthy: false, message: 'Server is currently unavailable' };
    }
}

export async function checkAliceVisionHealth(): Promise<{
    healthy: boolean;
    features: string[];
    message: string;
}> {
    try {
        const response = await axios.get(`${ALICEVISION_URL}/health`, { timeout: 5000 });
        return {
            healthy: true,
            features: response.data.features || [],
            message: 'AliceVision AI service is running',
        };
    } catch {
        return {
            healthy: false,
            features: [],
            message: 'AliceVision AI service is unavailable',
        };
    }
}
