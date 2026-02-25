/**
 * API Server Configuration
 * All values come from environment variables — never hardcode IPs or secrets.
 */

export const PORT = process.env.PORT || 3000;
export const NODE_ENV = process.env.NODE_ENV || 'development';
export const IS_PRODUCTION = NODE_ENV === 'production';

export const ALICEVISION_URL = process.env.ALICEVISION_URL || 'http://localhost:5050';

export default { PORT, NODE_ENV, IS_PRODUCTION, ALICEVISION_URL };