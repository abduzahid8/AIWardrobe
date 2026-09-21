export const SUPABASE_AUTH_ERRORS = {
  INVALID_CREDENTIALS: 'Invalid login credentials',
  EMAIL_NOT_CONFIRMED: 'Email not confirmed',
  EMAIL_CONFIRMATION_REQUIRED: 'EMAIL_CONFIRMATION_REQUIRED',
  USER_ALREADY_REGISTERED: 'User already registered',
  USERNAME_EXISTS: 'Username already exists',
  DATABASE_ERROR: 'Database error',
  NETWORK_FAILED: 'Network request failed',
  RATE_LIMITED: 'rate limit',
  TOO_MANY_REQUESTS: 'too many',
  PKCE_CODE: 'code',
  OAUTH_CANCELLED: 'cancel',
} as const;
