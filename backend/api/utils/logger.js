/**
 * Structured Logger
 * Replaces console.log with a proper logging utility.
 * In production, only WARN and ERROR level logs are emitted.
 * In development, all levels are emitted.
 */
const LOG_LEVELS = { DEBUG: 0, INFO: 1, WARN: 2, ERROR: 3 };

const isProduction = process.env.NODE_ENV === 'production';
const minLevel = isProduction ? LOG_LEVELS.WARN : LOG_LEVELS.DEBUG;

const formatMessage = (level, context, message, data) => {
    const timestamp = new Date().toISOString();
    const prefix = `[${timestamp}] [${level}]${context ? ` [${context}]` : ''}`;
    if (data !== undefined) {
        return `${prefix} ${message} ${typeof data === 'object' ? JSON.stringify(data) : data}`;
    }
    return `${prefix} ${message}`;
};

const logger = {
    debug(message, data, context) {
        if (minLevel <= LOG_LEVELS.DEBUG) {
            console.log(formatMessage('DEBUG', context, message, data));
        }
    },
    info(message, data, context) {
        if (minLevel <= LOG_LEVELS.INFO) {
            console.log(formatMessage('INFO', context, message, data));
        }
    },
    warn(message, data, context) {
        if (minLevel <= LOG_LEVELS.WARN) {
            console.warn(formatMessage('WARN', context, message, data));
        }
    },
    error(message, data, context) {
        if (minLevel <= LOG_LEVELS.ERROR) {
            console.error(formatMessage('ERROR', context, message, data));
        }
    },
};

export default logger;
