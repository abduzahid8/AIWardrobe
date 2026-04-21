const isDev = __DEV__;

interface LogMethod {
    (message: string, ...args: unknown[]): void;
}

interface Logger {
    info: LogMethod;
    warn: LogMethod;
    error: LogMethod;
    debug: LogMethod;
}

const createLogger = (scope: string): Logger => {
    const prefix = `[${scope}]`;
    return {
        info: (message: string, ...args: unknown[]) => {
            if (isDev) console.log(`${prefix} INFO: ${message}`, ...args);
        },
        warn: (message: string, ...args: unknown[]) => {
            if (isDev) console.warn(`${prefix} WARN: ${message}`, ...args);
        },
        error: (message: string, ...args: unknown[]) => {
            console.error(`${prefix} ERROR: ${message}`, ...args);
        },
        debug: (message: string, ...args: unknown[]) => {
            if (isDev) console.log(`${prefix} DEBUG: ${message}`, ...args);
        },
    };
};

const defaultLogger = createLogger('App');

export { createLogger };
export default defaultLogger;
