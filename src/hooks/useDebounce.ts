import { useRef, useCallback } from 'react';

/**
 * useDebounce — prevents rapid button taps and duplicate API calls.
 *
 * @param callback - The function to debounce
 * @param delay - Minimum time between calls (ms), default 500ms
 *
 * @example
 * const debouncedSubmit = useDebounce(handleSubmit, 500);
 * <TouchableOpacity onPress={debouncedSubmit}>
 */
export function useDebounce<T extends (...args: any[]) => any>(
    callback: T,
    delay = 500,
): (...args: Parameters<T>) => void {
    const lastCallTime = useRef(0);

    return useCallback(
        (...args: Parameters<T>) => {
            const now = Date.now();
            if (now - lastCallTime.current >= delay) {
                lastCallTime.current = now;
                callback(...args);
            }
        },
        [callback, delay],
    );
}

export default useDebounce;
