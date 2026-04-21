/**
 * Auth Events — decouples auth store from wardrobe store
 *
 * Instead of auth.ts importing wardrobeStore (circular),
 * auth emits events that wardrobeStore subscribes to.
 */

type AuthEventHandler = (userId: string) => void;

const loginHandlers: AuthEventHandler[] = [];
const logoutHandlers: (() => void)[] = [];

export const authEvents = {
    onLogin(handler: AuthEventHandler) {
        loginHandlers.push(handler);
        return () => {
            const idx = loginHandlers.indexOf(handler);
            if (idx >= 0) loginHandlers.splice(idx, 1);
        };
    },

    onLogout(handler: () => void) {
        logoutHandlers.push(handler);
        return () => {
            const idx = logoutHandlers.indexOf(handler);
            if (idx >= 0) logoutHandlers.splice(idx, 1);
        };
    },

    emitLogin(userId: string) {
        loginHandlers.forEach((h) => h(userId));
    },

    emitLogout() {
        logoutHandlers.forEach((h) => h());
    },
};
