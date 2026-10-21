type AuthEventHandler = (userId: string) => void;

let revision = 0;
let currentUserId: string | null = null;
let signedOut = false;

const loginHandlers: AuthEventHandler[] = [];
const logoutHandlers: Array<() => void> = [];

export const authEvents = {
  getRevision() { return revision; },
  /** Strict identity for caches: unknown startup state is not an authenticated owner. */
  getUserId() { return signedOut ? null : currentUserId; },
  canApplyToUser(userId?: string) {
    return !signedOut && (!userId || currentUserId === null || currentUserId === userId);
  },
  onLogin(handler: AuthEventHandler) {
    loginHandlers.push(handler);
    return () => {
      const index = loginHandlers.indexOf(handler);
      if (index >= 0) {
        loginHandlers.splice(index, 1);
      }
    };
  },

  onLogout(handler: () => void) {
    logoutHandlers.push(handler);
    return () => {
      const index = logoutHandlers.indexOf(handler);
      if (index >= 0) {
        logoutHandlers.splice(index, 1);
      }
    };
  },

  emitLogin(userId: string) {
    signedOut = false;
    if (currentUserId !== userId) { revision++; currentUserId = userId; }
    loginHandlers.forEach((handler) => handler(userId));
  },

  emitLogout() {
    signedOut = true;
    revision++;
    currentUserId = null;
    logoutHandlers.forEach((handler) => handler());
  },
};
