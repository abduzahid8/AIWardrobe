type AuthEventHandler = (userId: string) => void;

const loginHandlers: AuthEventHandler[] = [];
const logoutHandlers: Array<() => void> = [];

export const authEvents = {
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
    loginHandlers.forEach((handler) => handler(userId));
  },

  emitLogout() {
    logoutHandlers.forEach((handler) => handler());
  },
};
