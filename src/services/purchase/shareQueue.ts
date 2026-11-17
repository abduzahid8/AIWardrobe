import { authEvents } from '../../../store/authEvents';
import { decodePurchaseShare, isPurchaseShareLink } from './shareReceiving';
let pending: { input: string; owner: string | null; expiresAt: number } | undefined;
/** One bounded intent in memory, never disk/analytics. An anonymous cold-start
 * intent binds to the first authenticated owner; it cannot cross later logouts. */
export function queuePurchaseLink(url: string, now = Date.now()): boolean {
  if (!isPurchaseShareLink(url)) return false;
  const decoded = decodePurchaseShare(url);
  if (decoded.input)
    pending = { input: decoded.input, owner: authEvents.getUserId(), expiresAt: now + 15 * 60000 };
  return true;
}
export function takePurchaseInput(owner: string, now = Date.now()): string | undefined {
  const input = pending;
  if (!input) return;
  if (input.expiresAt < now || (input.owner && input.owner !== owner)) {
    pending = undefined;
    return;
  }
  pending = undefined;
  return input.input;
}
export function purchaseAwareLinking(
  base: {
    getInitialURL: () => Promise<string | null>;
    subscribe: (listener: (url: string) => void) => () => void;
  },
  flush: () => void
) {
  return {
    async getInitialURL() {
      const url = await base.getInitialURL();
      if (url && queuePurchaseLink(url)) {
        flush();
        return null;
      }
      return url;
    },
    subscribe(listener: (url: string) => void) {
      return base.subscribe((url) => {
        if (queuePurchaseLink(url)) flush();
        else listener(url);
      });
    },
  };
}
authEvents.onLogin((owner) => {
  if (pending && !pending.owner) pending.owner = owner;
});
authEvents.onLogout(() => {
  if (pending?.owner) pending = undefined;
});
