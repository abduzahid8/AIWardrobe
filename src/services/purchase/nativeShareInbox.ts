import { AppState, NativeModules } from 'react-native';
import { authEvents } from '../../../store/authEvents';
import { parseIncomingProduct } from './productInput';
/** Optional bridge. Its absence in Expo Go or an unsigned iOS target is normal. */
export function listenForPurchaseInbox(receive: (input: string) => void) {
  const bridge = NativeModules.PurchaseShareInbox as
    | undefined
    | { setOwner: (owner: string) => void; consume: (owner: string) => Promise<string | null> };
  if (!bridge || typeof bridge.setOwner !== 'function' || typeof bridge.consume !== 'function')
    return () => undefined;
  let active = true;
  const check = async () => {
    const owner = authEvents.getUserId(),
      revision = authEvents.getRevision();
    if (!owner) return;
    try {
      bridge.setOwner(owner);
      const input = await bridge.consume(owner);
      if (
        !active ||
        revision !== authEvents.getRevision() ||
        owner !== authEvents.getUserId() ||
        !input
      )
        return;
      const parsed = parseIncomingProduct(input);
      if (!parsed.error) receive(parsed.url || parsed.title || '');
    } catch {
      /* No sensitive input or body data goes to error reporting. */
    }
  };
  void check();
  const app = AppState.addEventListener('change', (state) => {
    if (state === 'active') void check();
  });
  const login = authEvents.onLogin(() => void check()),
    logout = authEvents.onLogout(() => {
      try {
        bridge.setOwner('');
      } catch {
        /* Optional native capability must never break logout. */
      }
    });
  return () => {
    active = false;
    app.remove();
    login();
    logout();
  };
}
