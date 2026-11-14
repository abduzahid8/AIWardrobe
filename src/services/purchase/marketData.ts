import { authEvents } from '../../../store/authEvents';
import { LocalPriceRepository } from './priceIntelligence';
import { createPriceReadRepository } from './priceRepository';
/** Providers may record actual, dated observations here. Reading/checking an
 * existing catalog or manual quote never creates a new market observation. */
export const purchaseMarket = new LocalPriceRepository();
let owner = authEvents.getUserId();
authEvents.onLogin((next) => {
  if (owner !== next) {
    purchaseMarket.clear();
    owner = next;
  }
});
authEvents.onLogout(() => {
  purchaseMarket.clear();
  owner = null;
});
// Public read path only; default off until the prepared schema/data rollout.
// This task never sets the flag, provisions collectors, or enables notifications.
export const purchasePriceReads = createPriceReadRepository(
  purchaseMarket,
  process.env.EXPO_PUBLIC_PURCHASE_PRICE_READS === 'true'
);
