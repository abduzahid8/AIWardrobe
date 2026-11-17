import type { VerifiedSaving } from '../../src/types/growth';
import { totalVerifiedSavings } from '../../src/services/growth/monetization';

/** Test helper: empty ledger totals zero. */
export function totalVerifiedSavingsShim(): number {
    const ledger: VerifiedSaving[] = [];
    return totalVerifiedSavings(ledger);
}
