/**
 * Single source of truth shared by the localhost bridge and dev launcher.
 * Reusing a responsive previous-revision listener silently serves stale
 * scoring and defeats screenshot reproducibility after git pull.
 */
export const STYLIST6_DEV_ENGINE_VERSION = 'stylist-6-dev-local-20261112-r3';

export function isCurrentStylistBridgeHealth(payload) {
  return payload != null && typeof payload === 'object' &&
    payload.engine === STYLIST6_DEV_ENGINE_VERSION && payload.ok === true;
}
