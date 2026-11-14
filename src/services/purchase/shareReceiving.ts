import { parseIncomingProduct } from './productInput';
export function isPurchaseShareLink(input: string): boolean {
  try {
    const url = new URL(input);
    return (
      url.protocol === 'com.aiwardrobe:' &&
      (url.hostname === 'buy-check' || url.pathname === '/buy-check')
    );
  } catch {
    return false;
  }
}
export function purchaseShareLink(input: string): string {
  const value = input.trim().slice(0, 4096);
  return `com.aiwardrobe://buy-check?input=${encodeURIComponent(value)}`;
}
export function decodePurchaseShare(input: string): { input?: string; error?: string } {
  try {
    const url = new URL(input);
    if (!isPurchaseShareLink(input)) return { error: 'Unsupported share route' };
    const text = url.searchParams.get('input');
    if (!text || text.length > 4096) return { error: 'Empty or oversized shared product' };
    const parsed = parseIncomingProduct(text);
    return parsed.error ? { error: parsed.error } : { input: parsed.url || parsed.title };
  } catch {
    return { error: 'Malformed share link' };
  }
}
export function parseShoppingIntent(text: string) {
  const budget = text.match(/(?:RM|MYR|USD|EUR|GBP|UZS|\$|€|£)\s*(\d+(?:\.\d{1,2})?)/i);
  const currency = budget
    ? { RM: 'MYR', $: 'USD', '€': 'EUR', '£': 'GBP' }[
        budget[0]
          .match(/^[^\d]+/)![0]
          .trim()
          .toUpperCase()
      ] ||
      budget[0]
        .match(/^[^\d]+/)![0]
        .trim()
        .toUpperCase()
    : undefined;
  const mode = /what.*buy|spend.*budget|i have|wardrobe gain/i.test(text)
    ? 'budget'
    : /cheaper|less money|save money/i.test(text)
      ? 'save'
      : /upgrade|improve.*look/i.test(text)
        ? 'upgrade'
        : /missing|complete.*wardrobe/i.test(text)
          ? 'complete'
          : /https?:|check.*buy/i.test(text)
            ? 'check'
            : 'find';
  return {
    mode: mode as 'budget' | 'save' | 'upgrade' | 'complete' | 'check' | 'find',
    budget: budget && currency ? { amount: Number(budget[1]), currency } : undefined,
    query: text
      .replace(budget?.[0] || /$^/, '')
      .replace(/\b(under|below|find|please|for me)\b/gi, ' ')
      .trim()
      .slice(0, 200),
  };
}
