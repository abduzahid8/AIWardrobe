/**
 * parseStylistIntent — read occasion, style, and budget from the comment
 * field so users can type "work, old money, $200" and get outfits without
 * relying on chips.
 */
import { parseBudgetFromText, type ParsedBudget } from '../../../src/utils/outfitBudget';
import type { StyleId } from './styleInference';

export interface StylistIntent {
  style: StyleId | null;
  occasion: string | null;
  budget: ParsedBudget | null;
}

const STYLE_PATTERNS: Array<{ id: StyleId; re: RegExp }> = [
  { id: 'old_money', re: /\bold[\s_-]*money\b|\bquiet[\s_-]*luxury\b|\boldmoney\b|олд[\s_-]*мани/i },
  { id: 'business_casual', re: /\bbusiness[\s_-]*casual\b|\bmodern[\s_-]*professional\b|офисн/i },
  { id: 'semi_classic', re: /\bsemi[\s_-]*classic\b|полуклассик/i },
  { id: 'minimalist', re: /\bminimalist\b|\bminimal\b|минимал/i },
  { id: 'classic', re: /\bclassic\b|классик/i },
  { id: 'casual', re: /\bsmart[\s_-]*casual\b|\bcasual\b|кэжуал|\boddiy\b/i },
];

const OCCASION_PATTERNS: Array<{ id: string; re: RegExp }> = [
  { id: 'work', re: /\b(work|office|interview|meeting|business|corporate|ishxona)\b|работ|офис|собеседован/i },
  { id: 'date', re: /\b(date|romantic|dinner|uchrashuv)\b|свидан/i },
  { id: 'formal', re: /\b(formal|wedding|gala|black[\s_-]*tie)\b|свадьб|торжеств/i },
  { id: 'sport', re: /\b(sport|gym|workout|run)\b|спорт|зал/i },
  { id: 'travel', re: /\b(travel|airport|trip|flight|sayohat)\b|поездк|путешеств/i },
  { id: 'casual', re: /\b(brunch|weekend|everyday|casual)\b|выходн/i },
];

export function parseStylistIntent(text?: string | null): StylistIntent {
  // Chip ids must never be concatenated onto comments before parse — leftover
  // "work" / "old_money" chips would beat "brunch" / "casual" in the comment.
  const src = String(text || '').trim();
  if (!src) return { style: null, occasion: null, budget: null };

  let style: StyleId | null = null;
  for (const row of STYLE_PATTERNS) {
    if (row.re.test(src)) {
      style = row.id;
      break;
    }
  }

  let occasion: string | null = null;
  for (const row of OCCASION_PATTERNS) {
    if (row.re.test(src)) {
      occasion = row.id;
      break;
    }
  }

  return {
    style,
    occasion,
    budget: parseBudgetFromText(src),
  };
}
