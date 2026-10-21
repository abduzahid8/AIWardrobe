/** Calendar days are local civil dates, never UTC timestamps. */
export function localDateKey(date: Date): string {
  return `${String(date.getFullYear()).padStart(4, '0')}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
}

export function parseCalendarDate(value: unknown): Date | null {
  if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return null;
  const [year, month, day] = value.split('-').map(Number);
  if (year < 1 || month < 1 || month > 12 || day < 1 || day > 31) return null;
  // setFullYear avoids JS's special treatment of constructor years 0..99.
  const date = new Date(2000, 0, 1, 12);
  date.setFullYear(year, month - 1, day);
  return localDateKey(date) === value ? date : null;
}

export function shiftCalendarDay(date: Date, days: number): Date {
  const shifted = new Date(date.getTime());
  shifted.setDate(shifted.getDate() + days);
  return shifted;
}
