import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { transpileModule, ModuleKind } from 'typescript';
import { createOutfitLog } from '../../features/calendar/types';

// Separate Node processes make timezone behavior independent of the Jest worker.
const source = transpileModule(readFileSync(require.resolve('../../src/utils/calendarDate.ts'), 'utf8'), {
  compilerOptions: { module: ModuleKind.CommonJS },
}).outputText;

it.each(['America/Los_Angeles', 'Pacific/Kiritimati', 'Asia/Tashkent', 'Europe/Berlin'])(
  'preserves calendar dates and day navigation in %s, including DST transitions', (TZ) => {
    const code = `${source}\n
      const checks = [
        localDateKey(parseCalendarDate('2026-03-08')),
        localDateKey(shiftCalendarDay(new Date(2026, 2, 8, 0, 30), 1)),
        localDateKey(shiftCalendarDay(new Date(2026, 10, 1, 0, 30), 1)),
        localDateKey(shiftCalendarDay(new Date(2026, 0, 1, 0, 30), -1)),
        localDateKey(shiftCalendarDay(new Date(2026, 2, 29, 0, 30), 1)),
        localDateKey(shiftCalendarDay(new Date(2026, 9, 25, 0, 30), 1)),
        localDateKey(parseCalendarDate('0099-01-01')),
        parseCalendarDate('2026-02-29'), parseCalendarDate('2026-04-31')
      ]; console.log(JSON.stringify(checks));`;
    const result = execFileSync(process.execPath, ['-e', code], { env: { ...process.env, TZ }, encoding: 'utf8' });
    expect(JSON.parse(result)).toEqual(['2026-03-08', '2026-03-09', '2026-11-02', '2025-12-31', '2026-03-30', '2026-10-26', '0099-01-01', null, null]);
  },
);

it.each(['2026-02-29', '2026-04-31', '2026-13-01', '2026-00-01', '2026-01-00', '2026-1-1', 'invalid'])(
  'rejects an impossible or malformed saved calendar date: %s', (date) => {
    expect(() => createOutfitLog(date, [{ id: 'top', type: 'top', image: 'file://top.png' }], 'casual')).toThrow('Invalid calendar date');
  },
);

test('accepts a leap day plan', () => {
  expect(createOutfitLog('2028-02-29', [{ id: 'top', type: 'top', image: 'file://top.png' }], 'casual').date).toBe('2028-02-29');
});
