import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { transpileModule, ModuleKind } from 'typescript';
import { createStore } from 'zustand/vanilla';
import { createWearLogSlice } from '../../store/slices/wardrobeWearLogSlice';
import { calculateStreak } from '../../store/slices/helpers';
import { localDateKey, shiftCalendarDay } from '../../src/utils/calendarDate';
import type { WardrobeState } from '../../store/wardrobeStore';
import type { ClothingItem, WearLog } from '../../src/types/domain';

const day = (offset: number) => localDateKey(shiftCalendarDay(new Date(), offset));
const log = (date: string, itemIds: string[]) => ({ date, itemIds } as WearLog);
function fixture() {
  return createStore<WardrobeState>()((set, get, store) => ({
    ...createWearLogSlice(set, get, store), items: [{ id: 'owned', wearCount: 0 } as ClothingItem],
    outfits: [], pendingActions: [], syncToServer: jest.fn().mockResolvedValue(undefined),
  } as unknown as WardrobeState));
}

test('utilization excludes deleted/external items and future wear logs', () => {
  const state = fixture();
  state.setState({ wearLogs: [log(day(0), ['owned', 'deleted', 'external']), log(day(1), ['future'])] });
  expect(state.getState().getClosetUtilization()).toBe(100);
  state.setState({ wearLogs: [log(day(0), ['deleted', 'external']), log(day(1), ['owned'])] });
  expect(state.getState().getClosetUtilization()).toBe(0);
  expect(state.getState().getUnwornItems().map(item => item.id)).toEqual(['owned']);
});

test('a 30-day window includes today and 29 previous local days', () => {
  const state = fixture();
  state.setState({ wearLogs: [log(day(-30), ['owned'])] });
  expect(state.getState().getClosetUtilization(30)).toBe(0);
  state.setState({ wearLogs: [log(day(-29), ['owned'])] });
  expect(state.getState().getClosetUtilization(30)).toBe(100);
});

test('streak ignores duplicates, invalid and future dates while retaining yesterday grace', () => {
  expect(calculateStreak([log(day(1), []), log('invalid', []), log(day(0), []), log(day(0), []), log(day(-1), [])])).toBe(2);
  expect(calculateStreak([log(day(-1), []), log(day(-2), [])])).toBe(2);
  expect(calculateStreak([log(day(-2), [])])).toBe(0);
});

test('empty wear input does not create history, counters or queued operations', () => {
  const state = fixture();
  state.getState().logWear([]);
  expect(state.getState().wearLogs).toEqual([]);
  expect(state.getState().pendingActions).toEqual([]);
  state.getState().logWear(['owned', 'owned', '']);
  expect(state.getState().wearLogs[0].itemIds).toEqual(['owned']);
  expect(state.getState().items[0].wearCount).toBe(1);
});


it.each([['Asia/Tashkent', '2026-10-16'], ['America/Los_Angeles', '2026-10-15'], ['Pacific/Kiritimati', '2026-10-16']])(
  'wear date and streak follow the local day near UTC midnight in %s', (TZ, expectedDay) => {
    const compile = (path: string) => transpileModule(readFileSync(require.resolve(path), 'utf8'), { compilerOptions: { module: ModuleKind.CommonJS } }).outputText;
    const calendar = compile('../../src/utils/calendarDate.ts');
    const helpers = compile('../../store/slices/helpers.ts');
    const code = `
      const NativeDate = Date;
      const instant = NativeDate.parse('2026-10-15T20:30:00Z');
      global.Date = class extends NativeDate {
        constructor(...args) { super(...(args.length ? args : [instant])); }
        static now() { return instant; }
      };
      const dates = {};
      new Function('exports', ${JSON.stringify(calendar)})(dates);
      const helpers = {};
      new Function('exports', 'require', ${JSON.stringify(helpers)})(helpers, () => dates);
      const today = helpers.getTodayDate();
      const yesterday = dates.localDateKey(dates.shiftCalendarDay(new Date(), -1));
      console.log(JSON.stringify([today, helpers.calculateStreak([{date: today}, {date: yesterday}])]));
    `;
    const result = execFileSync(process.execPath, ['-e', code], { env: { ...process.env, TZ }, encoding: 'utf8' });
    expect(JSON.parse(result)).toEqual([expectedDay, 2]);
  },
);
