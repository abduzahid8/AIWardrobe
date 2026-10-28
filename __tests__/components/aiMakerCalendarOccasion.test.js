/** @jest-environment node */
const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');
const calendar = require('../../features/calendar/types');
const source = ts.createSourceFile('AIOutfitmaker.tsx', fs.readFileSync(path.resolve(__dirname, '../../screens/AIOutfitmaker.tsx'), 'utf8'), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
function initializer(name) {
  let found;
  function visit(node) {
    if (ts.isVariableDeclaration(node) && node.name.getText(source) === name) found = node.initializer;
    else ts.forEachChild(node, visit);
  }
  visit(source); if (!found) throw Error(`Missing shipped callback: ${name}`); return found;
}
function callback(name, bindings) {
  const js = ts.transpileModule(`const handler=(${initializer(name).getText(source)});`, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText;
  return new Function(...Object.keys(bindings), `${js}\nreturn handler;`)(...Object.values(bindings));
}
test.each(['work','casual','date','formal','sport','travel'])('the shipped calendar save preserves selected %s instead of deriving it from Classic style', async selectedOccasion => {
  const selectedStyle = 'classic', logs = {}, alert = jest.fn();
  const getCalendarOccasion = callback('getCalendarOccasion', { selectedOccasion, resolveCalendarOccasion: calendar.resolveCalendarOccasion });
  const bindings = {
    selectedStyle, selectedOccasion, getCalendarOccasion,
    authEvents: { getRevision: () => 7 }, createOutfitLog: calendar.createOutfitLog,
    readCalendarLogs: async () => ({}), updateCalendarLogs: async (mutate, revision) => { expect(revision).toBe(7); mutate(logs); },
    Haptics: { notificationAsync: jest.fn(), NotificationFeedbackType: { Success: 'success' } },
    setCalendarVisible: jest.fn(), setCalendarOutfit: jest.fn(), formatPlanningDate: () => 'October 26',
    planningDateKey: undefined, navigation: { navigate: jest.fn() }, Alert: { alert }, t: key => key,
  };
  const add = callback('addToCalendar', bindings);
  const item = { id: 'catalog-top', type: 'top', imageUrl: 'https://images.example/top.jpg', name: 'Catalog shirt' };
  await add({ items: [item] }, new Date(2026,9,26,12));
  expect(logs['2026-10-26']).toMatchObject({ date: '2026-10-26', occasion: selectedOccasion,
    items: [{ id: item.id, type: item.type, image: item.imageUrl, name: item.name }] });
  expect(alert).toHaveBeenCalledWith('outfitMaker.savedToCalendar','outfitMaker.savedToCalendarSubtitle',expect.any(Array));
});
