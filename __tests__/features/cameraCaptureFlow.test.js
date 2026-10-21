/** @jest-environment node */
const fs = require('fs');
const ts = require('typescript');
const source = fs.readFileSync(process.env.CAMERA_SCREEN_SOURCE || require.resolve('../../screens/CameraScreen.tsx'), 'utf8');
const ast = ts.createSourceFile('CameraScreen.tsx', source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
function initializer(name) {
  let result;
  function visit(node) {
    if (ts.isVariableDeclaration(node) && node.name.getText(ast) === name) result = node.initializer.getText(ast);
    ts.forEachChild(node, visit);
  }
  visit(ast);
  if (!result) throw new Error(`Missing production handler: ${name}`);
  return ts.transpile(`const handler = ${result};`, { target: ts.ScriptTarget.ES2020 });
}
function harness(kind, { gallery = false, saveFails = false, microphone = true, uri = 'file:///capture.jpg' } = {}) {
  const navigation = { navigate: jest.fn() };
  const MediaLibrary = { saveToLibraryAsync: jest.fn(async () => { if (saveFails) throw new Error('disk full'); }) };
  const setIsRecording = jest.fn();
  const requestMicPermission = jest.fn(async () => ({ granted: microphone }));
  const cameraRef = { current: { takePictureAsync: jest.fn(async () => ({ uri })), recordAsync: jest.fn(async () => ({ uri })) } };
  const dependencies = { navigation, MediaLibrary, setIsRecording, requestMicPermission, cameraRef,
    mediaPermission: { granted: gallery }, micPermission: { granted: microphone },
    Haptics: { impactAsync: jest.fn(), ImpactFeedbackStyle: { Medium: 1, Heavy: 2 } },
    Alert: { alert: jest.fn() }, t: x => x, isRecording: false, recordScale: { value: 1 }, withSpring: x => x };
  const handler = Function(...Object.keys(dependencies), `${initializer(kind === 'photo' ? 'takePhoto' : 'startRecording')} return handler;`)(...Object.values(dependencies));
  return { handler, navigation, MediaLibrary, setIsRecording, requestMicPermission, cameraRef };
}
for (const kind of ['photo', 'video']) {
  test.each([
    { gallery: false, saveFails: false },
    { gallery: true, saveFails: false },
    { gallery: true, saveFails: true },
  ])(`${kind} reaches analysis with gallery setting %j`, async options => {
    const app = harness(kind, options);
    await app.handler();
    expect(app.navigation.navigate).toHaveBeenCalledTimes(1);
    expect(app.navigation.navigate).toHaveBeenCalledWith('WardrobeVideo', { [kind === 'photo' ? 'imageUri' : 'videoUri']: 'file:///capture.jpg' });
    expect(app.MediaLibrary.saveToLibraryAsync).toHaveBeenCalledTimes(options.gallery ? 1 : 0);
    if (kind === 'video') expect(app.setIsRecording).toHaveBeenLastCalledWith(false);
  });
  test(`${kind} with no captured file does not open analysis`, async () => {
    const app = harness(kind, { uri: undefined });
    // Explicitly replace the capture result because the harness has a default URI.
    app.cameraRef.current[kind === 'photo' ? 'takePictureAsync' : 'recordAsync'].mockResolvedValue({});
    await app.handler();
    expect(app.navigation.navigate).not.toHaveBeenCalled();
  });
}
test('microphone denial stops video capture and resets recording state', async () => {
  const app = harness('video', { microphone: false });
  await app.handler();
  expect(app.requestMicPermission).toHaveBeenCalledTimes(1);
  expect(app.cameraRef.current.recordAsync).not.toHaveBeenCalled();
  expect(app.navigation.navigate).not.toHaveBeenCalled();
  expect(app.setIsRecording).toHaveBeenLastCalledWith(false);
});

for (const count of [0, 2, 3]) {
  test(`post-paywall activation with ${count} garments opens the current feature`, () => {
    const text = fs.readFileSync(process.env.PAYWALL_SCREEN_SOURCE || require.resolve('../../screens/PaywallScreen.tsx'), 'utf8');
    const file = ts.createSourceFile('PaywallScreen.tsx', text, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
    let expression;
    function visit(node) {
      if (ts.isVariableDeclaration(node) && node.name.getText(file) === 'resetToActivation') expression = node.initializer.getText(file);
      ts.forEachChild(node, visit);
    }
    visit(file);
    const navigation = { reset: jest.fn() };
    const compiled = ts.transpile(`const handler = ${expression};`, { target: ts.ScriptTarget.ES2020 });
    Function('useCallback', 'wardrobeItemCount', 'navigationRef', 'navigation', 'setTimeout', `${compiled} handler();`)(
      callback => callback, count, { isReady: () => true }, navigation, callback => callback(),
    );
    expect(navigation.reset).toHaveBeenCalledWith({ index: 1, routes: [
      { name: 'Main' }, count >= 3 ? { name: 'AIOutfit', params: { source: 'wardrobe' } } : { name: 'Camera' },
    ] });
  });
}
