import {selectRevenueCatKey} from '../../src/config/revenueCatKey';

test('each store receives its own explicit key when both are configured',()=>{
 const keys={ios:'appl_fixture',android:'goog_fixture',legacy:'appl_legacy'};
 expect(selectRevenueCatKey('ios',keys)).toBe('appl_fixture');
 expect(selectRevenueCatKey('android',keys)).toBe('goog_fixture');
});
test('legacy single-key builds work only on the matching platform',()=>{
 expect(selectRevenueCatKey('ios',{legacy:'appl_fixture'})).toBe('appl_fixture');
 expect(selectRevenueCatKey('android',{legacy:'appl_fixture'})).toBe('');
 expect(selectRevenueCatKey('android',{legacy:'goog_fixture'})).toBe('goog_fixture');
 expect(selectRevenueCatKey('ios',{legacy:'goog_fixture'})).toBe('');
});
test('an invalid explicit store key is not silently replaced by a legacy key',()=>{
 expect(selectRevenueCatKey('android',{android:'appl_wrong',legacy:'goog_valid'})).toBe('');
 expect(selectRevenueCatKey('ios',{ios:'goog_wrong',legacy:'appl_valid'})).toBe('');
});
test('missing, unknown and unsupported platforms have no native SDK key',()=>{
 expect(selectRevenueCatKey('android',{})).toBe('');
 expect(selectRevenueCatKey('ios',{ios:'unrecognized'})).toBe('');
 expect(selectRevenueCatKey('web',{legacy:'appl_fixture'})).toBe('');
});
test('explicit RevenueCat test-store keys remain selectable for SDK testing',()=>{
 expect(selectRevenueCatKey('ios',{ios:'test_fixture'})).toBe('test_fixture');
 expect(selectRevenueCatKey('android',{android:'test_fixture'})).toBe('test_fixture');
});
