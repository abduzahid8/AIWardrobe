import {nextMediaImageLifecycle as next} from '../../src/utils/mediaImageLifecycle';
const identity={userId:'owner',accessToken:'private-token'};
test('equivalent headers and URI keep the same lifecycle despite new objects',()=>{
 const first=next(undefined,{uri:'https://image.test/a',headers:{A:'1',B:'2'}},identity);
 expect(next(first,{uri:'https://image.test/a',headers:{B:'2',A:'1'}},identity)).toBe(first);
});
test('changed request headers restart the same URL',()=>{
 const first=next(undefined,{uri:'https://image.test/a',headers:{A:'1'}},identity);
 expect(next(first,{uri:'https://image.test/a',headers:{A:'2'}},identity).version).toBe(1);
});
test('candidate arrays and static assets have stable semantic identity',()=>{
 const first=next(undefined,[1,{uri:'a'}],identity);expect(next(first,[1,{uri:'a'}],identity)).toBe(first);
 expect(next(first,[1,{uri:'b'}],identity).version).toBe(1);
});
test('caller recycling key and auth changes reset without putting secrets in the view key',()=>{
 const first=next(undefined,{uri:'a'},identity,'one');
 expect(next(first,{uri:'a'},identity,'two').version).toBe(1);
 const refreshed=next(first,{uri:'a'},{...identity,accessToken:'renewed-private-token'},'one');
 expect(refreshed.version).toBe(1);expect(String(refreshed.version)).not.toContain('private');
});
test('large inline images do not become large native keys',()=>{
 const source={uri:'data:image/png;base64,'+'A'.repeat(4000000)};
 const first=next(undefined,source,identity);expect(String(first.version).length).toBe(1);
 expect(next(first,{...source},identity)).toBe(first);
});
