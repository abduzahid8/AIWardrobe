import { revenueCatEnvironmentForUser } from '../../backend/supabase/functions/_shared/revenuecatSnapshot';
const tester='11111111-1111-4111-8111-111111111111',customer='22222222-2222-4222-8222-222222222222';
test('production remains the default without a server-designated tester',()=>expect(revenueCatEnvironmentForUser(customer,'PRODUCTION')).toBe('PRODUCTION'));
test('only an exact server-designated identity uses sandbox receipts',()=>{
 expect(revenueCatEnvironmentForUser(tester,'PRODUCTION',tester)).toBe('SANDBOX');
 expect(revenueCatEnvironmentForUser(customer,'PRODUCTION',tester)).toBe('PRODUCTION');
});
test.each(['*','anybody',`${tester},*`,'not-a-uuid'])('invalid sandbox configuration fails closed: %s',value=>expect(()=>revenueCatEnvironmentForUser(customer,'PRODUCTION',value)).toThrow());
test('standalone staging still supports its existing sandbox environment',()=>expect(revenueCatEnvironmentForUser(tester,'SANDBOX')).toBe('SANDBOX'));
