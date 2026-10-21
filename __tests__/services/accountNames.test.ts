import { AccountNameError, assertHandleAvailable, displayNameOf, profileNameUpdates } from '../../src/services/accountNames';
const mockRpc = jest.fn();
jest.mock('../../lib/supabase', () => ({ supabase: { rpc: (...args: any[]) => mockRpc(...args) } }));
beforeEach(() => { jest.clearAllMocks(); mockRpc.mockResolvedValue({data:true,error:null}); });
afterEach(() => jest.useRealTimers());
test('display name and unique handle are independent', () => {
    const user = {handle:'unique_handle',username:'Old Name',display_name:'Shared Name'};
    expect(displayNameOf(user)).toBe('Shared Name');
    expect(profileNameUpdates(user,'New Full Name','UNIQUE_HANDLE')).toEqual({display_name:'New Full Name'});
    expect(profileNameUpdates(user,'Shared Name',' new_handle ')).toEqual({handle:'new_handle'});
});
test.each(['', 'x'.repeat(201)])('invalid display name is rejected', name => expect(() => profileNameUpdates({handle:'unique'},name,'unique')).toThrow('invalid_display_name'));
test.each(['ab','bad handle','!', 'x'.repeat(31)])('invalid handle %s never reaches availability RPC', async handle => {
    await expect(assertHandleAvailable(handle)).rejects.toThrow('invalid_handle');expect(mockRpc).not.toHaveBeenCalled();
});
test('availability uses a canonical handle and rejects occupied values', async () => {
    mockRpc.mockResolvedValue({data:false,error:null});await expect(assertHandleAvailable(' Existing ')).rejects.toThrow('username_exists');
    expect(mockRpc).toHaveBeenCalledWith('is_username_available',{check_username:'existing'});
});
test.each([{data:null,error:null},{data:true,error:{message:'unavailable'}}])('availability errors fail closed',async result=>{
    mockRpc.mockResolvedValue(result);await expect(assertHandleAvailable('unique')).rejects.toThrow('availability_unavailable');
});
test('an unresponsive availability call has a bounded deadline',async()=>{
    jest.useFakeTimers();mockRpc.mockReturnValue(new Promise(()=>{}));const pending=expect(assertHandleAvailable('unique')).rejects.toBeInstanceOf(AccountNameError);
    await jest.advanceTimersByTimeAsync(15000);await pending;
});
