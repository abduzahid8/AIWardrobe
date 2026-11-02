jest.mock('../../hooks/useNetworkRecovery', () => ({ useNetworkRecovery: jest.fn() }));
import { act, renderHook, waitFor } from '@testing-library/react-native';
import useAuthStore from '../../store/auth';
import { authEvents } from '../../store/authEvents';
import { useAdminGuard } from '../../hooks/useAdminGuard';
const mockRead = jest.fn();
const mockEq = jest.fn();
jest.mock('../../store/auth', () => ({ __esModule: true, default: require('zustand').create(() => ({ user: null, session: null, isAuthenticated: false })) }));
jest.mock('../../lib/supabase', () => ({ supabase: { from: () => ({ select: () => ({ eq: (...args: unknown[]) => {
  mockEq(...args); return { abortSignal: () => ({ maybeSingle: () => mockRead() }) };
} }) }) } }));
let owner = 0;
function login(email = 'ordinary@example.invalid') {
  const id = `admin-check-${++owner}`;
  authEvents.emitLogin(id);
  useAuthStore.setState({ user: { id, email, username: '' }, session: { user: { id } } as never, isAuthenticated: true });
  return id;
}
beforeEach(() => {
  jest.clearAllMocks();
  authEvents.emitLogout();
  useAuthStore.setState({ user: null, session: null, isAuthenticated: false });
  mockRead.mockResolvedValue({ data: { is_admin: false }, error: null });
});

test('a matching displayed admin email does not grant admin visibility', async () => {
  const id = login('info@aiwardrobe.club');
  const { result } = renderHook(useAdminGuard);
  await waitFor(() => expect(result.current.loading).toBe(false));
  expect(result.current.isAdmin).toBe(false);
  expect(mockEq).toHaveBeenCalledWith('id', id);
  expect(mockRead).toHaveBeenCalledTimes(1);
});

test('only a true server-owned flag grants visibility and manual refresh observes revocation', async () => {
  login(); mockRead.mockResolvedValueOnce({ data: { is_admin: true }, error: null });
  const { result } = renderHook(useAdminGuard);
  await waitFor(() => expect(result.current.isAdmin).toBe(true));
  await act(async () => { expect(await result.current.checkAdmin()).toBe(false); });
  expect(result.current.isAdmin).toBe(false);
});

test('a late old-account admin result cannot expose controls to the new account', async () => {
  login(); let finish!: (value: unknown) => void;
  mockRead.mockReturnValueOnce(new Promise(resolve => { finish = resolve; }));
  const { result } = renderHook(useAdminGuard);
  act(() => { login(); });
  expect(result.current.isAdmin).toBe(false);
  await waitFor(() => expect(result.current.loading).toBe(false));
  await act(async () => { finish({ data: { is_admin: true }, error: null }); await Promise.resolve(); });
  expect(result.current.isAdmin).toBe(false);
});

test('concurrent components share the same in-flight owner check', async () => {
  login(); let finish!: (value: unknown) => void;
  mockRead.mockReturnValueOnce(new Promise(resolve => { finish = resolve; }));
  const first = renderHook(useAdminGuard), second = renderHook(useAdminGuard);
  expect(mockRead).toHaveBeenCalledTimes(1);
  await act(async () => { finish({ data: { is_admin: true }, error: null }); });
  expect(first.result.current.isAdmin).toBe(true);
  expect(second.result.current.isAdmin).toBe(true);
});

test('database failure hides admin controls and ends loading', async () => {
  login(); mockRead.mockResolvedValueOnce({ data: null, error: { message: 'unavailable' } });
  const { result } = renderHook(useAdminGuard);
  await waitFor(() => expect(result.current.loading).toBe(false));
  expect(result.current.isAdmin).toBe(false);
});


test('reconnect retries a failed admin check after the shared request settles', async () => {
  login(); let finish!: (value:unknown)=>void;
  mockRead.mockReturnValueOnce(new Promise(resolve=>{finish=resolve;}))
    .mockResolvedValueOnce({data:{is_admin:true},error:null});
  const recovery=require('../../hooks/useNetworkRecovery').useNetworkRecovery as jest.Mock;
  const {result}=renderHook(useAdminGuard);
  act(()=>recovery.mock.calls[recovery.mock.calls.length-1][0]());
  await act(async()=>finish({data:null,error:{message:'network unavailable'}}));
  await waitFor(()=>expect(result.current.isAdmin).toBe(true));
  expect(mockRead).toHaveBeenCalledTimes(2);
});

test('reconnect does not retry an old admin check after account change', async () => {
  login(); let finish!: (value:unknown)=>void;
  mockRead.mockReturnValueOnce(new Promise(resolve=>{finish=resolve;}));
  const recovery=require('../../hooks/useNetworkRecovery').useNetworkRecovery as jest.Mock;
  const {result}=renderHook(useAdminGuard);
  act(()=>recovery.mock.calls[recovery.mock.calls.length-1][0]());
  act(()=>login());
  await act(async()=>finish({data:null,error:{message:'network unavailable'}}));
  await waitFor(()=>expect(result.current.loading).toBe(false));
  expect(result.current.isAdmin).toBe(false);
  expect(mockRead).toHaveBeenCalledTimes(2);
});
