/** @jest-environment node */
const fs=require('fs'),ts=require('typescript');
const source=fs.readFileSync(require.resolve('../../screens/ProfileScreen.tsx'),'utf8');
const ast=ts.createSourceFile('ProfileScreen.tsx',source,ts.ScriptTarget.Latest,true,ts.ScriptKind.TSX);
let loader,focus,visible;
function visit(node){
 if(ts.isVariableDeclaration(node)&&node.name.getText(ast)==='loadTripPlans')loader=node.initializer.arguments[0].getText(ast);
 if(ts.isVariableDeclaration(node)&&node.name.getText(ast)==='tripPlanDays')visible=node.initializer.getText(ast);
 if(ts.isCallExpression(node)&&node.expression.getText(ast)==='useFocusEffect'&&node.arguments[0]?.getText(ast).includes('loadTripPlans'))focus=node.arguments[0].arguments[0].getText(ast);
 ts.forEachChild(node,visit);
}visit(ast);
const build=(expression,scope)=>Function(...Object.keys(scope),ts.transpile(`const value=${expression};`,{target:ts.ScriptTarget.ES2020})+';return value;')(...Object.values(scope));
function deferred(){let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return{promise,resolve,reject};}
function setup(){
 let id='owner',revision=1;
 const state={user:{id},tripPlanRequest:{current:0},authEvents:{getUserId:()=>id,getRevision:()=>revision},readCalendarLogs:jest.fn(),setTripPlanOwner:jest.fn(),setTripPlanErrorOwner:jest.fn(),setTripPlanDays:jest.fn(),setTripPlanRecency:jest.fn(),toTripPlanDay:day=>day};
 return{state,load:build(loader,state),changeOwner(){id='other';revision++;state.user.id=id;}};
}
const days={a:{date:'2099-01-01',items:[{id:'coat'}]},b:{date:'2099-01-02',items:[{id:'shoes'}]}};
it('loads the latest calendar plan on each focus even within the former one-minute TTL',async()=>{
 const load=jest.fn(),scheduled=[];
 const effect=build(focus,{loadTripPlans:load,setTimeout:callback=>{scheduled.push(callback);return scheduled.length;},clearTimeout:jest.fn()});
 const stop=effect();scheduled.shift()();stop();const stop2=effect();scheduled.shift()();
 expect(load).toHaveBeenCalledTimes(2);expect(load.mock.calls[0][0]()).toBe(false);expect(load.mock.calls[1][0]()).toBe(true);stop2();
});
it('publishes upcoming days only for the captured current owner',async()=>{
 const app=setup();app.state.readCalendarLogs.mockResolvedValue(days);await app.load(()=>true);
 expect(app.state.setTripPlanOwner).toHaveBeenCalledWith('owner');expect(app.state.setTripPlanDays).toHaveBeenCalledWith(Object.values(days));
});
it('an account change during reading cannot publish the previous account plan',async()=>{
 const app=setup(),read=deferred();app.state.readCalendarLogs.mockReturnValue(read.promise);const pending=app.load(()=>true);
 app.changeOwner();read.resolve(days);await pending;expect(app.state.setTripPlanDays).not.toHaveBeenCalled();expect(app.state.setTripPlanOwner).not.toHaveBeenCalled();
});
it('a late old-account failure cannot clear a new account plan',async()=>{
 const app=setup(),old=deferred();app.state.readCalendarLogs.mockReturnValueOnce(old.promise).mockResolvedValueOnce(days);
 const pending=app.load(()=>true);app.changeOwner();await app.load(()=>true);old.reject(Error('old cache'));await pending;
 expect(app.state.setTripPlanDays).toHaveBeenCalledTimes(1);expect(app.state.setTripPlanOwner).toHaveBeenCalledWith('other');
});
it('blur or unmount while reading prevents publishing data',async()=>{
 const app=setup(),read=deferred();let active=true;app.state.readCalendarLogs.mockReturnValue(read.promise);
 const pending=app.load(()=>active);active=false;read.resolve(days);await pending;expect(app.state.setTripPlanDays).not.toHaveBeenCalled();
});
it('an older read cannot replace a newer read for the same owner',async()=>{
 const app=setup(),old=deferred();app.state.readCalendarLogs.mockReturnValueOnce(old.promise).mockResolvedValueOnce({b:days.b});
 const pending=app.load(()=>true);await app.load(()=>true);old.resolve(days);await pending;
 expect(app.state.setTripPlanDays).toHaveBeenCalledTimes(1);expect(app.state.setTripPlanDays).toHaveBeenCalledWith([days.b]);
});
it('hides stored plan days immediately when render ownership no longer matches',()=>{
 const empty=[],stored=Object.values(days);const state={tripPlanOwner:'owner',user:{id:'other'},authEvents:{getUserId:()=> 'other'},storedTripPlanDays:stored,EMPTY_TRIP_PLAN_DAYS:empty};
 expect(build(visible,state)).toBe(empty);state.user.id='owner';expect(build(visible,state)).toBe(empty);
 state.authEvents.getUserId=()=> 'owner';expect(build(visible,state)).toBe(stored);
});

it('read failure retains already loaded days and a successful retry clears the error',async()=>{
 const app=setup();app.state.readCalendarLogs.mockResolvedValueOnce(days).mockRejectedValueOnce(Error('unreadable')).mockResolvedValueOnce({});
 await app.load(()=>true);await app.load(()=>true);
 expect(app.state.setTripPlanDays).toHaveBeenCalledTimes(1);expect(app.state.setTripPlanErrorOwner).toHaveBeenLastCalledWith('owner');
 await app.load(()=>true);expect(app.state.setTripPlanDays).toHaveBeenLastCalledWith([]);expect(app.state.setTripPlanErrorOwner).toHaveBeenLastCalledWith(null);
});
