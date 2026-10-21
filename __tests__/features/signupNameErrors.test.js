/** @jest-environment node */
const fs=require('fs'),ts=require('typescript');
jest.mock('../../lib/supabase',()=>({supabase:{}}));
const {AccountNameError}=require('../../src/services/accountNames');
const {SUPABASE_AUTH_ERRORS}=require('../../constants/authErrors');
const ast=ts.createSourceFile('signup.tsx',fs.readFileSync('screens/SignUpScreen.tsx','utf8'),ts.ScriptTarget.Latest,true,ts.ScriptKind.TSX);
let handler;function walk(node){if(ts.isVariableDeclaration(node)&&node.name.getText(ast)==='handleSignUp')handler=node.initializer.getText(ast);ts.forEachChild(node,walk);}walk(ast);
const body=ts.transpileModule(`const execute=${handler};`,{compilerOptions:{target:ts.ScriptTarget.ES2022}}).outputText+';return execute;';
test.each([['username_exists','signUp.usernameTaken','signUp.usernameAlreadyUse'],['invalid_handle','signUp.invalidUsername','signUp.usernameRequirements'],['availability_unavailable','signUp.connectionFailed','signUp.cannotConnectServer']])('actual signup handler translates %s',async(code,title,message)=>{
 const state={authRequestInFlight:{current:false},email:'new@example.invalid',password:'Synthetic123!',username:'test_handle',isEmailValid:true,isPasswordValid:true,isUsernameValid:true,normalizedEmail:'new@example.invalid',trimmedUsername:'test_handle',setIsLoading:jest.fn(),register:jest.fn().mockRejectedValue(new AccountNameError(code)),t:key=>key,Alert:{alert:jest.fn()},logger:{error:jest.fn()},SUPABASE_AUTH_ERRORS,AccountNameError,setSentEmail:jest.fn(),setIsSent:jest.fn()};
 await Function(...Object.keys(state),body)(...Object.values(state))();
 expect(state.Alert.alert).toHaveBeenCalledWith(title,message);expect(state.authRequestInFlight.current).toBe(false);expect(state.setIsLoading).toHaveBeenLastCalledWith(false);
});
