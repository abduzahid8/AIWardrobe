import {serve} from 'https://deno.land/std@0.168.0/http/server.ts';
import {createClient} from 'https://esm.sh/@supabase/supabase-js@2';
import {enforceRateLimit} from '../_shared/rateLimit.ts';
import {currentBudgetRates} from '../_shared/budgetFxCache.ts';
const headers={'Access-Control-Allow-Origin':'*','Access-Control-Allow-Headers':'authorization, x-client-info, apikey, content-type','Content-Type':'application/json'};
serve(async req=>{
 if(req.method==='OPTIONS')return new Response('ok',{headers});
 if(!['GET','POST'].includes(req.method))return new Response(JSON.stringify({error:'method_not_allowed'}),{status:405,headers});
 const denied=await enforceRateLimit(req,{feature:'budget-rates',limit:20,meterPaidWork:false,headers});
 if(denied)return denied;
 try{
  const admin=createClient(Deno.env.get('SUPABASE_URL')!,Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!,{auth:{persistSession:false,autoRefreshToken:false}});
  const snapshot=await currentBudgetRates(admin);
  return new Response(JSON.stringify({success:true,snapshot}),{headers:{...headers,'Cache-Control':'private, max-age=300'}});
 }catch{return new Response(JSON.stringify({success:false,error:'budget_rates_unavailable'}),{status:503,headers});}
});
