/** Dedicated native test entry; never imported by the production app entry. */
import React,{useEffect,useState,useRef} from 'react';
import {View,Text,Button} from 'react-native';
import {registerRootComponent} from 'expo';
import * as FileSystem from 'expo-file-system/legacy';
import {Image,ExpoImage} from '../../src/components/MediaImage';
import useAuthStore from '../../store/auth';
import {mediaImageSource} from '../../src/utils/mediaImageSource';
import {wardrobeMediaReference} from '../../src/services/wardrobeMedia';
import {authEvents} from '../../store/authEvents';
type Fixture={owner:string;token:string;uri:string;refreshUri?:string;refreshedToken?:string;expiresAt:number;autoRun?:boolean};
function Audit(){
 const [fixture,setFixture]=useState<Fixture|null>(null),[phase,setPhase]=useState('starting');
 const [native,setNative]=useState('pending'),[expo,setExpo]=useState('pending');
 const audit=useRef<{done:boolean;phases:unknown[];diagnostics:Record<string,unknown>}>({done:false,phases:[],diagnostics:{}});
 const classify=(error:unknown)=>{const s=String(error);return /401|403/.test(s)?'authorization':/SSL|certificate|CertPath/i.test(s)?'tls':/resolve|DNS|UnknownHost/i.test(s)?'dns':/timeout/i.test(s)?'timeout':'unclassified';};
 const finish=(passed:boolean,reason?:string)=>{
  if(audit.current.done)return;audit.current.done=true;
  const result=JSON.stringify({scope:'Native component callback instrumentation; no pixel assertion',passed,reason,phases:audit.current.phases,diagnostics:audit.current.diagnostics});
  console.warn('MEDIA_AUDIT_RESULT '+result);
  if(FileSystem.documentDirectory)FileSystem.writeAsStringAsync(`${FileSystem.documentDirectory}media-audit-result.json`,result).catch(()=>{});
 };
 const identity=(next:string,f:Fixture)=>{
  setNative('pending');setExpo('pending');setPhase(next);
  if(next==='signed out'){useAuthStore.setState({session:null,user:null});authEvents.emitLogout();}
  else{const id=next==='different identity'?'22222222-2222-4222-8222-222222222222':f.owner;
   const token=next==='owner invalid token'?'invalid-fixture-token':next==='owner refreshed token'?f.refreshedToken:f.token;
   useAuthStore.setState({session:{access_token:token,user:{id}} as any});authEvents.emitLogin(id);}
 };
 useEffect(()=>{(async()=>{
  try{const f=JSON.parse(await FileSystem.readAsStringAsync(`${FileSystem.bundleDirectory}media-audit.json`));
   if(Date.now()>f.expiresAt)throw Error('expired');setFixture(f);identity('owner',f);
   if(f.autoRun){
    const source=mediaImageSource({uri:f.uri},{userId:f.owner,accessToken:f.token}) as {headers?:Record<string,string>}|undefined;
    audit.current.diagnostics.source={privateReferenceDetected:!!wardrobeMediaReference(f.uri),authorizationAttached:source?.headers?.Authorization===`Bearer ${f.token}`};
    (async()=>{const c=new AbortController(),timer=setTimeout(()=>c.abort(),15000);try{
     const r=await fetch(f.uri,{headers:{Authorization:`Bearer ${f.token}`},signal:c.signal});const body=await r.arrayBuffer();audit.current.diagnostics.fetch={status:r.status,bytes:body.byteLength};
    }catch(e){audit.current.diagnostics.fetch={failed:true,category:classify(e)};}finally{clearTimeout(timer);}})();
   }
   if(f.autoRun&&FileSystem.documentDirectory)await FileSystem.writeAsStringAsync(`${FileSystem.documentDirectory}media-audit-status.json`,JSON.stringify({phase:'fixture_ready'}));
  }catch{setPhase('fixture unavailable');
   if(FileSystem.documentDirectory)FileSystem.writeAsStringAsync(`${FileSystem.documentDirectory}media-audit-status.json`,JSON.stringify({phase:'fixture_unavailable'})).catch(()=>{});}
 })();},[]);
 useEffect(()=>{
  if(!fixture?.autoRun)return;
  const timer=setTimeout(()=>finish(false,'native_callback_timeout'),120000);
  return ()=>clearTimeout(timer);
 },[fixture]);
 useEffect(()=>{
  if(!fixture?.autoRun||audit.current.done)return;
  audit.current.diagnostics.lastState={phase,native,expo};
  const loaded=native.startsWith('loaded')&&expo.startsWith('loaded');
  if(phase==='owner'&&loaded){audit.current.phases.push({phase,native,expo});identity('owner invalid token',fixture);}
  else if(phase==='owner invalid token'&&native==='failed'&&expo==='failed'){audit.current.phases.push({phase,native,expo});identity('owner refreshed token',fixture);}
  else if(phase==='owner refreshed token'&&loaded){audit.current.phases.push({phase,native,expo});finish(true);}
 },[fixture,phase,native,expo]);
 const uri=phase.includes('token')?fixture?.refreshUri:fixture?.uri;
 return <View onLayout={event=>{audit.current.diagnostics.layout=event.nativeEvent.layout;}} style={{flex:1,padding:24,paddingTop:80,backgroundColor:'white'}}>
  <Text accessibilityRole="header" style={{fontSize:22,fontWeight:'bold'}}>Private image native audit</Text>
  <Text style={{marginVertical:8}}>Identity: {phase}</Text>
  <Text>Native: {native}</Text><Text>Expo: {expo}</Text>
  <View style={{height:205,marginTop:10,backgroundColor:'#e5e7eb'}}>
   {fixture&&<Image source={{uri}} style={{width:'100%',height:205}} resizeMode="contain" onLoad={event=>setNative(`loaded ${event.nativeEvent.source.width} x ${event.nativeEvent.source.height}`)} onError={event=>{audit.current.diagnostics.nativeError=classify(event.nativeEvent.error);setNative('failed');}}/>}
  </View>
  <View style={{height:205,marginTop:10,backgroundColor:'#e5e7eb'}}>
   {fixture&&<ExpoImage source={{uri}} style={{width:'100%',height:205}} contentFit="contain" onLoad={event=>setExpo(`loaded ${event.source.width} x ${event.source.height}`)} onError={event=>{audit.current.diagnostics.expoError=classify(event.error);setExpo('failed');}}/>}
  </View>
  <Button title="Switch to different identity" onPress={()=>fixture&&identity('different identity',fixture)}/>
  <Button title="Sign out" onPress={()=>fixture&&identity('signed out',fixture)}/>
  <Button title="Use invalid token" onPress={()=>fixture?.refreshUri&&identity('owner invalid token',fixture)}/>
  <Button title="Apply refreshed token" onPress={()=>fixture?.refreshedToken&&identity('owner refreshed token',fixture)}/>
  <Button title="Restore owner" onPress={()=>fixture&&identity('owner',fixture)}/>
 </View>;
}
registerRootComponent(Audit);
