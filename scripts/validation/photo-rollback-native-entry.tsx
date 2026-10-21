/** Explicit audit entry: actual item slice + native staging/rollback, synthetic owner. */
import React, { useEffect, useState } from 'react';
import { Text, Platform } from 'react-native';
import { registerRootComponent } from 'expo';
import { Asset } from 'expo-asset';
import * as FileSystem from 'expo-file-system/legacy';
import { createStore } from 'zustand/vanilla';
import type { WardrobeState } from '../../store/wardrobeStore';
const owner = '11111111-1111-4111-8111-111111111111';
let started = false;
function Audit() {
 const [status,setStatus] = useState('Native photo rollback audit running');
 useEffect(() => { if(started)return;started=true;void(async()=>{
  let fileReads=0,networkAttempts=0;
  const report:{platform:string;passed:boolean;checks:string[];fileReads:number;networkAttempts:number;failure?:string}={platform:Platform.OS,passed:false,checks:[],fileReads,networkAttempts};
  const check=(value:boolean,message:string)=>{if(!value)throw new Error(message);report.checks.push(message);};
  const originalFetch=globalThis.fetch;
  const directory=FileSystem.documentDirectory+'aiwardrobe-pending-media/'+owner+'/';
  const picker=FileSystem.cacheDirectory+'photo-rollback-fixture.jpg';
  globalThis.fetch=async(input,init)=>{
   const uri=typeof input==='string'?input:input instanceof Request?input.url:String(input);
   if(!uri.startsWith(directory)&&!uri.startsWith(picker)){networkAttempts++;throw new Error('Audit network disabled');}
   fileReads++;return originalFetch(input,init);
  };
  let restoreThumbnail:(()=>void)|undefined;
  try{
   const media=require('../../src/services/pendingWardrobeMedia') as typeof import('../../src/services/pendingWardrobeMedia');
   const thumbnails=require('../../src/services/wardrobeThumbnail') as typeof import('../../src/services/wardrobeThumbnail');
   const auth=require('../../store/auth').default as typeof import('../../store/auth').default;
   const subscription=require('../../store/subscriptionStore') as typeof import('../../store/subscriptionStore');
   const {authEvents}=require('../../store/authEvents') as typeof import('../../store/authEvents');
   const {createItemsSlice}=require('../../store/slices/wardrobeItemsSlice') as typeof import('../../store/slices/wardrobeItemsSlice');
   auth.setState({user:{id:owner} as never});authEvents.emitLogin(owner);subscription.default.setState({tier:'premium'});
   const asset=Asset.fromModule(require('./fixtures/opaque-photo/photo-2.jpg'));await asset.downloadAsync();
   await FileSystem.writeAsStringAsync(picker,await FileSystem.readAsStringAsync(asset.localUri||asset.uri,{encoding:FileSystem.EncodingType.Base64}),{encoding:FileSystem.EncodingType.Base64});
   const fixture=()=>createStore<WardrobeState>()((set,get,store)=>({...createItemsSlice(set,get,store),pendingActions:[],outfits:[],syncToServer:async()=>undefined} as unknown as WardrobeState));
   const input={category:'top',imageUrl:picker} as never;
   const originalThumbnail=thumbnails.createNativeWardrobeThumbnail;restoreThumbnail=()=>{thumbnails.createNativeWardrobeThumbnail=originalThumbnail;};
   let stagedOriginal='';
   thumbnails.createNativeWardrobeThumbnail=async(_owner,_id,original)=>{
    stagedOriginal=original;check((await FileSystem.getInfoAsync(original)).exists,'original exists before preview rejection');throw new Error('synthetic preview rejection');
   };
   const rejected=fixture();let previewFailed=false;
   try{await rejected.getState().addItem(input);}catch(error){previewFailed=error instanceof Error&&error.message==='synthetic preview rejection';}
   check(previewFailed,'preview rejection is reported');
   check(!(await FileSystem.getInfoAsync(stagedOriginal)).exists,'rejected preview rolls back native original file');
   check(rejected.getState().items.length===0&&rejected.getState().pendingActions.length===0,'rejected preview publishes no item or action');
   const supplied=fixture();let suppliedFailed=false;
   try{await supplied.getState().addItem({category:'top',imageUrl:picker,thumbnailUrl:FileSystem.cacheDirectory+'missing-rollback-preview.png'} as never);}catch{suppliedFailed=true;}
   check(suppliedFailed,'missing supplied preview rejects save');
   check((await FileSystem.readDirectoryAsync(directory)).length===0,'supplied preview rejection leaves no managed photo files');
   let savedOriginal='',savedThumbnail='';
   thumbnails.createNativeWardrobeThumbnail=async(user,id,original)=>{savedOriginal=original;savedThumbnail=await media.stageNativeWardrobeImage(user,id,original);return savedThumbnail;};
   const queued=fixture();queued.setState({syncToServer:async()=>{throw new Error('synthetic offline sync');}});let offline=false;
   try{await queued.getState().addItem(input);}catch(error){offline=error instanceof Error&&error.message==='synthetic offline sync';}
   check(offline,'sync rejection is reported after publication');
   check(queued.getState().items.length===1&&queued.getState().pendingActions.length===1,'offline item and upload action remain queued');
   check((await FileSystem.getInfoAsync(savedOriginal)).exists&&(await FileSystem.getInfoAsync(savedThumbnail)).exists,'published offline save retains both native files for retry');
   await media.clearPendingWardrobeImages(owner);
   subscription.default.setState({tier:'free'});
   const cap=subscription.TIER_FEATURES.free.wardrobeItems;check(Number.isInteger(cap)&&cap>0,'free tier has a finite wardrobe cap for the race check');
   const full=fixture();let rejectedOriginal='',rejectedThumbnail='';
   thumbnails.createNativeWardrobeThumbnail=async(user,id,original)=>{
    rejectedOriginal=original;rejectedThumbnail=await media.stageNativeWardrobeImage(user,id,original);
    full.setState({items:Array.from({length:cap},(_,index)=>({id:'existing-'+index,imageUrl:'https://catalog.test/retained.jpg'} as never))});return rejectedThumbnail;
   };
   let limit=false;try{await full.getState().addItem(input);}catch(error){limit=Boolean(error&&typeof error==='object'&&'code' in error&&error.code==='WARDROBE_LIMIT_REACHED');}
   check(limit,'limit reached during staging rejects the unpublished save');
   check(!(await FileSystem.getInfoAsync(rejectedOriginal)).exists&&!(await FileSystem.getInfoAsync(rejectedThumbnail)).exists,'post-staging limit rejection deletes both native files');
   check(full.getState().items.length===cap&&full.getState().pendingActions.length===0,'rollback preserves the items that filled the wardrobe');
   check(networkAttempts===0,'no external fetch requests attempted');report.passed=true;
  }catch(error){report.failure=error instanceof Error?error.message:'Unknown failure';}
  finally{
   restoreThumbnail?.();
   await FileSystem.deleteAsync(directory,{idempotent:true});await FileSystem.deleteAsync(picker,{idempotent:true});
   globalThis.fetch=originalFetch;report.fileReads=fileReads;report.networkAttempts=networkAttempts;
   await FileSystem.writeAsStringAsync(FileSystem.documentDirectory+'photo-rollback-native.json',JSON.stringify(report));
  }
  setStatus(report.passed?'Native photo rollback passed':'Native photo rollback failed');
 })();},[]);
 return <Text style={{padding:40}}>{status}</Text>;
}
registerRootComponent(Audit);
