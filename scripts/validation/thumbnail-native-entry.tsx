/** Standalone native codec audit entry. Build only with explicit ENTRY_FILE override. */
import React,{useEffect,useState} from 'react';
import {View,Text,ScrollView,Platform} from 'react-native';
import {registerRootComponent} from 'expo';
import {Asset} from 'expo-asset';
import * as FileSystem from 'expo-file-system/legacy';
import {decode} from 'base64-arraybuffer';
import {ImageManipulator,SaveFormat} from 'expo-image-manipulator';
import {Image} from '../../src/components/MediaImage';
import {authEvents} from '../../store/authEvents';
import {stageNativeWardrobeImage} from '../../src/services/pendingWardrobeMedia';
import {createNativeWardrobeThumbnail} from '../../src/services/wardrobeThumbnail';
const owner='11111111-1111-4111-8111-111111111111';
const cases=[
 {name:'Repository photo',asset:require('../../assets/images/basic_outfit_1.png'),originalId:'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',thumbnailId:'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb',alpha:false},
 {name:'Transparent app icon',asset:require('../../assets/icon.png'),originalId:'cccccccc-cccc-4ccc-8ccc-cccccccccccc',thumbnailId:'dddddddd-dddd-4ddd-8ddd-dddddddddddd',alpha:true},
 {name:'Opaque photo 1',asset:require('./fixtures/opaque-photo/photo-1.jpg'),originalId:'dddddddd-aaaa-4aaa-8aaa-aaaaaaaaaaaa',thumbnailId:'eeeeeeee-aaaa-4aaa-8aaa-aaaaaaaaaaaa',alpha:false},
 {name:'Opaque photo 2',asset:require('./fixtures/opaque-photo/photo-2.jpg'),originalId:'dddddddd-bbbb-4bbb-8bbb-bbbbbbbbbbbb',thumbnailId:'eeeeeeee-bbbb-4bbb-8bbb-bbbbbbbbbbbb',alpha:false},
 {name:'EXIF orientation 1',asset:require('./fixtures/photo-orientation/orientation-1.jpg'),originalId:'9abb2726-92d7-5914-924d-835afa21221c',thumbnailId:'92c7812c-9674-54a5-b91f-7a63344e342e',alpha:false},
 {name:'EXIF orientation 2',asset:require('./fixtures/photo-orientation/orientation-2.jpg'),originalId:'f769d021-b9cf-5ef7-95f3-edf266c8bff5',thumbnailId:'d83dae59-4a15-58c8-b47d-99155d0223bd',alpha:false},
 {name:'EXIF orientation 3',asset:require('./fixtures/photo-orientation/orientation-3.jpg'),originalId:'ed8bbfb2-a028-58bd-9ea3-f5bd92433d15',thumbnailId:'d146a114-b15f-5110-8044-123c6506d767',alpha:false},
 {name:'EXIF orientation 4',asset:require('./fixtures/photo-orientation/orientation-4.jpg'),originalId:'3160b3b5-cbb9-5fa5-9d8b-aece5be43561',thumbnailId:'3f1e44c3-823d-5df7-9428-e95d44709da5',alpha:false},
 {name:'EXIF orientation 5',asset:require('./fixtures/photo-orientation/orientation-5.jpg'),originalId:'0de06817-1c78-508b-a9b5-4a3292f37273',thumbnailId:'aa7eb85a-1bba-58d0-9248-40b70d0709e9',alpha:false},
 {name:'EXIF orientation 6',asset:require('./fixtures/photo-orientation/orientation-6.jpg'),originalId:'26ce7003-b40f-5e2c-be40-1c7a10c1fef2',thumbnailId:'077ed74d-b4eb-51a7-88c4-16105a1fb526',alpha:false},
 {name:'EXIF orientation 7',asset:require('./fixtures/photo-orientation/orientation-7.jpg'),originalId:'2fac30ff-f76e-5f32-83de-d04529806420',thumbnailId:'4d9f9b58-8f35-5b4d-a700-761b63153f77',alpha:false},
 {name:'EXIF orientation 8',asset:require('./fixtures/photo-orientation/orientation-8.jpg'),originalId:'12f41c31-abf0-589c-9c67-7d8f6f9eddb5',thumbnailId:'a619f669-1cf3-56cf-94aa-7bbd29f868ed',alpha:false},
];
function Audit(){
 const [status,setStatus]=useState('Running native thumbnail audit');
 const [previews,setPreviews]=useState<Array<{name:string;source:string}>>([]);
 useEffect(()=>{void(async()=>{
  const report:{scope:string;platform:string;cases:Array<Record<string,unknown>>;passed?:boolean;failure?:string;currentStep?:string;sourceUri?:string;sourceInfo?:unknown}={scope:'Actual app staging/thumbnail source with installed native Expo codec; no cloud session or provider',platform:Platform.OS,cases:[]};
  const pickerInputs:string[]=[];
  try{
   authEvents.emitLogin(owner);
   for(const fixture of cases){
    report.currentStep=fixture.name+': download asset';
    const asset=Asset.fromModule(fixture.asset);await asset.downloadAsync();const input=asset.localUri||asset.uri;
    report.sourceUri=input;report.sourceInfo=await FileSystem.getInfoAsync(input).catch(error=>({error:String(error)}));
    report.currentStep=fixture.name+': read original bytes';
    const before=await FileSystem.readAsStringAsync(input,{encoding:FileSystem.EncodingType.Base64});
    const extension=asset.type==='jpg'||asset.type==='jpeg'?'jpg':'png';
    const pickerInput=FileSystem.cacheDirectory+'thumbnail-fixture-'+fixture.originalId+'.'+extension;
    pickerInputs.push(pickerInput);
    await FileSystem.writeAsStringAsync(pickerInput,before,{encoding:FileSystem.EncodingType.Base64});
    report.currentStep=fixture.name+': stage original';
    const original=await stageNativeWardrobeImage(owner,fixture.originalId,pickerInput);
    let baselinePath:string|undefined,baselineBytes:number|undefined;
    if(extension==='jpg'){
     const size=await Image.getSize(original),context=ImageManipulator.manipulate(original);let rendered;
     try{
      if(Math.max(size.width,size.height)>512)context.resize(size.width>=size.height?{width:512}:{height:512});
      rendered=await context.renderAsync();const baseline=await rendered.saveAsync({format:SaveFormat.PNG,compress:1});baselinePath=baseline.uri;
      baselineBytes=decode(await FileSystem.readAsStringAsync(baselinePath,{encoding:FileSystem.EncodingType.Base64})).byteLength;
     }finally{rendered?.release();context.release();}
    }
    report.currentStep=fixture.name+': generate thumbnail';
    const started=Date.now();const thumbnail=await createNativeWardrobeThumbnail(owner,fixture.thumbnailId,original);const elapsedMs=Date.now()-started;
    const after=await FileSystem.readAsStringAsync(original,{encoding:FileSystem.EncodingType.Base64});
    const encoded=await FileSystem.readAsStringAsync(thumbnail,{encoding:FileSystem.EncodingType.Base64});
    const bytes=new Uint8Array(decode(encoded)),dimensions=await Image.getSize(thumbnail);
    const originalBytes=decode(before).byteLength;
    const isPng=bytes[0]===137&&bytes[1]===80,isJpeg=bytes[0]===255&&bytes[1]===216&&bytes[2]===255;
    const alphaChannel=isPng&&(bytes[25]===4||bytes[25]===6);
    const passed=before===after&&(extension==='jpg'?isJpeg:isPng)&&Math.max(dimensions.width,dimensions.height)<=512
     &&(!fixture.alpha||alphaChannel);
    report.cases.push({name:fixture.name,originalPath:original,thumbnailPath:thumbnail,originalBytes,thumbnailBytes:bytes.length,elapsedMs,dimensions,
     smallerThanOriginal:bytes.length<originalBytes,baselinePngPath:baselinePath,baselinePngBytes:baselineBytes,encodedFormat:isJpeg?'jpeg':'png',pngColorType:isPng?bytes[25]:null,originalUnchanged:before===after,alphaChannel,passed});
    setPreviews(current=>[...current,{name:fixture.name,source:`data:image/${isJpeg?'jpeg':'png'};base64,`+encoded}]);
    if(!passed)throw new Error('Native thumbnail assertion failed');
   }
   report.passed=true;setStatus('Native thumbnail audit passed');
  }catch(error){report.passed=false;report.failure=error instanceof Error?error.message:'Unknown failure';setStatus('Native thumbnail audit failed: '+report.failure);}
  finally{for(const input of pickerInputs)await FileSystem.deleteAsync(input,{idempotent:true}).catch(()=>undefined);await FileSystem.writeAsStringAsync(FileSystem.documentDirectory+'thumbnail-audit.json',JSON.stringify(report));}
 })();},[]);
 return <ScrollView contentContainerStyle={{padding:30,paddingTop:80,backgroundColor:'#eee',flexGrow:1}}>
  <Text accessibilityRole="header" style={{fontSize:23,color:'#111'}}>{status}</Text>
  {previews.map(preview=><View key={preview.name} style={{marginTop:20}}><Text style={{color:'#111'}}>{preview.name}</Text><Image source={{uri:preview.source}} style={{width:260,height:300,backgroundColor:'#c7a0db'}} resizeMode="contain"/></View>)}
 </ScrollView>;
}
registerRootComponent(Audit);
