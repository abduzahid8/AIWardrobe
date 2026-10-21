/** Diagnostic-only entry; never imported by the shipping application. */
import React, { useEffect, useState } from 'react';
import { Text, View, Platform } from 'react-native';
import { registerRootComponent } from 'expo';
import * as FileSystem from 'expo-file-system/legacy';
import * as VideoThumbnails from 'expo-video-thumbnails';
import { readVideoSampleTimes, readAccurateVideoFrames } from '../../src/services/videoFrameSampling';
import { commitScanReview,loadScanReview,acknowledgeScanReviewItem,findScanReviewBySource } from '../../src/services/scanReviews';
import { authEvents } from '../../store/authEvents';
const owner='11111111-1111-4111-8111-111111111111';
function NativeVideoProbe() {
  const [status, setStatus] = useState('Reading test video…');
  useEffect(() => {
    void (async () => {
      const root=FileSystem.documentDirectory!;let report:Record<string,unknown>;let restarting=false;
      try {
        authEvents.emitLogin(owner); // Isolated cache-owner fixture, not a Supabase session.
        const statePath=root+'native-review-probe-state.json';
        if((await FileSystem.getInfoAsync(statePath)).exists){
          restarting=true;const state=JSON.parse(await FileSystem.readAsStringAsync(statePath));
          const restored=await loadScanReview(owner,state.reviewId);
          const exactImages=restored.detectedItems.every((item,index)=>item.frameImage===state.images[index]);
          const stableIDs=restored.detectedItems.every((item,index)=>item.reviewItemId===state.itemIds[index]);
          if(!exactImages||!stableIDs)throw Error('Review changed across native restart');
          for(const item of restored.detectedItems)await acknowledgeScanReviewItem(owner,state.reviewId,item.reviewItemId!);
          const completed=await findScanReviewBySource(owner,'native-review-queue');
          if(!completed||completed.detectedItems.length)throw Error('Missing completed-source acknowledgement');
          report={success:true,exactImages,stableIDs,acknowledgedItems:restored.detectedItems.length,completedSourceRemembered:true};
          setStatus('Native review restart passed');
        }else{
          const uri=root+'codec-audit.mp4',times=await readVideoSampleTimes(uri),frames=[];
          if(Platform.OS==='android'){
            const accurate=await readAccurateVideoFrames(uri);
            for(const sample of accurate){const file=`sample-${sample.timeMs}.jpg`;await FileSystem.writeAsStringAsync(root+file,sample.base64,{encoding:'base64'});frames.push({timeMs:sample.timeMs,base64Length:sample.base64.length,file});}
          }else for(const time of times){
            const frame=await VideoThumbnails.getThumbnailAsync(uri,{time,quality:0.9});const file=`sample-${time}.jpg`;
            await FileSystem.copyAsync({from:frame.uri,to:root+file});
            const content=await FileSystem.readAsStringAsync(frame.uri,{encoding:'base64'});
            frames.push({timeMs:time,width:frame.width,height:frame.height,base64Length:content.length,file});
            await FileSystem.deleteAsync(frame.uri,{idempotent:true});
          }
          const repeatedTimes=await readVideoSampleTimes(uri);
          const items=[];
          for(const frame of frames){const base64=await FileSystem.readAsStringAsync(root+frame.file,{encoding:'base64'});items.push({itemType:'jacket',color:'beige',style:'casual',description:'Generated probe fixture',position:'upper' as const,confidence:0,confidenceLevel:'low' as const,detectionSources:['fixture'],frameImage:'data:image/jpeg;base64,'+base64});}
          const review=await commitScanReview(owner,{detectedItems:items},'native-review-queue');
          await FileSystem.writeAsStringAsync(statePath,JSON.stringify({reviewId:review.reviewId,itemIds:review.detectedItems.map(item=>item.reviewItemId),images:review.detectedItems.map(item=>item.frameImage)}));
          report={success:true,times,repeatedTimes,frames,reviewCommitted:true};setStatus('Native video sampling completed');
        }
      }catch(error){report={success:false,error:error instanceof Error?error.message:'Native probe failed'};setStatus('Native probe failed');}
      await FileSystem.writeAsStringAsync(root+(restarting?'native-review-restart-result.json':'native-video-sampling-result.json'),JSON.stringify(report));
    })();
  },[]);
  return <View style={{padding:40}}><Text>{status}</Text></View>;
}
registerRootComponent(NativeVideoProbe);
