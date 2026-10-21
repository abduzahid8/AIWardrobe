/** Three bounded direct calls through the actual classifier; credentials arrive via stdin only. */
import {readFile,writeFile} from 'node:fs/promises';
import {createInterface} from 'node:readline';
import {createHash} from 'node:crypto';
import ts from 'typescript';
const line=await new Promise(resolve=>{const input=createInterface({input:process.stdin});input.once('line',line=>{input.close();resolve(line);});});
const {token}=JSON.parse(line);if(typeof token!=='string'||token.length<10)throw Error('Provider credential unavailable');
const path='backend/supabase/functions/ai-process/index.ts',source=await readFile(path,'utf8');
const file=ts.createSourceFile(path,source,ts.ScriptTarget.Latest,true);
const required=['buildClothingClassificationPrompt','parseNvidiaResponseText','parseClassificationContent','mapCategoryToSection','ClassificationUnavailableError','classifyClothingWithNvidia'];
const declarations=file.statements.filter(node=>node.name&&required.includes(node.name.text));
if(declarations.length!==required.length)throw Error('Classifier extraction incomplete');
let observations=[];
const trackedFetch=async(url,options)=>{
 if(url!=='https://integrate.api.nvidia.com/v1/chat/completions')throw Error('Unexpected provider target');
 const response=await fetch(url,options);const data=await response.clone().json().catch(()=>null);
 observations.push({status:response.status,finishReason:data?.choices?.[0]?.finish_reason??null,content:data?.choices?.[0]?.message?.content??null});return response;
};
const compiled=ts.transpileModule(`const NVIDIA_API_URL='https://integrate.api.nvidia.com/v1/chat/completions';\n${declarations.map(node=>node.getText(file)).join('\n')}`,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.CommonJS}}).outputText;
const classify=Function('fetch',compiled+'\nreturn classifyClothingWithNvidia;')(trackedFetch);
const report={measuredAt:new Date().toISOString(),sourceHash:createHash('sha256').update(source).digest('hex'),scope:'Three sequential direct NVIDIA calls through exact current production classifier; no hosted auth/quota/capacity claim',cases:[]};
const fixture='scripts/validation/fixtures/opaque-photo/photo-1.jpg';
for(const [name,path] of [['original1',fixture],['original2',fixture],['nativeThumbnail','docs/reports/readiness-evidence/jpeg-thumbnail-native-ios/02-originalPath.jpg']]){
 observations=[];const bytes=await readFile(path),started=Date.now();let accepted=null,error=null;
 try{accepted=await classify(token,`data:image/jpeg;base64,${bytes.toString('base64')}`);}catch(e){error=e.message;}
 const c=accepted?.classification;
 report.cases.push({name,fixture:path,imageBytes:bytes.length,imageHash:createHash('sha256').update(bytes).digest('hex'),latencyMs:Date.now()-started,provider:observations,accepted,error,expected:'Beige/cream torso shirt or jacket; not jeans or draped green layer',primaryMatches:!!c&&['shirt','jacket'].includes(c.category)&&['beige','cream'].includes(c.attributes.color)});
 await writeFile('docs/reports/readiness-evidence/nvidia-primary-garment-current.json',JSON.stringify({...report,inProgress:true},null,2)+'\n');
}
report.allPrimaryMatches=report.cases.every(c=>c.primaryMatches);
await writeFile('docs/reports/readiness-evidence/nvidia-primary-garment-current.json',JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report));
if(!report.allPrimaryMatches)process.exitCode=1;
