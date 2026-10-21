/** Bounded error metadata only: never persist arbitrary messages, URLs or tokens. */
const names=new Set(['Error','TypeError','AbortError','TimeoutError','StorageApiError','StorageUnknownError']);
const codes=new Set(['ETIMEDOUT','ECONNRESET','ENOTFOUND','EAI_AGAIN','UND_ERR_CONNECT_TIMEOUT','UND_ERR_HEADERS_TIMEOUT','UND_ERR_BODY_TIMEOUT','ERR_HTTP2_STREAM_CANCEL']);
export function diagnosticError(error,elapsedMs){
 const status=Number(error?.statusCode??error?.status);
 const original=error?.originalError;
 const code=error?.cause?.code??original?.cause?.code??error?.code;
 return {name:names.has(error?.name)?error.name:'UnclassifiedError',
  originalName:original?(names.has(original.name)?original.name:'UnclassifiedError'):null,
  status:Number.isInteger(status)&&status>=100&&status<=599?status:null,
  networkCode:codes.has(code)?code:null,
  elapsedMs:Number.isFinite(elapsedMs)&&elapsedMs>=0?Math.round(elapsedMs):null};
}
