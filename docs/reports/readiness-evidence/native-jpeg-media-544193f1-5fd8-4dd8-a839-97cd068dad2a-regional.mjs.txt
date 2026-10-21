export const REGIONAL_MEDIA_CASES = Object.freeze([
  Object.freeze({suffix:'tokyo',region:'ap-northeast-1',cohort:'remoteIds'}),
  Object.freeze({suffix:'frankfurt',region:'eu-central-1',cohort:'localIds'}),
]);

/** A requested header alone does not establish where a probe ran. */
export function verifyRegionalMediaExecution(result, expectedRegion, responseRegion) {
  if(!REGIONAL_MEDIA_CASES.some(value=>value.region===expectedRegion)
    || result?.region!==expectedRegion || responseRegion!==expectedRegion)throw new Error('ProbeRegionMismatch');
}
