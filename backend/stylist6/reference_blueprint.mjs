/**
 * Curated textual outfit intentions. Do not treat the user-uploaded collage
 * metadata as an inference license or a photographic reference.
 */
export const REFERENCE_BLUEPRINT_VERSION = 's6-reference-blueprint-v1';
const ID = /^rt_\d{3}$/;
const safe = (value, max = 90) => typeof value === 'string'
  ? value.replace(/[\r\n\t\x00-\x1f]/g, ' ').replace(/\s+/g, ' ').slice(0,max) : '';
const roles=['top','outerwear','bottom','shoes'];
export function buildReferenceBlueprints(index) {
  if(index?.source !== 'user_uploaded_flat_lay_reference_archive' || !Array.isArray(index.images))
    throw Error('invalid_reference_blueprint_index');
  const map=new Map();
  for(const row of index.images){
    if(!ID.test(row?.id||'') || row?.complete_outfit_collage!==true ||
      !Array.isArray(row.palette) || !row.garments || map.has(row.id))continue;
    map.set(row.id,Object.freeze({
      id:row.id,name:safe(row.title),palette:row.palette.filter(x=>typeof x==='string').slice(0,6).map(x=>safe(x,24)),
      silhouette:safe(row.silhouette,60),
      layerPolicy:['required','absent'].includes(row.layer_policy)?row.layer_policy:'unknown',
      garments:Object.fromEntries(roles.map(role=>[role,safe(row.garments[role])])),
      evidenceMode:'textual_editorial_brief_not_visual_reference',
    }));
  }
  return map;
}
