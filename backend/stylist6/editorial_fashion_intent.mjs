/**
 * One human-centered artistic objective shared by the FIRST designer, exact
 * source-SKU artist, independently hosted original-photo critic and the final
 * whole-collection editor. These are server-authored editorial QUESTIONS,
 * never subtype bans, fixed color recipes, user-supplied instructions or
 * changes to any physical-source/photo admission threshold.
 *
 * A classic Date Night was repeatedly treated as an office meeting by the
 * separate AI studios; they were interpreting "classic" as the OCCASION.
 * Resolve the intended social scene ONCE and pass that same artistic target
 * to EVERY stage, so late independent review checks the FIRST commission's
 * actual human goal rather than rewarding generic office polish.
 */
export const EDITORIAL_FASHION_INTENT_VERSION='s6-human-scene-whole-collection-v1';
const CAPSULE='Across the collection, seek three compelling, genuinely different full-look artistic points of view, each appropriate to this same person and event. Compare actual silhouette, layer construction, material interplay and footwear; a recolor or renamed office uniform is not a new design. Do not pad a weak third look.';
const SCENES=Object.freeze({
  date: Object.freeze({
    scene:'personal_evening_date',
    creativeGoal:'Dress a person for a real social DATE evening: attractive, confident, relaxed elegance with individuality, warmth and intentional detail; interpret their own style rather than dressing for a business meeting.',
    photoJudgment:'Assess the PHOTOGRAPHED complete look for a personal evening date, not an office event. Tailoring can be excellent, but office-safe polish ALONE is not proof of date-night charisma or distinctive styling. Look for visible garment relationships and whether they create an intentional SOCIAL impression; do not apply an invented color or garment ban.',
  }),
  work: Object.freeze({
    scene:'professional_work',
    creativeGoal:'Dress the person for their real WORK setting: credible, well-considered polish, proportion, comfort and personal identity. Elegant creativity can coexist with professionalism; do not repeatedly submit the same safe office uniform under different names.',
    photoJudgment:'Assess the PHOTOGRAPHED complete outfit for professional work, paying attention to credible formality and cohesive real proportions. Reward intentional styling rather than only a predictable jacket/trouser silhouette. Do not require an invented dress code beyond the supplied one.',
  }),
  formal: Object.freeze({
    scene:'formal_event',
    creativeGoal:'Create thoughtful, genuinely refined formal-event dressing that respects the ACTUAL requested dress code. Seek original expression in proportion, material, finishing and color relationships where appropriate; do not substitute street or leisure style merely to increase novelty.',
    photoJudgment:'Assess the PHOTOGRAPHED complete outfit against the actual formal event and requested dress code, not generic office rules. Elegant material/cut differences may be meaningful without changing garment roles; assess what is visible rather than inventing a rule.',
  }),
  casual: Object.freeze({
    scene:'relaxed_social',
    creativeGoal:'Style an intentional relaxed social outfit: ease, authentic personality and pleasing tactile and proportion contrast. Avoid turning this relaxed occasion into a corporate uniform simply because the wearer likes classic clothing.',
    photoJudgment:'Assess the PHOTOGRAPHED outfit for a relaxed social setting: does its construction, comfort and visual composition suit the selected event rather than a business dress code? Do not automatically assume formal garments are superior.',
  }),
  athletic: Object.freeze({
    scene:'active_movement',
    creativeGoal:'Style the real physical ACTIVITY requested, balancing function, movement, climate and a coherent personal look. Do not prioritize decorative fashion over the actual activity.',
    photoJudgment:'Assess the PHOTOGRAPHED complete workout/activity outfit for the requested movement and garment functionality first; do not impose office or evening fashion as the aesthetic standard.',
  }),
  other: Object.freeze({
    scene:'user_selected_event',
    creativeGoal:'Style the actual user-selected occasion and personal fashion identity. Interpret the stated social setting, formality and mood as a coherent whole rather than defaulting to generic office tailoring.',
    photoJudgment:'Assess the PHOTOGRAPHED complete outfit for the user-selected event and preferences, not an assumed office setting or an invented fashion dress code.',
  }),
});
export function editorialFashionIntent({occasion}={}){
  // Untrusted customer strings select only a finite trusted editorial scene.
  // No arbitrary user or merchant text is ever copied into these prompts.
  const selected=typeof occasion==='string'?occasion.slice(0,90)
    .toLowerCase().replace(/[_-]+/g,' '):'';
  const scene=/\bdate\b|свидан/.test(selected)?'date':
    /\b(wedding|formal|black tie|ceremony|gala)\b|свадь|торжест/.test(selected)?'formal':
    /\b(work|office|business|conference|meeting)\b|работ/.test(selected)?'work':
    /\b(workout|gym|training|sport|fitness|run)\b|трениров|спортзал/.test(selected)?'athletic':
    /\b(casual|weekend|friends|coffee|hangout|brunch)\b|прогул|выходн/.test(selected)?'casual':'other';
  return {...SCENES[scene],collectionGoal:CAPSULE};
}
