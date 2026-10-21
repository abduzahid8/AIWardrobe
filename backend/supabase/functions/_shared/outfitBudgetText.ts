/** Money syntax shared by the client and Edge; no exchange-rate assumptions. */
export interface BudgetTextCandidate { amount: number; currency: string }
const currencies: Record<string,string> = {'$':'USD',usd:'USD',dollar:'USD',dollars:'USD','€':'EUR',eur:'EUR',euro:'EUR',euros:'EUR','£':'GBP',gbp:'GBP',pound:'GBP',pounds:'GBP',uzs:'UZS'};
const words = '(?:usd|eur|gbp|uzs|dollars?|euros?|pounds?)\\b';
const currency = `([$€£]|${words})`;
const amount = '(\\d{1,3}(?:[ ,.\\u00a0\\u202f]\\d{3})+(?:[.,]\\d{1,2})?|\\d+(?:[.,]\\d{1,2})?)';
const units = /^(?:\s*(?:°\s*)?(?:c|f|celsius|fahrenheit|cm|kg|lbs?|items?|pieces?|degrees?)\b)/i;
function parseAmount(raw: string, code: string): number | null {
    const compact = raw.replace(/[\s\u00a0\u202f]/g,'');
    const decimal = /[.,]\d{1,2}$/.exec(compact);
    const normalized = decimal ? compact.slice(0,decimal.index).replace(/[.,]/g,'') + '.' + decimal[0].slice(1) : compact.replace(/[.,]/g,'');
    const value = Number(normalized), limit = code === 'UZS' ? 1e12 : 100000;
    return Number.isFinite(value) && value >= 0 && value <= limit ? Math.round(value*100)/100 : null;
}
function scan(text?: string | null): {budgets:BudgetTextCandidate[];spans:Array<[number,number]>} {
    const src = String(text || ''), budgets:BudgetTextCandidate[]=[],spans:Array<[number,number]>=[];
    const add = (match:RegExpExecArray, raw:string, token:string, after:string, extra=0) => {
        if(units.test(after))return;
        const code=currencies[token.trim().toLowerCase()] || currencies[token.trim()] || 'USD';
        const value=parseAmount(raw,code);if(value===null)return;
        budgets.push({amount:value,currency:code});spans.push([match.index,match.index+match[0].length+extra]);
    };
    let match:RegExpExecArray|null;
    const prefix=new RegExp(`([$€£])\\s*${amount}`,'g');
    while((match=prefix.exec(src))!==null)add(match,match[2],match[1],src.slice(match.index+match[0].length));
    const suffix=new RegExp(`${amount}\\s*${currency}`,'gi');
    while((match=suffix.exec(src))!==null)add(match,match[1],match[2],src.slice(match.index+match[0].length));
    const keyword=new RegExp(`\\b(?:budget(?:\\s*(?:of|is|:))?|under|below|within|up\\s*to|at\\s*most|no\\s+more\\s+than|less\\s+than|max(?:imum)?(?:\\s+of)?)\\s*${currency}?\\s*${amount}`,'gi');
    while((match=keyword.exec(src))!==null){
        const after=src.slice(match.index+match[0].length),trailing=!match[1]?new RegExp(`^\\s*${currency}`,'i').exec(after):null;
        add(match,match[2],match[1]||trailing?.[1]||'',after,trailing?.[0].length||0);
    }
    return {budgets,spans};
}
export function budgetCandidatesFromText(text?:string|null):BudgetTextCandidate[]{return scan(text).budgets;}
export function stripBudgetText(text?:string|null):string{
    const src=String(text||''),spans=scan(src).spans.sort((a,b)=>a[0]-b[0]);
    let cursor=0,out='';
    for(const [start,end]of spans){if(start>=cursor)out+=src.slice(cursor,start)+' ';cursor=Math.max(cursor,end);}
    return (out+src.slice(cursor)).replace(/\s{2,}/g,' ').replace(/\s+([,.;:!?])/g,'$1').replace(/^[,.;:\s]+/,'').trim();
}
