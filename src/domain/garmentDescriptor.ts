import { normalizeGarmentText } from './garmentText.ts';
import { colorFamilyOf, materialsOf, patternOf, textureOf } from './outfitDiversity.ts';
export interface GarmentEvidence {
  id?: string | number;
  name?: string | null;
  type?: string | null;
  macroCategory?: string | null;
  category?: string | null;
  isShopItem?: boolean;
  description?: string | null;
  color?: string | null;
  pattern?: string | null;
  material?: unknown;
  materials?: unknown;
}
export interface GarmentDescriptor {
  slot: string;
  subtype: string;
  color: string;
  materials: string[];
  texture: string;
  pattern: string;
  formality: 'athletic' | 'casual' | 'smart' | 'formal' | 'unknown';
  volume: 'slim' | 'regular' | 'wide' | 'unknown';
  bulk: 'light' | 'heavy' | 'unknown';
  collar: 'shirt' | 'high' | 'open' | 'crew' | 'unknown';
  length: 'cropped' | 'long' | 'regular' | 'unknown';
  closure: 'openable' | 'pullover' | 'unknown';
  /** Positive construction evidence; rubber alone does not imply casual. */
  casualSole: boolean;
  soleProfile: 'dress' | 'comfort' | 'sneaker' | 'rugged' | 'unknown';
  fadedDenim: boolean;
  loudBranding: boolean;
  sportyZip: boolean;
  utilityConstruction: boolean;
  technicalLeisure: boolean;
  rugged: boolean;
  athletic: boolean;
  structured: boolean;
  evening: boolean;
  insulated: boolean;
  evidence: string[];
}
/** Text classifications are derived evidence, never authoritative catalog writes. */
export function describeGarment(item: GarmentEvidence): GarmentDescriptor {
  const materialText = typeof item.material === 'string' ? item.material
    : Array.isArray(item.material) ? item.material.filter(value => typeof value === 'string').join(' ') : '';
  const text = normalizeGarmentText(
    [item.name, item.type, item.description, materialText,
      Array.isArray(item.materials) ? item.materials.filter(value => typeof value === 'string').join(' ') : item.materials, item.pattern]
      .filter(value => typeof value === 'string').join(' ')
  );
  const slot = String(item.macroCategory || item.category || 'unknown');
  const rules: Array<[string, RegExp]> =
    slot === 'shoes'
      ? [
          ['hiking-boot', /\b(hiking|trekking|mountaineer|snow boot|combat|rugged|lug[ -]sole)\b/],
          ['oxford', /\boxfords?\b/],
          ['derby', /\bderb(?:y|ies)\b/],
          ['monk', /\bmonk/],
          ['espadrille', /\bespadrilles?\b/],
          ['sandal', /\b(sandals?|slides?|flip[ -]flops?)\b/],
          ['loafer', /\b(loafers?|penny|horsebit)\b/],
          ['driver', /\b(driving|drivers?)\b/],
          ['chelsea', /\bchelsea\b/],
          ['dress-boot', /\b(dress|sleek|slim)[ -].*boots?\b/],
          ['sport-sneaker', /\b(running|training|trail|performance|athletic)\b/],
          [
            'minimal-sneaker',
            /\b(minimal|plain|clean|low[ -]top|leather)\b.*\b(sneakers?|trainers?)\b/,
          ],
          ['sneaker', /\b(sneakers?|trainers?)\b/],
          ['boot', /\bboots?\b/],
        ]
      : slot === 'bottom'
        ? [
            [
              'athletic-trouser',
              /\b(joggers?|track[ -]?(pants?|trousers?)|training|sweatpants?|draw[ -]?strings?|athletic|sports? pants?)\b/,
            ],
            ['cargo', /\bcargo\b/],
            ['jeans', /\bjeans|denim\b/],
            ['chino', /\bchinos?\b/],
            [
              'tailored-trouser',
              /\b(tailored|suit|tuxedo|dress|formal|pleated|darted|pinstriped|flannel|smart)\b/,
            ],
            // Knit construction alone does not prove sports detailing. Keep it
            // distinct from tailoring unless the catalog explicitly says tailored.
            ['knit-trouser', /\b(interlock|jersey|knitted|knit)\b/],
            ['trouser', /\b(trousers?|pants?|slacks?)\b/],
          ]
        : slot === 'outerwear'
          ? [
              [
                'blazer',
                /\b(blazers?|suit jacket|tailored jacket|sports? coat|dinner jacket|tuxedo)\b/,
              ],
              ['cardigan', /\bcardigans?\b/],
              ['overshirt', /\b(overshirts?|shackets?)\b/],
              ['hoodie', /\bhoodies?\b/],
              ['coat', /\b(coats?|peacoats?|overcoats?|trench|parka|puffer)\b/],
              ['bomber', /\bbomber\b/],
              ['pullover', /\b(sweaters?|pullovers?|jumpers?|knitwear)\b/],
              ['jacket', /\b(jackets?|trucker)\b/],
            ]
          : [
              ['t-shirt', /\b(t[ -]?shirts?|tees?)\b/],
              ['polo', /\bpolos?\b/],
              ['field-shirt', /\b(field|utility|western|military|ripstop)\b.*\bshirts?\b/],
              ['dress-shirt', /\b(dress|formal|poplin|oxford)\b.*\bshirts?\b/],
              ['shirt', /\bshirts?\b/],
              ['fine-knit', /\b(fine|lightweight|thin|gauge|merino)\b.*\b(knit|sweater|pullover)/],
              ['sweater', /\b(sweaters?|pullovers?|jumpers?)\b/],
            ];
  const subtype = rules.find(([, r]) => r.test(text))?.[0] || 'unknown';
  const athletic =
    /\b(athletic|training|track[ -]?pants?|joggers?|sweatpants?|workout|sportswear)\b/.test(text) ||
    subtype === 'sport-sneaker' ||
    subtype === 'athletic-trouser';
  const rugged =
    subtype === 'hiking-boot' ||
    /\b(rugged|lug[ -]sole|hiking|trekking|combat)\b/.test(text) ||
    (slot === 'shoes' &&
      /\bwaterproof\b/.test(text) &&
      /\b(grip|chunky|traction|tread)\b/.test(text));
  // The imported Flextride row omits its construction. Model-specific
  // evidence: Geox U66MAC0009BC9999, cushioning and synthetic/rubber outsole.
  // https://www.geox.com/en-GB/leather_loafers-black-flextride_plus_man-U66MAC0009BC9999.html
  // Exact Spherica leather rows additionally document a layered Zero Shock
  // comfort sole. This is comfort evidence, never a sneaker classification:
  // https://www.geox.com/en-GB/leather_dress_loafers-coffee-spherica_ec16_man-U65HYB00046C6009.html
  // https://www.geox.com/en-US/leather_dress_loafers-black-spherica_ec1_b_man-U66CJC00047C9999.html
  const verifiedComfortLoafer = slot === 'shoes' && (/^(?:shop_)?geox-(?:u66mac|u44q6d|u56q6a|u65hyb00046|u66cjc00047)/i.test(String(item.id || ''))
    || /^(?:flextride plus|moner w 2fit)\b/i.test(item.name || ''));
  // Silhouette outranks a marketing "dress" label. Cushioning/rubber alone
  // never establishes a sneaker sole; a slim rubber dress sole stays dressy.
  const sneakerSole = slot === 'shoes' && /\b(sneaker[ -]+(?:style[ -]+)?soles?|cup[ -]?soles?|hybrid[ -]+(?:oxfords?|derb(?:y|ies)|loafers?|dress shoes?)|(?:oxfords?|loafers?).{0,25}hybrid)\b/.test(text);
  const comfortSole = slot === 'shoes' && (verifiedComfortLoafer
    || /\b(chunky|thick|platform|lug|commando|track)[ -]+(?:rubber[ -]+)?(?:out)?sole(?:s|d)?\b/.test(text)
    || /\b(?:crepe|gum|white|contrast)[ -]+(?:rubber[ -]+)?soles?\b/.test(text));
  const soleProfile: GarmentDescriptor['soleProfile'] = slot !== 'shoes' ? 'unknown'
    : sneakerSole || ['sport-sneaker', 'minimal-sneaker', 'sneaker'].includes(subtype) ? 'sneaker'
    : rugged ? 'rugged' : comfortSole ? 'comfort'
    : /\b(?:leather[ -]+(?:out)?soles?|(?:slim|thin|low[ -]profile)[ -]+(?:rubber[ -]+)?soles?|blake|goodyear|welted)\b/.test(text) ? 'dress' : 'unknown';
  const casualSole = ['comfort', 'sneaker', 'rugged'].includes(soleProfile);
  // A lone patch pocket is normal on an Oxford shirt. Paired/flapped utility
  // pockets, including the verified Bond/Armani rows, are different evidence.
  const utilityConstruction = slot === 'top' && (subtype === 'field-shirt'
    || /^(?:shop_)?(?:allsaints-usm108sf-|armani_exchange-xm003247_)/i.test(String(item.id || ''))
    || /\b(?:utility|cargo|bellows)[ -]+pockets?\b/.test(text)
    || /\b(?:chest[ -]+flap|flap[ -]+chest|(?:two|twin|double)[ -]+(?:chest|front)(?:[ -]+flap)?)[ -]+pockets?\b/.test(text)
    || ( /\bshort[ -]sleeve(?:d)?\b/.test(text) && /\b(?:two|twin|double)[ -]+patch[ -]+pockets?\b/.test(text)));
  const technicalLeisure = slot === 'outerwear'
    && ((/\btechnical\b/.test(text) && /\b(leisure|shell|nylon|zip(?:ped)?|windproof|track|jacket)\b/.test(text))
      || (/\b(nylon|shell|windbreaker|track)\b/.test(text) && /\b(zip(?:ped)?|jacket|windproof)\b/.test(text)));
  const sportyZip = slot === 'outerwear' && /\b(zip(?:ped)?|zip[ -]up|full[ -]zip)\b/.test(text)
    && (technicalLeisure || /\b(track|athletic|sports?|sweatshirt|fleece|contrast[ -]+(?:bands?|panels?))\b/.test(text));
  const fadedDenim = subtype === 'jeans'
    && /\b(faded|fading|bleached|acid[ -]wash(?:ed)?|stone[ -]wash(?:ed)?|light[ -]wash(?:ed)?|distressed|ripped|shredded|whisker(?:ed|ing)?)\b/.test(text);
  const loudBranding = /\b(monograms?|graphics?|slogans?|lettering|all[ -]over[ -]+(?:logos?|branding)|repeat(?:ed|ing)?[ -]+logos?)\b/.test(text)
    || (/\blogos?\b/.test(text) && /\b(large|oversized|bold|printed|repeat(?:ed|ing)?|all[ -]over)\b/.test(text));
  const structured =
    subtype === 'blazer' || subtype === 'tailored-trouser' || subtype === 'dress-shirt';
  // A single marketing word ("dress", "formal") is not proof on its own
  // when the rest of the name contradicts it. Dress-loafer status requires
  // either a construction cue (structured last, Blake stitching, welt,
  // leather outsole, calfskin, polished apron, saddle, or a named loafer
  // construction such as penny/tassel/horsebit — cf. occasionRules: these
  // go with suits) or leather — itself genuine dress-shoe construction
  // evidence — plus a dress/formal claim with no casual counter-evidence
  // (suede, canvas, boat, moccasin…).
  const dressLoafer =
    subtype === 'loafer' && !casualSole &&
    (/\b(structured|blake|welt|calfskin|polished|apron|saddle|penny|tassel|horsebit|leather\s+out?soles?)\b/.test(
      text
    ) ||
      (/\bleather\b/.test(text) &&
        /\b(dress|formal)\b/.test(text) &&
        !/\b(suede|canvas|fabric|knit|sneaker|sport|sports|casual|boat|moccasin|driver)\b/.test(
          text
        )));
  const formality = athletic
    ? 'athletic'
    : casualSole || technicalLeisure || utilityConstruction ? 'casual'
    : [
          'oxford',
          'derby',
          'monk',
          'dress-boot',
          'blazer',
          'tailored-trouser',
          'dress-shirt',
        ].includes(subtype) || dressLoafer
      ? 'formal'
      : ['loafer', 'chelsea', 'chino', 'fine-knit', 'cardigan'].includes(subtype)
        ? 'smart'
        : subtype === 'unknown'
          ? 'unknown'
          : 'casual';
  const volume = /\b(slim|skinny|fitted|tight)\b/.test(text)
    ? 'slim'
    : /\b(wide|oversized|baggy|boxy|relaxed)\b/.test(text)
      ? 'wide'
      : /\b(regular|straight)\b/.test(text)
        ? 'regular'
        : 'unknown';
  const bulk =
    /\b(chunky|heavyweight|bulky|thick|cable[ -]knit|padded|shearling|puffer|down jacket)\b/.test(
      text
    )
      ? 'heavy'
      : /\b(lightweight|fine|thin|unlined|linen|poplin|gauze)\b/.test(text)
        ? 'light'
        : 'unknown';
  const collar = /\b(turtle|roll|high)[ -]?neck\b/.test(text)
    ? 'high'
    : /\b(shirts?|polos?)\b/.test(text)
      ? 'shirt'
      : /\b(v[ -]?neck|open[ -]?neck)\b/.test(text)
        ? 'open'
        : /\bcrew[ -]?neck\b/.test(text)
          ? 'crew'
          : 'unknown';
  const length = /\b(cropped|crop)\b/.test(text)
    ? 'cropped'
    : /\b(longline|long[ -]length|long[ -]hem)\b/.test(text)
      ? 'long'
      : /\b(regular[ -]length|hip[ -]length)\b/.test(text)
        ? 'regular'
        : 'unknown';
  const closure = /\b(pullover|pull[ -]over)\b/.test(text)
    ? 'pullover'
    : ['blazer', 'cardigan', 'overshirt', 'jacket', 'coat', 'bomber'].includes(subtype)
      ? 'openable'
      : 'unknown';
  return {
    slot,
    subtype,
    length,
    closure,
    color: colorFamilyOf(item),
    materials: materialsOf(item),
    texture:
      /\b(knit|knitted|jersey|suede|denim|technical|wool|corduroy|smooth|ribbed|tweed)\b/.test(
        `${text} ${String(item.material || '')}`
      )
        ? textureOf(item)
        : 'unknown',
    pattern:
      typeof item.pattern === 'string' &&
      item.pattern.trim() &&
      !/^(unknown|other)$/i.test(item.pattern.trim())
        ? patternOf(item)
        : /\b(solid|plain|unpatterned|striped?|checks?|checked|plaid|graphic|logo|floral|paisley|print(?:ed)?|camo|jacquard|polka)\b/.test(
              text
            )
          ? patternOf(item)
          : 'unknown',
    formality,
    volume,
    bulk,
    collar,
    rugged,
    casualSole,
    soleProfile,
    fadedDenim,
    loudBranding,
    sportyZip,
    utilityConstruction,
    technicalLeisure,
    athletic,
    structured,
    insulated: /\b(puffer|down jacket|down filling|insulated|shearling)\b/.test(text),
    evening: /\b(tuxedo|dinner jacket|dinner suit|opera pump|patent)\b/.test(text),
    evidence: [
      ...(soleProfile !== 'unknown' ? [`sole:${soleProfile}`] : []),
      ...(fadedDenim ? ['surface:faded-denim'] : []),
      ...(loudBranding ? ['pattern:loud-branding'] : []),
      ...(sportyZip ? ['construction:sporty-zip'] : []),
      ...(utilityConstruction ? ['construction:utility-pockets'] : []),
      ...(technicalLeisure ? ['construction:technical-leisure'] : []),
      ...(verifiedComfortLoafer ? ['catalog:verified-comfort-sole'] : []),
      ...(subtype !== 'unknown' ? [`text:${subtype}`] : []),
      ...(bulk !== 'unknown' ? [`text:${bulk}`] : []),
      ...(volume !== 'unknown' ? [`text:${volume}`] : []),
    ],
  };
}
