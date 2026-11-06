// src/domain/garmentText.ts
var SHORT_SLEEVE_RE = /\bshort[\s-]?sleeve(?:s|d)?\b/;
function normalizeGarmentText(value) {
  const aliases = [
    [/свитшот/gi, "sweatshirt"],
    [/худи/gi, "hoodie"],
    [/логотип|monologo/gi, "logo"],
    [/лофер[а-я]*/gi, "loafers"],
    [/кроссов[а-я]*|кеды/gi, "sneakers"],
    [/рубаш[а-я]*/gi, "shirt"],
    [/брюк[а-я]*/gi, "trousers"],
    [/пиджак[а-я]*/gi, "blazer"],
    [/смокинг[а-я]*/gi, "tuxedo"],
    [/замш[а-я]*/gi, "suede"],
    [/коричнев[а-я]*/gi, "brown"],
    [/черн[а-я]*|чёрн[а-я]*/gi, "black"],
    [/бел[а-я]*/gi, "white"],
    [/футбол[а-я]*/gi, "t-shirt"],
    [/спортивн[а-я]*/gi, "athletic"],
    [/трениров[а-я]*/gi, "training"],
    [/шорт[а-я]*/gi, "shorts"]
  ];
  return aliases.reduce(
    (text3, [pattern, replacement]) => text3.replace(pattern, ` ${replacement} `),
    value.toLowerCase()
  );
}

// src/domain/outfitDiversity.ts
var DIVERSITY_SLOTS = ["outerwear", "top", "bottom", "shoes"];
var FORMULA_SLOTS = ["outerwear", "top", "bottom"];
var DIVERSITY_WEIGHTS = {
  archetype: 0.32,
  families: 0.18,
  silhouette: 0.1,
  color: 0.1,
  footwear: 0.08,
  material: 0.07,
  pattern: 0.05,
  formality: 0.1
};
var DIVERSITY_BONUS_PER_ARCHETYPE = 0.35;
function text(value) {
  return typeof value === "string" ? value : value == null ? "" : String(value);
}
function nameBlob(item) {
  if (!item) return "";
  return `${text(item.name)} ${text(item.type)} ${text(item.subCategory)} ${text(item.category)}`.toLowerCase();
}
function garmentBlobs(item) {
  return {
    name: text(item?.name).toLowerCase(),
    meta: `${text(item?.type)} ${text(item?.subCategory)} ${text(item?.category)}`.toLowerCase()
  };
}
function fullBlob(item) {
  if (!item) return "";
  return `${text(item.name)} ${text(item.type)} ${text(item.subCategory)} ${text(item.category)} ${text(item.description)} ${text(item.material)} ${text(item.brand)}`.toLowerCase();
}
var NON_ALPHANUMERIC = (() => {
  try {
    return new RegExp("[^\\p{L}\\p{N}]+", "gu");
  } catch {
    return /[^a-z0-9\u00c0-\u024f\u0370-\u03ff\u0400-\u052f\u4e00-\u9fff\uac00-\ud7af]+/gi;
  }
})();
function unicodeFold(value) {
  try {
    return typeof value.normalize === "function" ? value.normalize("NFKC") : value;
  } catch {
    return value;
  }
}
function normalizeNameKey(name) {
  const raw = unicodeFold(text(name)).toLowerCase();
  return raw.replace(NON_ALPHANUMERIC, " ").replace(/\s+/g, " ").trim();
}
function imageKey(item) {
  const raw = text(item?.imageUrl || item?.image).trim().toLowerCase();
  if (!raw) return "";
  return raw.split("#")[0].split("?")[0].replace(/^https?:\/\//, "").replace(/^\/\//, "").replace(/\/+$/, "");
}
function productIdentityOf(item) {
  const id = text(item?.id).trim();
  const idKey = id ? `id:${id}` : "";
  const name = normalizeNameKey(item?.name);
  const nameKey = name ? `n:${name}` : "";
  const image = imageKey(item);
  const imageKeyValue = image ? `i:${image}` : "";
  const canonicalKey = nameKey || imageKeyValue || idKey;
  const keys = [idKey, nameKey, imageKeyValue].filter(Boolean);
  return { idKey, nameKey, imageKey: imageKeyValue, canonicalKey, keys };
}
function productIdentityKeys(item) {
  return productIdentityOf(item).keys;
}
function canonicalProductKey(item) {
  return productIdentityOf(item).canonicalKey;
}
function countDistinctProducts(items) {
  const groups = [];
  for (const item of Array.isArray(items) ? items : []) {
    if (!item) continue;
    const keys = productIdentityKeys(item);
    if (keys.length === 0) continue;
    const matches = groups.filter((group) => keys.some((key) => group.has(key)));
    if (matches.length === 0) {
      groups.push(new Set(keys));
      continue;
    }
    const [first, ...rest] = matches;
    for (const key of keys) first.add(key);
    for (const group of rest) {
      for (const key of group) first.add(key);
      groups.splice(groups.indexOf(group), 1);
    }
  }
  return groups.length;
}
var SLOT_BY_MACRO = {
  outerwear: "outerwear",
  top: "top",
  bottom: "bottom",
  shoes: "shoes",
  // A dress replaces the top AND the bottom. The structural validator
  // normalizes it to the 'top' slot, so product identity and capacity must
  // use the same mapping or a dress deck would look "bottomless".
  dress: "top"
};
function diversitySlotOf(item) {
  const macro = text(item?.macroCategory).toLowerCase().trim();
  if (SLOT_BY_MACRO[macro]) return SLOT_BY_MACRO[macro];
  const blob = nameBlob(item);
  if (/\b(shoe|sneaker|boot|loafer|sandal|trainer|derby|moccasin|oxford)\b/.test(blob)) return "shoes";
  if (/\b(pant|trouser|jean|chino|short|skirt|jogger)\b/.test(blob)) return "bottom";
  if (/\b(jacket|coat|blazer|hoodie|cardigan|sweater|pullover|overshirt|vest|puffer|bomber|trench|peacoat)\b/.test(blob)) return "outerwear";
  if (/\b(shirt|tee|t-?shirt|polo|blouse|henley|tank|top)\b/.test(blob)) return "top";
  return null;
}
var OUTERWEAR_FAMILIES = [
  [/\b(tuxedos?|dinner\s*suit|black[\s-]?tie)\b/, "tuxedo"],
  // A dinner jacket is a different garment from a tuxedo: a shawl-collar
  // midnight dinner jacket reads as a different look than a classic black
  // tuxedo, and black-tie decks must be able to show both.
  [/\b(dinner\s*jackets?|tuxedo\s*jackets?)\b/, "dinner-jacket"],
  [/\bsuede\b/, "suede-jacket"],
  [/\b(blazers?|suit\s*jackets?|sport\s*coats?|blousons?)\b/, "blazer"],
  [/\b(overcoats?|top\s*coats?|peacoats?|trench|duffle|greatcoats?|mac\s*coats?|winter\s*coats?|parkas?|puffer|down\s*jackets?|capes?)\b/, "coat"],
  [/\b(overshirts?|shackets?|safari\s*jackets?|field\s*jackets?|chore\s*(coats?|jackets?)|casual\s*jackets?|work\s*jackets?|coach\s*jackets?|ruckers?)\b/, "overshirt"],
  [/\bcardigans?\b/, "cardigan"],
  [/\b(hoodies?|sweatshirts?|sweat[\s-]shirts?)\b/, "hoodie"],
  [/\bbombers?\b/, "bomber"],
  [/\b(gilets?|waistcoats?|vests?)\b/, "vest"],
  [/\b(technical|softshell|windbreakers?|anoraks?|rain\s*jackets?|shell\s*jackets?|3-?in-?1)\b/, "utility"],
  [/\b(sweaters?|pullovers?|jumpers?|crewnecks?|roll[\s-]?necks?|turtlenecks?|fisherman|cardigan\s*jackets?)\b/, "knitwear"],
  // A plain coat and a peacoat are the same ROLE; a plain jacket is not.
  [/\b(coats?)\b/, "coat"],
  [/\b(jackets?)\b/, "jacket-generic"]
];
var TOP_FAMILIES = [
  [/\b(tuxedos?|dress\s*shirts?|formal\s*shirts?|wing[\s-]?collar|french\s*cuff|pleated\s*shirts?)\b/, "dress-shirt"],
  [/\b(oxfords?|poplin)\b/, "dress-shirt"],
  [/\b(polos?|piqu[eé])\b/, "polo"],
  [/\b(sweaters?|crewnecks?|roll[\s-]?necks?|turtlenecks?|merino|cashmere|knitwear|fine[\s-]?gauge\s*knit)\b/, "knit"],
  [/\bhenleys?\b/, "henley"],
  [/\bblouses?\b/, "blouse"],
  [/\b(athletic|training|performance|dri-?fit|gym|running)\b/, "athletic"],
  [/\b(t-?shirts?|tees?)\b/, "tee"],
  [/\b(tanks?|sleeveless)\b/, "tank"],
  [/\b(shirts?|button[\s-]?downs?|flannel|chambray|linen)\b/, "shirt"]
];
var BOTTOM_FAMILIES = [
  [/\b(shorts?|bermudas?)\b/, "short"],
  [/\b(jeans?|denim)\b/, "jean"],
  [/\b(corduroy|cords?)\b/, "corduroy"],
  [/\b(joggers?|sweatpants?|sweat[\s-]pants?|track\s*pants?|leisure)\b/, "jogger"],
  [/\bleather\s*(pants?|trousers?)/, "leather"],
  [/\b(chinos?|khakis?|cotton\s*(pants?|trousers?)|casual\s*(pants?|trousers?)|twill)\b/, "chino"],
  [/\b(tailored?|dress\s*(pants?|trousers?)|suit|pleated|wool\s*(pants?|trousers?)|slacks?|formal\s*(pants?|trousers?)|tux|trousers)\b/, "tailored"],
  [/\b(pants?)\b/, "trousers-generic"]
];
var SHOE_FAMILIES = [
  [/\b(patent|opera\s*pumps?|tuxedo\s*shoes?)\b/, "tuxedo-shoe"],
  [/\b(moccasins?|driving|driver|gommino)\b/, "moccasin"],
  [/\b(loafers?|penny|tassel)\b/, "loafer"],
  [/\b(oxfords?|derbies|derby|brogues?|monk|cap-?toe|wingtip)\b/, "dress-shoe"],
  [/\b(boots?|chelsea|chukka)\b/, "boot"],
  [/\b(sneakers?|trainers?|running\s*shoes?|court)\b/, "sneaker"],
  [/\b(sandals?|slides?|mules?|espadrilles?|flip[\s-]?flops?)\b/, "sandal"],
  [/\bshoes?\b/, "dress-shoe"]
];
var FAMILY_TABLES = {
  outerwear: OUTERWEAR_FAMILIES,
  top: TOP_FAMILIES,
  bottom: BOTTOM_FAMILIES,
  shoes: SHOE_FAMILIES
};
var SLOT_PREFIX = {
  outerwear: "outer",
  top: "top",
  bottom: "bottom",
  shoes: "shoe"
};
function familyFor(slot, item) {
  const { name, meta } = garmentBlobs(item);
  if (name) {
    for (const [pattern, family] of FAMILY_TABLES[slot]) {
      if (pattern.test(name)) return family;
    }
  }
  for (const [pattern, family] of FAMILY_TABLES[slot]) {
    if (meta && pattern.test(meta)) return family;
  }
  return "other";
}
function garmentVisualFamily(item) {
  const slot = diversitySlotOf(item);
  if (!slot) return "";
  return `${SLOT_PREFIX[slot]}:${familyFor(slot, item)}`;
}
var ARCHETYPE_TOKEN_BY_FAMILY = {
  // outerwear
  "outer:tuxedo": "tuxedo",
  "outer:dinner-jacket": "dinner-jacket",
  "outer:blazer": "blazer",
  "outer:coat": "coat",
  "outer:suede-jacket": "jacket",
  "outer:jacket-generic": "jacket",
  "outer:overshirt": "overshirt",
  "outer:cardigan": "cardigan",
  "outer:knitwear": "knitwear",
  "outer:hoodie": "hoodie",
  "outer:bomber": "bomber",
  "outer:vest": "vest",
  "outer:utility": "utility",
  // top
  "top:dress-shirt": "shirt",
  "top:shirt": "shirt",
  "top:blouse": "shirt",
  "top:henley": "tee",
  "top:tee": "tee",
  "top:tank": "tee",
  "top:knit": "knit",
  "top:polo": "polo",
  "top:athletic": "athletic",
  // bottom
  "bottom:tailored": "tailored",
  "bottom:trousers-generic": "tailored",
  "bottom:chino": "chino",
  "bottom:corduroy": "chino",
  "bottom:jean": "jean",
  "bottom:jogger": "jogger",
  "bottom:short": "shorts",
  "bottom:leather": "leather",
  // shoes
  "shoe:tuxedo-shoe": "dress-shoe",
  "shoe:dress-shoe": "dress-shoe",
  "shoe:loafer": "loafer",
  "shoe:moccasin": "moccasin",
  "shoe:sneaker": "sneaker",
  "shoe:boot": "boot",
  "shoe:sandal": "sandal"
};
function garmentArchetypeToken(item) {
  const slot = diversitySlotOf(item);
  if (!slot) return "none";
  const family = garmentVisualFamily(item);
  return ARCHETYPE_TOKEN_BY_FAMILY[family] ?? familyFor(slot, item);
}
var VARIANT_PATTERNS = [
  [/double[\s-]?breasted/, "double-breasted"],
  [/single[\s-]?breasted/, "single-breasted"],
  [/shawl/, "shawl-collar"],
  [/peak/, "peak-lapel"],
  [/notch/, "notch-lapel"],
  [/unstructured|soft[\s-]?shoulder/, "unstructured"],
  [/structured/, "structured"],
  [/quilted/, "quilted"],
  [/pleated/, "pleated"],
  [/double[\s-]?faced|reversible/, "double-faced"],
  [/patch[\s-]?pocket/, "patch-pocket"],
  [/hooded|hood/, "hooded"],
  [/zip/, "zipped"],
  [/buttoned|button[\s-]?up/, "buttoned"]
];
function variantTokensOf(item) {
  const { name, meta } = garmentBlobs(item);
  const blob = name || meta;
  const tokens = [];
  for (const [pattern, token] of VARIANT_PATTERNS) {
    if (!tokens.includes(token) && pattern.test(blob)) tokens.push(token);
  }
  return tokens;
}
var STUDIO_PHRASE_RE = /\b(white|plain|light|light[\s-]grey|light[\s-]gray|grey|gray)\s+background\b|\bstudio(\s+shot)?\b|\bon[\s-]?model\b|\bghost\s+mannequin\b|\bflat[\s-]?lay\b/gi;
function garmentColorBlob(item) {
  const blob = `${text(item?.color)} ${text(item?.name)} ${text(item?.description)}`.toLowerCase();
  return blob.replace(STUDIO_PHRASE_RE, " ").replace(/\s+/g, " ").trim();
}
var COLOR_TOKENS = [
  // [pattern, canonical token, coarse family]
  [/\b(navy|midnight|ink)\b/, "navy", "cool-dark"],
  [/\b(indigo)\b/, "indigo", "cool-dark"],
  [/\b(black|noir)\b/, "black", "dark"],
  [/\b(charcoal|graphite)\b/, "charcoal", "dark"],
  [/\b(dark\s*grey|dark\s*gray|dark\s*brown)\b/, "dark-neutral", "dark"],
  [/\b(grey|gray|greige|slate)\b/, "grey", "mid"],
  [/\b(white|off[\s-]?white)\b/, "white", "light"],
  [/\b(cream|ivory|ecru|milk|off[\s-]?white)\b/, "cream", "light"],
  [/\b(beige|sand|stone|taupe|oat)\b/, "beige", "light"],
  [/\b(light\s*blue|sky|ice)\b/, "light-blue", "light"],
  [/\b(blue|chambray)\b/, "blue", "cool-mid"],
  [/\b(tan|camel|khaki|fawn|cognac)\b/, "tan", "earth"],
  [/\b(brown|chocolate|coffee|espresso|walnut)\b/, "brown", "earth"],
  [/\b(olive|green|forest|sage|mint|bottle)\b/, "green", "earth"],
  [/\b(burgundy|wine|maroon|oxblood)\b/, "burgundy", "warm"],
  [/\b(red|scarlet|crimson)\b/, "red", "warm"],
  [/\b(rust|terracotta|orange)\b/, "rust", "warm"],
  [/\b(pink|lilac|purple|mauve|plum)\b/, "pink", "warm"],
  [/\b(yellow|mustard)\b/, "yellow", "warm"]
];
function colorTokensOf(item) {
  const blob = garmentColorBlob(item);
  const tokens = [];
  for (const [pattern, token] of COLOR_TOKENS) {
    if (tokens.includes(token)) continue;
    if (pattern.test(blob)) tokens.push(token);
  }
  return tokens;
}
function colorFamilyOf(item) {
  const blob = garmentColorBlob(item);
  for (const [pattern, , family] of COLOR_TOKENS) {
    if (pattern.test(blob)) return family;
  }
  return "neutral";
}
var MATERIAL_PATTERNS = [
  [/\bsuede\b/, "suede"],
  [/\bcashmere\b/, "cashmere"],
  [/\bmerino\b/, "merino"],
  [/\b(wool|flannel|tweed|herringbone|harris|tropical)\b/, "wool"],
  [/\blinen\b/, "linen"],
  [/\bdenim\b/, "denim"],
  [/\bcorduroy\b/, "corduroy"],
  [/\bvelvet\b/, "velvet"],
  [/\bsilk\b/, "silk"],
  [/\bleather\b/, "leather"],
  [/\bcotton\b/, "cotton"],
  [/\b(jersey|knit\w*)\b/, "jersey"],
  [/\b(synthetic|polyester|nylon|technical|performance|recycled)\b/, "technical"]
];
function materialsOf(item) {
  const blob = fullBlob(item);
  const out = [];
  for (const [pattern, material] of MATERIAL_PATTERNS) {
    if (!out.includes(material) && pattern.test(blob)) out.push(material);
  }
  return out;
}
var PATTERN_PATTERNS = [
  [/\b(stripes?|striped|pinstripes?|pinstriped)\b/, "stripe"],
  [/\b(check|checked|plaid|tartan|gingham|houndstooth|windowpane)\b/, "check"],
  [/\b(herringbone|tweed|jacquard|textured|texture)\b/, "texture"],
  [/\b(print|printed|graphic|floral|logo|camo|patterned)\b/, "print"]
];
function patternOf(item) {
  const declared = text(item?.pattern).toLowerCase().trim();
  const blob = `${declared} ${text(item?.name)} ${text(item?.description)}`.toLowerCase();
  for (const [pattern, profile] of PATTERN_PATTERNS) {
    if (pattern.test(blob)) return profile;
  }
  if (/solid/.test(blob)) return "solid";
  return "solid";
}
function textureOf(item) {
  const materials = materialsOf(item);
  if (materials.includes("jersey")) return "knit";
  if (materials.includes("suede")) return "suede";
  if (materials.includes("denim")) return "denim";
  if (materials.includes("technical")) return "technical";
  if (materials.includes("wool") || materials.includes("corduroy")) return "textured";
  if (materials.includes("knit")) return "knit";
  return "smooth";
}
var FORMALITY_SCORE = {
  athletic: 0,
  casual: 1,
  "smart-casual": 2,
  formal: 3,
  evening: 4
};
var FORMALITY_BY_FAMILY = {
  // outerwear
  "outer:tuxedo": "evening",
  "outer:suede-jacket": "smart-casual",
  "outer:blazer": "formal",
  "outer:coat": "formal",
  "outer:overshirt": "casual",
  "outer:cardigan": "smart-casual",
  "outer:hoodie": "casual",
  "outer:bomber": "casual",
  "outer:vest": "smart-casual",
  "outer:utility": "casual",
  "outer:knitwear": "smart-casual",
  "outer:jacket-generic": "smart-casual",
  // top
  "top:dress-shirt": "formal",
  "top:polo": "smart-casual",
  "top:knit": "smart-casual",
  "top:henley": "casual",
  "top:blouse": "formal",
  "top:athletic": "athletic",
  "top:tee": "casual",
  "top:tank": "casual",
  "top:shirt": "smart-casual",
  // bottom
  "bottom:short": "casual",
  "bottom:jean": "casual",
  "bottom:corduroy": "smart-casual",
  "bottom:jogger": "athletic",
  "bottom:leather": "casual",
  "bottom:chino": "smart-casual",
  "bottom:tailored": "formal",
  "bottom:trousers-generic": "smart-casual",
  // shoes
  "shoe:tuxedo-shoe": "evening",
  "shoe:moccasin": "smart-casual",
  "shoe:loafer": "smart-casual",
  "shoe:dress-shoe": "formal",
  "shoe:boot": "smart-casual",
  "shoe:sneaker": "casual",
  "shoe:sandal": "casual"
};
function formalityOf(item) {
  const slot = diversitySlotOf(item);
  if (!slot) return "casual";
  const family = garmentVisualFamily(item);
  const tier = FORMALITY_BY_FAMILY[family];
  if (tier) return tier;
  if (slot === "shoes") return "casual";
  return "smart-casual";
}
function identityKeysOf(item) {
  return productIdentityKeys(item);
}
function silhouetteOf(slots) {
  const outer = slots.outerwear;
  const bottom = slots.bottom;
  const top = slots.top;
  const shoe = slots.shoes;
  const has = (entry, families) => Boolean(entry && families.includes(entry.family));
  if (has(outer, ["outer:tuxedo"]) || shoe?.family === "shoe:tuxedo-shoe") return "evening-formal";
  const allAthletic = [outer, top, bottom, shoe].filter(Boolean).every((entry) => entry.formality === "athletic");
  if (allAthletic && [outer, top, bottom, shoe].some(Boolean)) return "sporty";
  if (has(outer, ["outer:blazer", "outer:coat"]) && has(bottom, ["bottom:tailored"])) return "tailored";
  if (has(bottom, ["bottom:jean", "bottom:short", "bottom:jogger"])) return "relaxed";
  if (has(top, ["top:knit"]) || has(outer, ["outer:cardigan", "outer:knitwear"])) return "layered-knit";
  if (has(outer, ["outer:overshirt", "outer:bomber", "outer:utility", "outer:hoodie", "outer:vest"])) return "relaxed";
  return "smart-casual";
}
function buildOutfitFingerprint(items) {
  const slots = {};
  const list2 = Array.isArray(items) ? items : [];
  for (const item of list2) {
    if (!item) continue;
    const slot = diversitySlotOf(item);
    if (!slot) continue;
    if (slots[slot]) continue;
    const family = garmentVisualFamily(item);
    const archetypeToken = garmentArchetypeToken(item);
    const materials = materialsOf(item);
    slots[slot] = {
      slot,
      family,
      archetypeToken,
      itemId: text(item?.id),
      colors: colorTokensOf(item),
      colorFamily: colorFamilyOf(item),
      materials,
      variants: variantTokensOf(item),
      pattern: patternOf(item),
      texture: textureOf(item),
      formality: formalityOf(item),
      formalityScore: FORMALITY_SCORE[formalityOf(item)],
      identityKeys: identityKeysOf(item)
    };
  }
  const archetypeTokens = {};
  for (const slot of DIVERSITY_SLOTS) {
    const entry = slots[slot];
    if (entry) archetypeTokens[slot] = entry.archetypeToken;
  }
  const archetype = DIVERSITY_SLOTS.filter((slot) => archetypeTokens[slot]).map((slot) => archetypeTokens[slot]).join("-");
  const allColors = [];
  const allMaterials = [];
  let formalitySum = 0;
  let formalityCount = 0;
  for (const slot of DIVERSITY_SLOTS) {
    const entry = slots[slot];
    if (!entry) continue;
    for (const color of entry.colors) if (!allColors.includes(color)) allColors.push(color);
    for (const material of entry.materials) if (!allMaterials.includes(material)) allMaterials.push(material);
    formalitySum += entry.formalityScore;
    formalityCount += 1;
  }
  const formalityScore = formalityCount > 0 ? formalitySum / formalityCount : 2;
  const families = DIVERSITY_SLOTS.filter((slot) => slots[slot]).map((slot) => slots[slot].family);
  const colorCounts = /* @__PURE__ */ new Map();
  for (const slot of DIVERSITY_SLOTS) {
    const entry = slots[slot];
    if (!entry) continue;
    colorCounts.set(entry.colorFamily, (colorCounts.get(entry.colorFamily) || 0) + 1);
  }
  let colorFamily = "mixed";
  let best = 0;
  for (const [family, count] of colorCounts) {
    if (count > best) {
      best = count;
      colorFamily = family;
    }
  }
  if (best <= 1 && colorCounts.size > 1) colorFamily = "mixed";
  const patternProfile = dominantProfile(DIVERSITY_SLOTS.map((slot) => slots[slot]).filter(Boolean).map((e) => e.pattern));
  const textureProfile = dominantProfile(DIVERSITY_SLOTS.map((slot) => slots[slot]).filter(Boolean).map((e) => e.texture));
  return {
    slots,
    archetype,
    archetypeTokens,
    silhouette: silhouetteOf(slots),
    formalityScore,
    formality: formalityTierFor(formalityScore),
    colors: allColors,
    colorFamily,
    materials: allMaterials,
    patternProfile,
    textureProfile,
    footwearFamily: slots.shoes?.family ?? "",
    families,
    itemIds: DIVERSITY_SLOTS.map((slot) => slots[slot]?.itemId).filter(Boolean)
  };
}
function dominantProfile(values) {
  if (values.length === 0) return "solid";
  const counts = /* @__PURE__ */ new Map();
  for (const value of values) counts.set(value, (counts.get(value) || 0) + 1);
  let best = "";
  let bestCount = 0;
  for (const [value, count] of counts) {
    if (count > bestCount) {
      best = value;
      bestCount = count;
    }
  }
  return best;
}
function formalityTierFor(score) {
  if (score >= 3.5) return "evening";
  if (score >= 2.5) return "formal";
  if (score >= 1.5) return "smart-casual";
  if (score >= 0.5) return "casual";
  return "athletic";
}
function jaccard(a, b) {
  if (a.length === 0 && b.length === 0) return 0;
  const setA = new Set(a);
  const setB = new Set(b);
  if (setA.size === 0 || setB.size === 0) return 0;
  let intersection = 0;
  for (const value of setA) if (setB.has(value)) intersection += 1;
  const union = setA.size + setB.size - intersection;
  return union === 0 ? 0 : intersection / union;
}
var SILHOUETTE_RELATIONS = {
  "tailored|smart-casual": 0.5,
  "smart-casual|relaxed": 0.5,
  "tailored|relaxed": 0.25,
  "smart-casual|layered-knit": 0.5,
  "layered-knit|relaxed": 0.35,
  "evening-formal|tailored": 0.25,
  "evening-formal|smart-casual": 0.2,
  "sporty|relaxed": 0.2
};
function silhouetteRelation(a, b) {
  if (a === b) return 1;
  const key = [a, b].sort().join("|");
  return SILHOUETTE_RELATIONS[key] ?? 0;
}
function isVisuallyDistinctVariant(a, b) {
  if (a.family !== b.family) return false;
  const variantsDiffer = a.variants.join("|") !== b.variants.join("|") && (a.variants.length > 0 || b.variants.length > 0);
  if (variantsDiffer) return true;
  const colorDiffers = a.colorFamily !== b.colorFamily;
  const materialDiffers = a.materials.length > 0 && b.materials.length > 0 && a.materials.join("|") !== b.materials.join("|");
  return colorDiffers && materialDiffers;
}
function clusterProductOccurrences(occurrences) {
  const parent = occurrences.map((_, index) => index);
  const find = (index) => {
    let current = index;
    while (parent[current] !== current) {
      parent[current] = parent[parent[current]];
      current = parent[current];
    }
    return current;
  };
  const union = (a, b) => {
    const rootA = find(a);
    const rootB = find(b);
    if (rootA !== rootB) parent[rootB] = rootA;
  };
  const owner = /* @__PURE__ */ new Map();
  occurrences.forEach((occurrence, index) => {
    for (const key of occurrence.keys) {
      const previous = owner.get(key);
      if (previous === void 0) owner.set(key, index);
      else union(index, previous);
    }
  });
  const groups = /* @__PURE__ */ new Map();
  occurrences.forEach((_, index) => {
    const root = find(index);
    if (!groups.has(root)) groups.set(root, []);
    groups.get(root).push(index);
  });
  return [...groups.values()];
}
function analyzeDeckProductDiversity(looksItems, opts = {}) {
  const occurrences = [];
  const lookProductKeys = [];
  const anchorKeys = opts.anchorKeys;
  (looksItems || []).forEach((items, look) => {
    const canonicalForLook = [];
    for (const item of Array.isArray(items) ? items : []) {
      if (!item) continue;
      const identity = productIdentityOf(item);
      const anchor = Boolean(anchorKeys && identity.keys.some((key) => anchorKeys.has(key)));
      occurrences.push({
        keys: identity.keys,
        canonicalKey: identity.canonicalKey,
        look,
        anchor,
        slot: diversitySlotOf(item)
      });
      canonicalForLook.push(identity.canonicalKey);
    }
    lookProductKeys.push(canonicalForLook);
  });
  const repeatedProductKeys = /* @__PURE__ */ new Set();
  const repeatedProducts = [];
  const forcedSlots = new Set(opts.forcedSlots || []);
  const forcedRepeatProducts = /* @__PURE__ */ new Set();
  const anchorRepeatProducts = /* @__PURE__ */ new Set();
  let nonAnchorProductReuseCount = 0;
  let illegalProductReuseCount = 0;
  for (const group of clusterProductOccurrences(occurrences)) {
    const looks = new Set(group.map((index) => occurrences[index].look));
    if (looks.size < 2) continue;
    const nonAnchor = group.map((index) => occurrences[index]).filter((occurrence) => !occurrence.anchor);
    if (nonAnchor.length === 0) {
      for (const occurrence of group.map((index) => occurrences[index])) {
        anchorRepeatProducts.add(occurrence.canonicalKey);
      }
      continue;
    }
    nonAnchorProductReuseCount += 1;
    repeatedProductKeys.add(nonAnchor[0].canonicalKey);
    const lookIds = [...new Set(nonAnchor.map((occurrence) => occurrence.look))].sort((a, b) => a - b);
    const slot = nonAnchor[0].slot;
    const legitimate = slot != null && forcedSlots.has(slot);
    if (legitimate) forcedRepeatProducts.add(nonAnchor[0].canonicalKey);
    else illegalProductReuseCount += 1;
    repeatedProducts.push({
      key: nonAnchor[0].canonicalKey,
      slot,
      lookCount: lookIds.length,
      lookIds,
      legitimate
    });
  }
  repeatedProducts.sort((a, b) => String(a.slot).localeCompare(String(b.slot)) || a.key.localeCompare(b.key));
  const canonicalLooks = /* @__PURE__ */ new Map();
  for (const occurrence of occurrences) {
    if (occurrence.anchor || !occurrence.canonicalKey) continue;
    if (!canonicalLooks.has(occurrence.canonicalKey)) canonicalLooks.set(occurrence.canonicalKey, /* @__PURE__ */ new Set());
    canonicalLooks.get(occurrence.canonicalKey).add(occurrence.look);
  }
  const repeatedCanonicalKeys = [...canonicalLooks.entries()].filter(([, looks]) => looks.size > 1).map(([key]) => key);
  const anchorLookCount = new Set(
    occurrences.filter((occurrence) => occurrence.anchor).map((occurrence) => occurrence.look)
  ).size;
  return {
    nonAnchorProductReuseCount,
    repeatedProductKeys: [...repeatedProductKeys],
    repeatedCanonicalKeys,
    repeatedProducts,
    illegalProductReuseCount,
    forcedRepeatProducts: [...forcedRepeatProducts],
    anchorRepeatProducts: [...anchorRepeatProducts],
    anchorLookCount,
    anchorRepetitionCount: Math.max(0, anchorLookCount - 1),
    lookProductKeys
  };
}
function visualSimilarity(a, b, opts = {}) {
  const ignore = new Set(opts.ignoreSlots ?? []);
  const slots = DIVERSITY_SLOTS.filter(
    (slot) => !ignore.has(slot) && (a.slots[slot] || b.slots[slot])
  );
  if (slots.length === 0) return 0;
  let tokenMatches = 0;
  let comparedSlots = 0;
  for (const slot of slots) {
    const left = a.slots[slot];
    const right = b.slots[slot];
    if (!left || !right) continue;
    comparedSlots += 1;
    if (left.archetypeToken !== right.archetypeToken) continue;
    if (left.family === right.family && isVisuallyDistinctVariant(left, right)) {
      continue;
    }
    tokenMatches += 1;
  }
  const archetypeScore = comparedSlots > 0 ? tokenMatches / comparedSlots : 0;
  const leftFamilies = slots.map((slot) => a.slots[slot]?.family).filter(Boolean);
  const rightFamilies = slots.map((slot) => b.slots[slot]?.family).filter(Boolean);
  const familyScore = jaccard(leftFamilies, rightFamilies);
  const silhouetteScore = silhouetteRelation(a.silhouette, b.silhouette);
  const colorScore = jaccard(a.colors, b.colors) * 0.7 + (a.colorFamily !== "mixed" && a.colorFamily === b.colorFamily ? 0.3 : 0);
  const footwearScore = a.footwearFamily && a.footwearFamily === b.footwearFamily ? 1 : 0;
  const materialScore = jaccard(a.materials, b.materials);
  const patternScore = a.patternProfile === b.patternProfile ? 1 : 0;
  const formalityDelta = Math.abs(a.formalityScore - b.formalityScore);
  const formalityScore = Math.max(0, 1 - formalityDelta / 2);
  const score = DIVERSITY_WEIGHTS.archetype * archetypeScore + DIVERSITY_WEIGHTS.families * familyScore + DIVERSITY_WEIGHTS.silhouette * silhouetteScore + DIVERSITY_WEIGHTS.color * colorScore + DIVERSITY_WEIGHTS.footwear * footwearScore + DIVERSITY_WEIGHTS.material * materialScore + DIVERSITY_WEIGHTS.pattern * patternScore + DIVERSITY_WEIGHTS.formality * formalityScore;
  return Math.max(0, Math.min(1, score));
}
function assessInventoryDiversity(items, opts = {}) {
  const ignore = new Set(opts.ignoreSlots ?? []);
  const families = /* @__PURE__ */ new Map();
  const tokens = /* @__PURE__ */ new Map();
  const products = /* @__PURE__ */ new Map();
  const seenSlots = /* @__PURE__ */ new Set();
  for (const item of Array.isArray(items) ? items : []) {
    if (!item) continue;
    const slot = diversitySlotOf(item);
    if (!slot || ignore.has(slot)) continue;
    seenSlots.add(slot);
    if (!products.has(slot)) products.set(slot, []);
    products.get(slot).push(item);
    const family = garmentVisualFamily(item);
    if (family) {
      if (!families.has(slot)) families.set(slot, /* @__PURE__ */ new Set());
      families.get(slot).add(family);
    }
    const token = garmentArchetypeToken(item);
    if (token && token !== "none") {
      if (!tokens.has(slot)) tokens.set(slot, /* @__PURE__ */ new Set());
      tokens.get(slot).add(token);
    }
  }
  const activeSlots = opts.activeSlots ? opts.activeSlots.filter((slot) => !ignore.has(slot)) : DIVERSITY_SLOTS.filter((slot) => seenSlots.has(slot) && !ignore.has(slot));
  const familiesPerSlot = {};
  const tokensPerSlot = {};
  const uniqueProductsPerSlot = {};
  for (const slot of activeSlots) {
    familiesPerSlot[slot] = families.get(slot)?.size ?? 0;
    tokensPerSlot[slot] = tokens.get(slot)?.size ?? 0;
    uniqueProductsPerSlot[slot] = countDistinctProducts(products.get(slot) ?? []);
  }
  const formulaSlots = activeSlots.filter((slot) => FORMULA_SLOTS.includes(slot));
  const tokenCounts = formulaSlots.map((slot) => tokensPerSlot[slot] ?? 0);
  const maxDistinctArchetypes = tokenCounts.length ? Math.min(...tokenCounts) : 0;
  const conceptCapacity = tokenCounts.length ? tokenCounts.reduce((product, count) => product * Math.max(count, 1), 1) : 0;
  const target = opts.target ?? 3;
  const supportsTargetConcepts = conceptCapacity >= target;
  const productCounts = activeSlots.map((slot) => uniqueProductsPerSlot[slot] ?? 0);
  const maxDistinctProducts = productCounts.length ? Math.min(...productCounts) : 0;
  const supportsUniqueProducts = activeSlots.length > 0 && maxDistinctProducts >= target;
  const homogeneous = formulaSlots.length > 0 && tokenCounts.every((count) => count <= 1);
  const forcedSharedFamilies = activeSlots.filter(
    (slot) => (familiesPerSlot[slot] ?? 0) === 1
  ).length;
  const slotCapacity = activeSlots.map((slot) => ({
    slot,
    uniqueProducts: uniqueProductsPerSlot[slot] ?? 0,
    forced: (uniqueProductsPerSlot[slot] ?? 0) < target,
    anchor: ignore.has(slot)
  }));
  const forcedRepeatSlots = slotCapacity.filter((entry) => entry.forced).map((entry) => entry.slot);
  const strictSlots = slotCapacity.filter((entry) => !entry.forced && !entry.anchor).map((entry) => entry.slot);
  return {
    familiesPerSlot,
    tokensPerSlot,
    uniqueProductsPerSlot,
    maxDistinctProducts,
    supportsUniqueProducts,
    activeSlots,
    slotCapacity,
    forcedRepeatSlots,
    strictSlots,
    forcedSharedFamilies,
    maxDistinctArchetypes,
    conceptCapacity: Math.min(conceptCapacity, 1e5),
    supportsTargetConcepts,
    homogeneous,
    repetitionAvoidable: !homogeneous
  };
}
function identityKeySets(item, keysOf) {
  if (keysOf) return keysOf(item).filter(Boolean);
  return identityKeysOf(item);
}
var QUALITY_WEIGHT = 1e3;
var MMR_QUALITY_BAND = 0.12;
var MMR_SIMILARITY_WEIGHT = 1;
var MMR_RECENCY_WEIGHT = 0.5;
function assembleDiverseDeck(opts) {
  const target = Math.max(1, Math.floor(opts.target ?? 3));
  const maxFillAttempts = opts.maxFillAttempts ?? 48;
  const ignoreSlots = opts.anchorSlot ? [opts.anchorSlot] : [];
  const debug = opts.debug;
  const pool = [];
  const clamp013 = (value, fallback) => typeof value === "number" && Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : fallback;
  const pushCandidate = (outfit, source) => {
    const items = opts.itemsOf(outfit) || [];
    const quality = opts.qualityOf ? opts.qualityOf(outfit) : 0.9;
    const candidate = {
      outfit,
      fingerprint: buildOutfitFingerprint(items),
      quality: clamp013(quality, 0.9),
      recency: opts.recencyOf ? clamp013(opts.recencyOf(outfit), 0) : 0,
      source,
      index: pool.length
    };
    pool.push(candidate);
    return candidate;
  };
  for (const candidate of opts.candidates || []) pushCandidate(candidate, "edge");
  const refsCache = /* @__PURE__ */ new Map();
  const strictKeysCache = /* @__PURE__ */ new Map();
  const itemRefsFor = (candidate) => {
    let refs = refsCache.get(candidate.index);
    if (!refs) {
      refs = (opts.itemsOf(candidate.outfit) || []).map((item) => ({ keys: identityKeySets(item, opts.keysOf), slot: diversitySlotOf(item) }));
      refsCache.set(candidate.index, refs);
    }
    return refs;
  };
  const explicitActiveSlots = opts.activeSlots && opts.activeSlots.length ? DIVERSITY_SLOTS.filter((slot) => opts.activeSlots.includes(slot)).filter((slot) => !ignoreSlots.includes(slot)) : null;
  const capacity = opts.capacity;
  const capacityActiveSlots = capacity?.activeSlots && capacity.activeSlots.length ? DIVERSITY_SLOTS.filter((slot) => capacity.activeSlots.includes(slot)).filter((slot) => !ignoreSlots.includes(slot)) : null;
  const deriveActiveSlots = () => {
    if (explicitActiveSlots) return explicitActiveSlots;
    if (capacityActiveSlots) return capacityActiveSlots;
    const seen = /* @__PURE__ */ new Set();
    for (const candidate of pool) {
      for (const ref of itemRefsFor(candidate)) {
        if (ref.slot) seen.add(ref.slot);
      }
    }
    for (const item of opts.poolItems || []) {
      const slot = diversitySlotOf(item);
      if (slot) seen.add(slot);
    }
    return DIVERSITY_SLOTS.filter((slot) => seen.has(slot) && !ignoreSlots.includes(slot));
  };
  const similarityCache = /* @__PURE__ */ new Map();
  const similarityOf = (a, b) => {
    const key = a.index < b.index ? `${a.index}:${b.index}` : `${b.index}:${a.index}`;
    const cached = similarityCache.get(key);
    if (cached !== void 0) return cached;
    const score = visualSimilarity(a.fingerprint, b.fingerprint, { ignoreSlots });
    similarityCache.set(key, score);
    return score;
  };
  const computeInventory = () => assessInventoryDiversity(
    [
      ...pool.flatMap((member) => opts.itemsOf(member.outfit) || []),
      ...opts.poolItems || []
    ],
    { ignoreSlots, activeSlots: deriveActiveSlots(), target }
  );
  const reuseCache = /* @__PURE__ */ new Map();
  const resolveForcedSlots = (assessment) => {
    const computed = new Set(assessment.forcedRepeatSlots);
    if (!capacity) return computed;
    const forced = new Set(capacity.forcedRepeatSlots || []);
    const strict = new Set(capacity.strictSlots || []);
    for (const slot of [...computed, ...forced, ...strict]) {
      if (forced.has(slot)) computed.add(slot);
      else if (strict.has(slot)) computed.delete(slot);
    }
    return computed;
  };
  const applyCapacity = (assessment) => {
    const forced = resolveForcedSlots(assessment);
    return {
      ...assessment,
      forcedRepeatSlots: assessment.activeSlots.filter((slot) => forced.has(slot)),
      strictSlots: assessment.activeSlots.filter((slot) => !forced.has(slot))
    };
  };
  let inventory = applyCapacity(computeInventory());
  let strictSlots = new Set(inventory.strictSlots);
  let forcedSlots = resolveForcedSlots(inventory);
  const refreshInventory = () => {
    inventory = applyCapacity(computeInventory());
    strictSlots = new Set(inventory.strictSlots);
    forcedSlots = resolveForcedSlots(inventory);
    reuseCache.clear();
  };
  if (debug) {
    debug("ACTIVE_DIVERSITY_SLOTS", {
      activeSlots: inventory.activeSlots,
      target,
      uniqueProductsPerSlot: inventory.uniqueProductsPerSlot
    });
    debug(inventory.supportsUniqueProducts ? "UNIQUE_INVENTORY_SUPPORTED" : "UNIQUE_INVENTORY_UNSUPPORTED", {
      strictSlots: inventory.strictSlots,
      forcedRepeatSlots: inventory.forcedRepeatSlots,
      maxDistinctProducts: inventory.maxDistinctProducts
    });
  }
  const strictIdentityOf = (item) => {
    const identity = productIdentityOf(item);
    const keys = [];
    if (identity.imageKey) keys.push(identity.imageKey);
    if (identity.idKey) keys.push(identity.idKey);
    return keys;
  };
  const looksCarrySameProducts = (a, b) => {
    const keysOf = (candidate) => {
      let keys = strictKeysCache.get(candidate.index);
      if (!keys) {
        keys = (opts.itemsOf(candidate.outfit) || []).filter((item) => !opts.anchorKeys || !identityKeySets(item, opts.keysOf).some((key) => opts.anchorKeys.has(key))).map(strictIdentityOf);
        strictKeysCache.set(candidate.index, keys);
      }
      return keys;
    };
    const leftKeys = keysOf(a), rightKeys = keysOf(b);
    if (leftKeys.length === 0 || leftKeys.length !== rightKeys.length) return false;
    const used = new Array(rightKeys.length).fill(false);
    const match = (index) => {
      if (index === leftKeys.length) return true;
      for (let j = 0; j < rightKeys.length; j++) {
        if (used[j]) continue;
        const sharesStrictKey = leftKeys[index].length > 0 && rightKeys[j].length > 0 && leftKeys[index].some((key) => rightKeys[j].includes(key));
        if (!sharesStrictKey) continue;
        used[j] = true;
        if (match(index + 1)) return true;
        used[j] = false;
      }
      return false;
    };
    return match(0);
  };
  const reuseOf = (a, b) => {
    const key = a.index < b.index ? `${a.index}:${b.index}` : `${b.index}:${a.index}`;
    const cached = reuseCache.get(key);
    if (cached) return cached;
    const result = { violations: [], forced: [], anchor: [] };
    const refsA = itemRefsFor(a);
    const refsB = itemRefsFor(b);
    for (const refA of refsA) {
      for (const refB of refsB) {
        const shared = refA.keys.filter((identityKey) => refB.keys.includes(identityKey));
        if (!shared.length) continue;
        const anchorKeys = opts.anchorKeys;
        if (anchorKeys && shared.some((identityKey) => anchorKeys.has(identityKey))) {
          result.anchor.push({ keys: shared });
          continue;
        }
        const slot = refA.slot ?? refB.slot;
        if (slot && forcedSlots.has(slot)) {
          result.forced.push({ slot, keys: shared });
        } else {
          result.violations.push({ slot, keys: shared });
        }
      }
    }
    if (looksCarrySameProducts(a, b)) {
      result.violations.push({ slot: null, keys: ["duplicate_look"] });
    }
    reuseCache.set(key, result);
    return result;
  };
  const qualityBand = opts.qualityBand ?? MMR_QUALITY_BAND;
  const similarityWeight = opts.similarityWeight ?? MMR_SIMILARITY_WEIGHT;
  const recencyWeight = opts.recencyWeight ?? MMR_RECENCY_WEIGHT;
  const priorityCache = /* @__PURE__ */ new Map();
  const priorityFor = (indices) => {
    const key = indices.join(":");
    let priority = priorityCache.get(key);
    if (priority === void 0) {
      priority = opts.priorityOfDeck?.(indices.map((i) => pool[i].outfit)) ?? 0;
      priorityCache.set(key, priority);
    }
    return priority;
  };
  const scoreCache = /* @__PURE__ */ new Map();
  const deckScore = (indices) => {
    const key = indices.join(":");
    const cached = scoreCache.get(key);
    if (cached !== void 0) return cached;
    const members = indices.map((index) => pool[index]);
    const qualitySum = members.reduce((sum, member) => sum + member.quality, 0);
    const qualityBucket = Math.round(qualitySum / qualityBand) * qualityBand;
    const archetypes = new Set(members.map((member) => member.fingerprint.archetype));
    const bonus = DIVERSITY_BONUS_PER_ARCHETYPE * 0.35 * (archetypes.size - 1);
    let similarityPenalty = 0;
    let pairs = 0;
    for (let i = 0; i < members.length; i++) {
      for (let j = i + 1; j < members.length; j++) {
        similarityPenalty += similarityOf(members[i], members[j]);
        pairs += 1;
      }
    }
    const meanSimilarity = pairs > 0 ? similarityPenalty / pairs : 0;
    const meanRecency = members.length > 0 ? members.reduce((sum, member) => sum + member.recency, 0) / members.length : 0;
    const variety = bonus - similarityWeight * 0.5 * meanSimilarity - recencyWeight * meanRecency;
    const priority = priorityFor(indices);
    const score = (typeof priority === "number" && Number.isFinite(priority) ? Math.max(0, Math.floor(priority)) : 0) * (target * QUALITY_WEIGHT + 10) + qualityBucket * QUALITY_WEIGHT + variety;
    scoreCache.set(key, score);
    return score;
  };
  const compareSelections = (a, b) => {
    const first = priorityFor(a);
    const second = priorityFor(b);
    if (Array.isArray(first) && Array.isArray(second)) {
      for (let i = 0; i < Math.max(first.length, second.length); i++) {
        const delta = (first[i] || 0) - (second[i] || 0);
        if (delta) return delta;
      }
    }
    return deckScore(a) - deckScore(b);
  };
  const loggedRejections = /* @__PURE__ */ new Set();
  const logRejection = (a, b, reuse) => {
    if (!debug) return;
    if (loggedRejections.size >= 12) return;
    const key = `${Math.min(a.index, b.index)}:${Math.max(a.index, b.index)}:shared_product`;
    if (loggedRejections.has(key)) return;
    loggedRejections.add(key);
    debug("PRODUCT_REUSE_REJECT", {
      pair: [a.index + 1, b.index + 1],
      slots: reuse.violations.map((entry) => entry.slot),
      products: reuse.violations.map((entry) => entry.keys[0]),
      archetypes: [a.fingerprint.archetype, b.fingerprint.archetype]
    });
  };
  const logAllowedReuse = (a, b, reuse) => {
    if (!debug) return;
    for (const entry of reuse.forced) {
      debug("FORCED_SLOT_REUSE", {
        slot: entry.slot,
        product: entry.keys[0],
        pair: [a.index + 1, b.index + 1],
        uniqueProductsInSlot: inventory.uniqueProductsPerSlot[entry.slot ?? "top"] ?? 0
      });
    }
    for (const entry of reuse.anchor) {
      debug("ANCHOR_REUSE_ALLOWED", {
        product: entry.keys[0],
        pair: [a.index + 1, b.index + 1]
      });
    }
  };
  const combinationAllowed = (indices) => {
    for (let i = 0; i < indices.length; i++) {
      for (let j = i + 1; j < indices.length; j++) {
        const a = pool[indices[i]];
        const b = pool[indices[j]];
        const reuse = reuseOf(a, b);
        if (reuse.violations.length > 0) {
          logRejection(a, b, reuse);
          return false;
        }
        logAllowedReuse(a, b, reuse);
      }
    }
    return true;
  };
  const MAX_COMBINATIONS = 25e3;
  let stepsLeft = Math.max(1, Math.floor(opts.maxSearchSteps ?? Number.MAX_SAFE_INTEGER));
  const selectCombination = (size) => {
    const n = pool.length;
    if (n < size || size < 1) return null;
    const startedAt = Date.now();
    let best = null;
    const consider = (indices) => {
      if (stepsLeft-- <= 0 || !combinationAllowed(indices)) return;
      if (!best || compareSelections(indices, best) > 0) {
        best = indices;
      }
    };
    let combinations = 1;
    for (let i = 0; i < size; i++) {
      combinations = combinations * (n - i) / (i + 1);
      if (combinations > MAX_COMBINATIONS) break;
    }
    if (combinations <= Math.min(MAX_COMBINATIONS, stepsLeft)) {
      const current = [];
      const walk = (start) => {
        if (current.length === size) {
          consider([...current]);
          return;
        }
        for (let i = start; i < n && stepsLeft > 0; i++) {
          current.push(i);
          walk(i + 1);
          current.pop();
        }
      };
      walk(0);
      if (debug) debug("SELECT_COMBINATION", { size, pool: n, exhaustive: true, ms: Date.now() - startedAt });
      return best;
    }
    let bestSeed = null;
    for (let seed = 0; seed < n && stepsLeft > 0; seed++) {
      const chosen2 = [seed];
      let failed = false;
      while (chosen2.length < size) {
        let bestAdd = -1;
        for (let i = 0; i < n && stepsLeft > 0; i++) {
          if (chosen2.includes(i)) continue;
          stepsLeft--;
          const trial = [...chosen2, i];
          if (!combinationAllowed(trial)) continue;
          if (bestAdd < 0 || compareSelections(trial, [...chosen2, bestAdd]) > 0) {
            bestAdd = i;
          }
        }
        if (bestAdd < 0) {
          failed = true;
          break;
        }
        chosen2.push(bestAdd);
      }
      if (failed) continue;
      if (!bestSeed || compareSelections(chosen2, bestSeed) > 0) {
        bestSeed = chosen2;
      }
    }
    if (bestSeed) best = bestSeed;
    if (debug) debug("SELECT_COMBINATION", { size, pool: n, exhaustive: false, ms: Date.now() - startedAt });
    return best;
  };
  const selectBestUpTo = (maxSize) => {
    for (let size = Math.min(maxSize, pool.length); size >= 1; size--) {
      const selection2 = selectCombination(size);
      if (selection2) return selection2;
    }
    return null;
  };
  const seedUsedKeys = (indices) => {
    for (const index of indices) {
      const member = pool[index];
      for (const item of opts.itemsOf(member.outfit) || []) {
        for (const key of identityKeySets(item, opts.keysOf)) {
          if (opts.anchorKeys && opts.anchorKeys.has(key)) continue;
          opts.usedKeys.add(key);
        }
        const slot = diversitySlotOf(item);
        if (slot && ignoreSlots.includes(slot)) continue;
        const family = garmentVisualFamily(item);
        if (family) opts.usedKeys.add(`__vf:${family}`);
      }
    }
  };
  let selection = selectCombination(target);
  let fillsUsed = 0;
  if (opts.buildFill) {
    if (selection) seedUsedKeys(selection);
    let staleAttempts = 0;
    const patience = 5;
    while (fillsUsed < maxFillAttempts) {
      if (selection && staleAttempts >= patience) break;
      const attempt = fillsUsed;
      fillsUsed += 1;
      const fill = opts.buildFill(opts.usedKeys, attempt);
      if (!fill) {
        if (!selection) staleAttempts = 0;
        continue;
      }
      const candidate = pushCandidate(fill, "fill");
      refreshInventory();
      if (debug) {
        debug(`replacement selected: ${candidate.fingerprint.archetype}`, {
          attempt,
          archetype: candidate.fingerprint.archetype,
          products: (opts.itemsOf(fill) || []).map((item) => canonicalProductKey(item))
        });
      }
      const next = selectCombination(target);
      if (!next) {
        staleAttempts = 0;
        continue;
      }
      if (!selection || compareSelections(next, selection) > 0) {
        selection = next;
        staleAttempts = 0;
      } else {
        staleAttempts += 1;
      }
    }
  }
  let relaxed = false;
  let relaxReason = null;
  if (!selection) {
    selection = selectBestUpTo(target);
    if (selection && selection.length < target) {
      relaxed = true;
      relaxReason = forcedSlots.size > 0 ? "insufficient_unique_products" : "insufficient_unique_candidates";
    }
  }
  if (debug) {
    for (const member of pool) {
      debug(`candidate ${member.index + 1} archetype: ${member.fingerprint.archetype}`, {
        source: member.source,
        quality: member.quality,
        products: (opts.itemsOf(member.outfit) || []).map((item) => canonicalProductKey(item))
      });
    }
  }
  if (!selection) {
    return {
      deck: [],
      relaxed: true,
      relaxReason: relaxReason ?? "no_candidates",
      archetypes: [],
      similarities: [],
      inventory,
      fillsUsed,
      productDiversity: analyzeDeckProductDiversity([], { anchorKeys: opts.anchorKeys }),
      hasQualitySignal: false,
      qualityRange: { min: 0, max: 0 }
    };
  }
  const chosen = selection.map((index) => pool[index]).sort((a, b) => b.quality - a.quality || a.index - b.index);
  const similarities = [];
  for (let i = 0; i < chosen.length; i++) {
    for (let j = i + 1; j < chosen.length; j++) {
      similarities.push({ a: i, b: j, score: similarityOf(chosen[i], chosen[j]) });
    }
  }
  const productDiversity = analyzeDeckProductDiversity(
    chosen.map((member) => opts.itemsOf(member.outfit) || []),
    { anchorKeys: opts.anchorKeys, forcedSlots: inventory.forcedRepeatSlots }
  );
  const forbiddenRepeats = productDiversity.repeatedProducts.filter(
    (entry) => !(entry.slot && forcedSlots.has(entry.slot))
  );
  if (forbiddenRepeats.length > 0) {
    relaxed = true;
    relaxReason = "repeated_products_in_strict_slot";
  } else if (productDiversity.repeatedProducts.length > 0) {
    relaxed = true;
    relaxReason = "insufficient_unique_products";
  }
  const qualities = pool.map((member) => member.quality);
  const qualityMin = qualities.length ? Math.min(...qualities) : 0;
  const qualityMax = qualities.length ? Math.max(...qualities) : 0;
  const hasQualitySignal = qualities.length > 0 && qualityMax - qualityMin > 1e-9;
  if (debug) {
    for (const entry of similarities) {
      debug(`similarity ${entry.a + 1}<->${entry.b + 1} = ${entry.score.toFixed(2)}`, {
        archetypes: [chosen[entry.a].fingerprint.archetype, chosen[entry.b].fingerprint.archetype]
      });
    }
    debug("DIVERSE_DECK_SELECTED", {
      deckSize: chosen.length,
      target,
      archetypes: chosen.map((member) => member.fingerprint.archetype),
      nonAnchorProductReuseCount: productDiversity.nonAnchorProductReuseCount,
      repeatedProductKeys: productDiversity.repeatedProductKeys,
      repeatedSlots: productDiversity.repeatedProducts.map((entry) => entry.slot),
      anchorRepetitionCount: productDiversity.anchorRepetitionCount,
      activeSlots: inventory.activeSlots,
      forcedRepeatSlots: inventory.forcedRepeatSlots,
      relaxed,
      relaxReason,
      hasQualitySignal
    });
    if (chosen.length < target) {
      debug("UNIQUE_CANDIDATE_SUBSET_ONLY", {
        deckSize: chosen.length,
        target,
        reason: "no fully unique combination of the requested size"
      });
    }
  }
  seedUsedKeys(selection);
  return {
    deck: chosen.map((member) => member.outfit),
    relaxed,
    relaxReason,
    archetypes: chosen.map((member) => member.fingerprint.archetype),
    similarities,
    inventory,
    fillsUsed,
    productDiversity,
    hasQualitySignal,
    qualityRange: { min: qualityMin, max: qualityMax }
  };
}

// src/domain/stylingProfiles.ts
var STYLING_PROFILES = {
  minimalist: {
    preferred: [
      "shirt",
      "fine-knit",
      "cardigan",
      "overshirt",
      "tailored-trouser",
      "chino",
      "minimal-sneaker",
      "chelsea",
      "dress-boot"
    ],
    textures: ["smooth", "fine-knit"],
    restrained: true
  },
  old_money: {
    preferred: ["polo", "dress-shirt", "cardigan", "blazer", "chino", "tailored-trouser", "loafer"],
    textures: ["suede", "knit", "fine-knit", "wool"],
    restrained: true
  },
  semi_classic: {
    preferred: [
      "dress-shirt",
      "shirt",
      "blazer",
      "tailored-trouser",
      "derby",
      "oxford",
      "monk",
      "dress-boot"
    ],
    textures: ["smooth", "wool"],
    restrained: false
  },
  casual: {
    preferred: [
      "t-shirt",
      "polo",
      "shirt",
      "overshirt",
      "jacket",
      "bomber",
      "jeans",
      "chino",
      "sneaker",
      "minimal-sneaker"
    ],
    textures: ["denim", "cotton"],
    restrained: false
  }
};

// src/domain/garmentDescriptor.ts
function describeGarment(item) {
  const materialText = typeof item.material === "string" ? item.material : Array.isArray(item.material) ? item.material.filter((value) => typeof value === "string").join(" ") : "";
  const text3 = normalizeGarmentText(
    [item.name, item.type, item.description, materialText].filter(Boolean).join(" ")
  );
  const slot = String(item.macroCategory || item.category || "unknown");
  const rules = slot === "shoes" ? [
    ["hiking-boot", /\b(hiking|trekking|mountaineer|snow boot|combat|rugged|lug[ -]sole)\b/],
    ["oxford", /\boxfords?\b/],
    ["derby", /\bderb(?:y|ies)\b/],
    ["monk", /\bmonk/],
    ["espadrille", /\bespadrilles?\b/],
    ["sandal", /\b(sandals?|slides?|flip[ -]flops?)\b/],
    ["loafer", /\b(loafers?|penny|horsebit)\b/],
    ["driver", /\b(driving|drivers?)\b/],
    ["chelsea", /\bchelsea\b/],
    ["dress-boot", /\b(dress|sleek|slim)[ -].*boots?\b/],
    ["sport-sneaker", /\b(running|training|trail|performance|athletic)\b/],
    [
      "minimal-sneaker",
      /\b(minimal|plain|clean|low[ -]top|leather)\b.*\b(sneakers?|trainers?)\b/
    ],
    ["sneaker", /\b(sneakers?|trainers?)\b/],
    ["boot", /\bboots?\b/]
  ] : slot === "bottom" ? [
    [
      "athletic-trouser",
      /\b(joggers?|track[ -]?(pants?|trousers?)|training|sweatpants?|drawstring|athletic|sports? pants?)\b/
    ],
    ["cargo", /\bcargo\b/],
    ["jeans", /\bjeans|denim\b/],
    ["chino", /\bchinos?\b/],
    [
      "tailored-trouser",
      /\b(tailored|suit|tuxedo|dress|formal|pleated|darted|pinstriped|flannel|smart)\b/
    ],
    // Knit construction alone does not prove sports detailing. Keep it
    // distinct from tailoring unless the catalog explicitly says tailored.
    ["knit-trouser", /\b(interlock|jersey|knitted|knit)\b/],
    ["trouser", /\b(trousers?|pants?|slacks?)\b/]
  ] : slot === "outerwear" ? [
    [
      "blazer",
      /\b(blazers?|suit jacket|tailored jacket|sports? coat|dinner jacket|tuxedo)\b/
    ],
    ["cardigan", /\bcardigans?\b/],
    ["overshirt", /\b(overshirts?|shackets?)\b/],
    ["hoodie", /\bhoodies?\b/],
    ["coat", /\b(coats?|peacoats?|overcoats?|trench|parka|puffer)\b/],
    ["bomber", /\bbomber\b/],
    ["pullover", /\b(sweaters?|pullovers?|jumpers?|knitwear)\b/],
    ["jacket", /\b(jackets?|trucker)\b/]
  ] : [
    ["t-shirt", /\b(t[ -]?shirts?|tees?)\b/],
    ["polo", /\bpolos?\b/],
    ["field-shirt", /\b(field|utility|western|military|ripstop)\b.*\bshirts?\b/],
    ["dress-shirt", /\b(dress|formal|poplin|oxford)\b.*\bshirts?\b/],
    ["shirt", /\bshirts?\b/],
    ["fine-knit", /\b(fine|lightweight|thin|gauge|merino)\b.*\b(knit|sweater|pullover)/],
    ["sweater", /\b(sweaters?|pullovers?|jumpers?)\b/]
  ];
  const subtype = rules.find(([, r]) => r.test(text3))?.[0] || "unknown";
  const athletic = /\b(athletic|training|track[ -]?pants?|joggers?|sweatpants?|workout|sportswear)\b/.test(text3) || subtype === "sport-sneaker" || subtype === "athletic-trouser";
  const rugged = subtype === "hiking-boot" || /\b(rugged|lug[ -]sole|hiking|trekking|combat)\b/.test(text3) || slot === "shoes" && /\bwaterproof\b/.test(text3) && /\b(grip|chunky|traction|tread)\b/.test(text3);
  const verifiedComfortLoafer = slot === "shoes" && (/^(?:shop_)?geox-u66mac/i.test(String(item.id || "")) || /^flextride plus\b/i.test(item.name || ""));
  const casualSole = slot === "shoes" && (verifiedComfortLoafer || /\b(chunky|thick|platform|lug|commando|track|cup)[ -]+(?:rubber[ -]+)?(?:out)?sole(?:s|d)?\b/.test(text3) || /\bsneaker[ -]+soles?\b/.test(text3));
  const structured = subtype === "blazer" || subtype === "tailored-trouser" || subtype === "dress-shirt";
  const dressLoafer = subtype === "loafer" && !casualSole && (/\b(structured|blake|welt|calfskin|polished|apron|saddle|penny|tassel|horsebit|leather\s+out?soles?)\b/.test(
    text3
  ) || /\bleather\b/.test(text3) && /\b(dress|formal)\b/.test(text3) && !/\b(suede|canvas|fabric|knit|sneaker|sport|sports|casual|boat|moccasin|driver)\b/.test(
    text3
  ));
  const formality = athletic ? "athletic" : [
    "oxford",
    "derby",
    "monk",
    "dress-boot",
    "blazer",
    "tailored-trouser",
    "dress-shirt"
  ].includes(subtype) || dressLoafer ? "formal" : ["loafer", "chelsea", "chino", "fine-knit", "cardigan"].includes(subtype) ? "smart" : subtype === "unknown" ? "unknown" : "casual";
  const volume = /\b(slim|skinny|fitted|tight)\b/.test(text3) ? "slim" : /\b(wide|oversized|baggy|boxy|relaxed)\b/.test(text3) ? "wide" : /\b(regular|straight)\b/.test(text3) ? "regular" : "unknown";
  const bulk = /\b(chunky|heavyweight|bulky|thick|cable[ -]knit|padded|shearling|puffer|down jacket)\b/.test(
    text3
  ) ? "heavy" : /\b(lightweight|fine|thin|unlined|linen|poplin|gauze)\b/.test(text3) ? "light" : "unknown";
  const collar = /\b(turtle|roll|high)[ -]?neck\b/.test(text3) ? "high" : /\b(shirts?|polos?)\b/.test(text3) ? "shirt" : /\b(v[ -]?neck|open[ -]?neck)\b/.test(text3) ? "open" : /\bcrew[ -]?neck\b/.test(text3) ? "crew" : "unknown";
  const length = /\b(cropped|crop)\b/.test(text3) ? "cropped" : /\b(longline|long[ -]length|long[ -]hem)\b/.test(text3) ? "long" : /\b(regular[ -]length|hip[ -]length)\b/.test(text3) ? "regular" : "unknown";
  const closure = /\b(pullover|pull[ -]over)\b/.test(text3) ? "pullover" : ["blazer", "cardigan", "overshirt", "jacket", "coat", "bomber"].includes(subtype) ? "openable" : "unknown";
  return {
    slot,
    subtype,
    length,
    closure,
    color: colorFamilyOf(item),
    materials: materialsOf(item),
    texture: /\b(knit|knitted|jersey|suede|denim|technical|wool|corduroy|smooth|ribbed|tweed)\b/.test(
      `${text3} ${String(item.material || "")}`
    ) ? textureOf(item) : "unknown",
    pattern: typeof item.pattern === "string" && item.pattern.trim() && !/^(unknown|other)$/i.test(item.pattern.trim()) ? patternOf(item) : /\b(solid|plain|unpatterned|striped?|checks?|checked|plaid|graphic|logo|floral|paisley|print(?:ed)?|camo|jacquard|polka)\b/.test(
      text3
    ) ? patternOf(item) : "unknown",
    formality,
    volume,
    bulk,
    collar,
    rugged,
    casualSole,
    athletic,
    structured,
    insulated: /\b(puffer|down jacket|down filling|insulated|shearling)\b/.test(text3),
    evening: /\b(tuxedo|dinner jacket|dinner suit|opera pump|patent)\b/.test(text3),
    evidence: [
      ...verifiedComfortLoafer ? ["catalog:geox-u66mac-comfort-sole"] : [],
      ...subtype !== "unknown" ? [`text:${subtype}`] : [],
      ...bulk !== "unknown" ? [`text:${bulk}`] : [],
      ...volume !== "unknown" ? [`text:${volume}`] : []
    ]
  };
}

// src/domain/garmentContext.ts
function garmentContextReason(item, context) {
  const shop = item.isShopItem === true || String(item.id || "").startsWith("shop_");
  if (context.sourceMode === "wardrobe" && shop) return "source_requires_wardrobe";
  if (context.sourceMode === "catalog" && !shop) return "source_requires_catalog";
  const d = describeGarment(item);
  if (context.dressCode === "smart" && /\b(work|office|business)\b/.test(context.occasion) && ["classic", "old_money", "semi_classic", "business_casual"].includes(context.style)) {
    if (d.slot === "top" && d.subtype === "t-shirt") return "work_requires_smart_base";
    const text3 = normalizeGarmentText(`${item.name || ""} ${item.description || ""}`);
    const cleanWovenJacket = d.subtype === "jacket" && /\b(wool|linen)\b/.test(text3) && !/\b(biker|trucker|racer|denim|leather|suede|utility|military|athletic|running)\b/.test(text3);
    if (d.slot === "outerwear" && !["blazer", "cardigan", "coat"].includes(d.subtype) && !cleanWovenJacket)
      return "work_requires_smart_layer";
  }
  const explicit = typeof item.color === "string" && !/^(unknown|neutral)$/i.test(item.color) ? item.color : item.name || "";
  const colorEvidence = " " + normalizeGarmentText(explicit).replace(/[^\p{L}]+/gu, " ").trim() + " ";
  if (context.preferences?.avoidColors?.some((color) => {
    const value = normalizeGarmentText(color).replace(/[^\p{L}]+/gu, " ").trim();
    return !!value && colorEvidence.includes(" " + value + " ");
  }))
    return "excluded_color";
  if (d.pattern !== "unknown" && context.preferences?.dislikedPatterns?.some(
    (pattern) => /stripe|check|plaid|graphic|logo|floral|camo|plain|solid|paisley|jacquard/i.test(pattern) && patternOf({ pattern }) === d.pattern
  ))
    return "excluded_pattern";
  if (context.dressCode === "formal") {
    const baseText = normalizeGarmentText(`${item.name || ""} ${item.description || ""}`);
    if (d.slot === "top" && SHORT_SLEEVE_RE.test(baseText))
      return "formal_requires_long_sleeve_dress_base";
    if (d.slot === "top" && (/\b(henley|grandad|collarless|camp[ -]collar|cuban[ -]collar)\b/.test(baseText) || /\b(knit|knitted|jersey)\b/.test(baseText) || /\bmerino\b/.test(baseText) && /\bbutton[ -]through\b/.test(baseText) && !/\b(woven|dress|formal|poplin|oxford)\b/.test(baseText))) return "formal_requires_dress_base";
    if (["top", "outerwear", "bottom"].includes(d.slot) && /\b(corduroy|denim|fleece|sweatshirt)\b/i.test(`${item.name || ""} ${item.description || ""}`))
      return "formal_rejects_casual_fabric";
    if (d.slot === "outerwear" && d.subtype !== "blazer") return "formal_requires_structured_layer";
    if (d.slot === "top" && (!["shirt", "dress-shirt"].includes(d.subtype) || d.texture === "knit" && d.subtype !== "dress-shirt"))
      return "formal_requires_dress_base";
    if (d.slot === "bottom" && d.subtype !== "tailored-trouser")
      return "formal_requires_tailored_bottom";
    if (d.slot === "shoes" && (!["oxford", "derby", "monk", "dress-boot", "chelsea", "loafer"].includes(d.subtype) || d.rugged || d.casualSole || d.subtype === "loafer" && d.formality !== "formal" || d.athletic || /\b(hybrids?|cup[ -]?soles?)\b/.test(baseText)))
      return "formal_requires_dress_footwear";
  }
  const outdoor = /\b(hik(?:e|ing)|trek(?:king)?|trail|mountain|snow)\b/i.test(
    `${context.activity} ${context.request}`
  );
  const severe = context.weather?.temp != null && context.weather.temp < 5 && /snow|ice/i.test(context.weather.condition || "");
  if (context.dressCode === "smart" && d.slot === "shoes" && d.rugged && !outdoor && !severe)
    return "rugged_footwear_conflicts_with_smart_context";
  if (context.dressCode === "athletic") {
    if (d.slot === "shoes" && !d.athletic) return "workout_requires_athletic_footwear";
    if (d.slot === "bottom" && d.structured) return "workout_rejects_restrictive_tailoring";
  }
  return null;
}
function officeRefinementPenalty(item, context, d = describeGarment(item)) {
  if (context.style !== "old_money" || context.dressCode !== "smart" || !/\b(work|office|business)\b/.test(context.occasion)) return 0;
  const text3 = normalizeGarmentText(`${item.name || ""} ${item.description || ""}`);
  if (d.slot === "shoes" && d.casualSole) return 0.35;
  if (d.slot === "outerwear" && (/\b(leisure|utility|flight|racer|bomber)\b/.test(text3) || /\b(technical|nylon|shell)\b/.test(text3) && /\b(zip|waterproof|windproof|performance)\b/.test(text3))) return 0.3;
  if (d.slot === "top" && d.pattern === "stripe" && /\b(rugby|rugger|sporty|athletic|jersey|bold|wide|contrast|block|candy)\b/.test(text3)) return 0.25;
  return 0;
}
function compareUnbudgetedValue(a, b) {
  if (!a.isShopItem || !b.isShopItem || a.isWardrobe === true || b.isWardrobe === true || a.macroCategory !== b.macroCategory) return 0;
  const currency2 = typeof a.currency === "string" ? a.currency.trim().toUpperCase() : "";
  if (!/^[A-Z]{3}$/.test(currency2) || currency2 !== String(b.currency || "").trim().toUpperCase()) return 0;
  const first = Number(a.price), second = Number(b.price);
  return Number.isFinite(first) && first > 0 && Number.isFinite(second) && second > 0 ? first - second : 0;
}
function scoreGarmentForContext(item, context) {
  const d = describeGarment(item);
  let score = 0.5;
  const evening = /black[ -]?tie|tuxedo|gala|смокинг/i.test(
    `${context.occasion} ${context.request}`
  );
  if (d.evening && !evening) return 0.05;
  if (d.subtype === "unknown") return score;
  if (context.dressCode === "formal")
    return d.structured || ["oxford", "derby", "monk", "dress-boot"].includes(d.subtype) ? 1 : 0.8;
  if (context.dressCode === "athletic") return d.athletic ? 0.95 : 0.25;
  if (context.dressCode === "smart") {
    score = [
      "blazer",
      "cardigan",
      "dress-shirt",
      "shirt",
      "fine-knit",
      "tailored-trouser",
      "chino",
      "loafer",
      "chelsea",
      "dress-boot",
      "oxford",
      "derby"
    ].includes(d.subtype) ? 0.9 : 0.45;
    if (d.rugged) score = 0.1;
    if (/\b(work|office|business)\b/.test(context.occasion) && d.slot === "outerwear") {
      if (["blazer", "cardigan"].includes(d.subtype)) score = 1;
      if (/\b(biker|trucker|racer|denim)\b/.test(normalizeGarmentText(`${item.name || ""} ${item.description || ""}`))) score = 0.15;
    }
  } else {
    score = [
      "overshirt",
      "jacket",
      "bomber",
      "t-shirt",
      "polo",
      "shirt",
      "chino",
      "jeans",
      "sneaker",
      "minimal-sneaker",
      "driver"
    ].includes(d.subtype) ? 0.9 : 0.6;
  }
  const travel = /\b(travel|trip|sightseeing)\b/.test(`${context.occasion} ${context.request}`);
  const businessTravel = /\b(business|meeting|conference)\b/.test(context.request);
  if (travel && !businessTravel) {
    if (["oxford", "monk", "blazer"].includes(d.subtype)) score -= 0.3;
    if (d.evening) score = 0.05;
    if (d.bulk === "heavy" && (!context.weather || (context.weather.temp ?? 15) > 12)) score -= 0.2;
    if (["cardigan", "overshirt", "chino", "minimal-sneaker"].includes(d.subtype)) score += 0.2;
    const text3 = normalizeGarmentText(`${item.name || ""} ${item.description || ""}`);
    if (d.slot === "shoes" && /\b(cushioned|flexible|walking|padded insole|comfort)\b/.test(text3)) score += 0.2;
  }
  const profile = STYLING_PROFILES[context.style];
  if (profile) {
    score = score * 0.75 + (profile.preferred.includes(d.subtype) ? 1 : 0.4) * 0.25;
    if (profile.textures.includes(d.texture)) score += 0.025;
  }
  if (context.style === "minimalist" && d.pattern !== "solid" && d.pattern !== "unknown")
    score -= 0.08;
  if (!context.weather && d.bulk === "heavy") score -= 0.1;
  if (d.bulk === "light" && ["cardigan", "blazer", "overshirt"].includes(d.subtype)) score += 0.05;
  return Math.max(0, Math.min(1, score - officeRefinementPenalty(item, context, d)));
}

// src/domain/outfitVisualSignature.ts
function outfitVisualSignature(items) {
  return "sq1:" + ["top", "outerwear", "bottom", "shoes"].map((slot) => {
    const item = items.find((i) => i.macroCategory === slot);
    if (!item) return `${slot}:missing`;
    const d = describeGarment(item);
    return [slot, d.subtype, d.color, d.volume, d.texture, d.formality, d.length].join(":");
  }).join("|");
}
function recentLookSimilarity(items, recent = []) {
  if (items.filter((item) => describeGarment(item).subtype !== "unknown").length < 2) return 0;
  const signature = outfitVisualSignature(items), parts = signature.slice(4).split("|");
  let maximum = 0;
  for (const previous of recent.slice(-36)) {
    if (!previous.startsWith("sq1:")) continue;
    const other = previous.slice(4).split("|");
    let equal = 0, known = 0;
    for (let i = 0; i < parts.length; i++) {
      const a = parts[i].split(":"), b = (other[i] || "").split(":");
      for (let j = 1; j < a.length; j++)
        if (a[j] !== "unknown" && b[j] && b[j] !== "unknown") {
          known++;
          if (a[j] === b[j]) equal++;
        }
    }
    const similarity = known >= 8 ? equal / known : 0;
    maximum = Math.max(maximum, similarity);
  }
  return maximum;
}

// src/domain/layerCompatibility.ts
function layerCompatibility(top, layer, context) {
  const a = describeGarment(top), b = describeGarment(layer);
  const reasons = [], evidence = [];
  let score = 0.7;
  if (a.bulk === "heavy" && b.bulk === "heavy") reasons.push("competing_heavy_layers");
  if (a.volume === "wide" && b.volume === "slim") reasons.push("wide_base_under_fitted_layer");
  if (context.dressCode === "formal" && b.subtype !== "blazer")
    reasons.push("formal_requires_structured_layer");
  if (context.dressCode === "formal" && !["shirt", "dress-shirt"].includes(a.subtype))
    reasons.push("formal_requires_dress_base");
  if (context.dressCode === "athletic" && b.structured)
    reasons.push("structured_layer_restricts_workout");
  const topText = normalizeGarmentText(
    [top.name, top.type, top.description].filter(Boolean).join(" ")
  );
  if (b.structured && a.athletic) reasons.push("tailored_layer_over_athletic_base");
  if (b.subtype === "blazer" && /\bhood/.test(topText))
    reasons.push("hood_under_tailored_blazer");
  if (b.subtype === "blazer" && a.volume !== "slim" && (a.bulk === "heavy" || a.subtype === "sweater" && a.volume === "wide"))
    reasons.push("bulky_base_under_tailored_blazer");
  if (b.subtype === "blazer" && a.subtype === "sweater" && a.bulk === "unknown" && a.volume === "unknown") {
    score -= 0.1;
    evidence.push("unverified_knit_bulk_under_blazer");
  }
  if (a.formality === "formal" && b.athletic)
    reasons.push("athletic_outerwear_over_formal_base");
  if (a.bulk === "light" && b.closure === "openable") {
    score += 0.15;
    evidence.push("light_base_under_openable_layer");
  }
  if (["shirt", "dress-shirt", "t-shirt", "polo"].includes(a.subtype) && ["blazer", "cardigan", "overshirt", "jacket", "coat", "bomber"].includes(b.subtype)) {
    score += 0.1;
    evidence.push("base_and_outer_layer_roles");
  }
  if (a.subtype === "fine-knit" && b.subtype === "blazer") {
    score += 0.1;
    evidence.push("fine_knit_under_blazer");
  }
  if (a.bulk === "heavy" && b.bulk === "light" && !b.structured) {
    score -= 0.1;
    evidence.push("heavy_base_under_light_unstructured_layer");
  }
  if (a.length === "long" && b.length === "cropped") {
    score -= 0.15;
    evidence.push("long_base_below_cropped_layer");
  }
  if (a.collar === "shirt" && b.collar === "high") {
    score -= 0.2;
    evidence.push("competing_necklines");
  }
  if (a.bulk === "heavy" && b.subtype === "pullover") {
    score -= 0.3;
    evidence.push("bulky_base_under_pullover");
  }
  if ((context.weather === null || context.weather.temp != null && context.weather.temp > 22) && b.bulk === "heavy") {
    score -= 0.15;
    evidence.push("heavy_layer_without_cold_need");
  }
  return {
    valid: reasons.length === 0,
    score: reasons.length ? 0 : Math.round(Math.max(0, Math.min(1, score)) * 1e3) / 1e3,
    reasons,
    evidence,
    confidence: a.subtype === "unknown" || b.subtype === "unknown" ? "low" : a.bulk === "unknown" && b.bulk === "unknown" ? "medium" : "high"
  };
}

// src/domain/colorHarmony.ts
var CORE_NEUTRALS = /* @__PURE__ */ new Set(["white", "black", "grey", "charcoal", "cream"]);
var WORKHORSE_NEUTRALS = /* @__PURE__ */ new Set(["navy", "brown", "tan", "beige"]);
var SATURATED = /* @__PURE__ */ new Set(["red", "rust", "pink", "yellow", "green", "light-blue", "blue"]);
var LIGHT_TOKENS = /* @__PURE__ */ new Set(["white", "cream", "beige", "light-blue", "grey"]);
var DARK_TOKENS = /* @__PURE__ */ new Set(["black", "charcoal", "navy", "dark-neutral", "brown", "burgundy"]);
var CLASSIC_PAIRS = [
  ["navy", "brown"],
  ["grey", "navy"],
  ["black", "white"],
  ["white", "navy"],
  ["beige", "brown"],
  ["cream", "beige"]
];
var clamp01 = (value) => Math.max(0, Math.min(1, value));
function scoreColorHarmony(garments) {
  const reasons = [];
  const evidence = [];
  const list2 = Array.isArray(garments) ? garments : [];
  const withColor = list2.filter((g) => g && g.colors.length > 0);
  const unknownCount = list2.length - withColor.length;
  const uncertain = unknownCount >= 2 || withColor.length < 2;
  let score = 0.8;
  const tokenSet = /* @__PURE__ */ new Set();
  for (const garment of withColor) for (const token of garment.colors) tokenSet.add(token);
  const tokens = [...tokenSet];
  const accents = tokens.filter((token) => SATURATED.has(token));
  const chromatics = tokens.filter(
    (token) => !CORE_NEUTRALS.has(token) && !WORKHORSE_NEUTRALS.has(token)
  );
  if (accents.length === 0) {
    score += 0.05;
    evidence.push("restrained_palette_no_loud_accents");
  } else if (accents.length === 1) {
    const supported = tokens.some(
      (token) => CORE_NEUTRALS.has(token) || WORKHORSE_NEUTRALS.has(token)
    );
    score += supported ? 0.1 : 0.05;
    evidence.push(`single_accent_${accents[0]}_on_neutral_base`);
  } else {
    score -= 0.15 + 0.2 * (accents.length - 2);
    reasons.push("overloaded_accent_combination");
  }
  const chromaticFamilies = new Set(
    withColor.flatMap(
      (g) => g.colors.filter((t) => !CORE_NEUTRALS.has(t) && !WORKHORSE_NEUTRALS.has(t)).map(() => g.colorFamily)
    )
  );
  if (withColor.length > 0 && chromatics.length > 0 && chromaticFamilies.size <= 1) {
    score += 0.1;
    evidence.push("tonal_palette_single_chromatic_story");
  }
  const hasLight = tokens.some((token) => LIGHT_TOKENS.has(token));
  const hasDark = tokens.some((token) => DARK_TOKENS.has(token));
  if (hasLight && hasDark) {
    score += 0.05;
    evidence.push("balanced_light_dark_contrast");
  }
  const nonNeutral = tokens.filter((token) => !CORE_NEUTRALS.has(token));
  if (tokens.length > 0 && nonNeutral.length <= 1) {
    score += 0.05;
    evidence.push("monochromatic_discipline");
  }
  if (CLASSIC_PAIRS.some(([a, b]) => tokenSet.has(a) && tokenSet.has(b))) {
    score += 0.05;
    evidence.push("classic_neutral_pairing");
  }
  const prints = list2.filter((g) => g && g.pattern === "print").length;
  const stripes = list2.filter((g) => g && (g.pattern === "stripe" || g.pattern === "check")).length;
  if (prints >= 2 || prints >= 1 && stripes >= 1) {
    score -= 0.1;
    reasons.push("clashing_prints_break_harmony");
  }
  if (uncertain) {
    score = Math.max(0.45, Math.min(0.75, score));
    evidence.push("uncertain_palette_insufficient_color_evidence");
  }
  return { score: clamp01(Math.round(score * 1e3) / 1e3), reasons, evidence, uncertain };
}

// src/domain/outfitCoherence.ts
var LAYER_REGISTER_CLASH = /* @__PURE__ */ new Set([
  "tailored_layer_over_athletic_base",
  "hood_under_tailored_blazer",
  "bulky_base_under_tailored_blazer",
  "athletic_outerwear_over_formal_base"
]);
function wholeOutfitRegisterClash(items, context) {
  if (!items || items.length === 0) return null;
  const descriptors = items.map(describeGarment);
  const evidenceFor = (slot) => {
    const index = descriptors.findIndex((d) => d.slot === slot);
    return index >= 0 ? items[index] : void 0;
  };
  const top = evidenceFor("top"), layer = evidenceFor("outerwear"), bottomIndex = descriptors.findIndex((d) => d.slot === "bottom"), shoeIndex = descriptors.findIndex((d) => d.slot === "shoes");
  if (top && layer) {
    const pair = layerCompatibility(top, layer, context);
    const hit = pair.reasons.find((reason) => LAYER_REGISTER_CLASH.has(reason));
    if (hit) return hit;
  }
  const visiblePattern = (item) => ["check", "stripe", "print"].includes(patternOf(item)) || /\b(chevron|geometric|paisley|jacquard|pattern(?:ed)?)\b/i.test(`${item.name || ""} ${item.description || ""} ${item.pattern || ""}`);
  if (context.dressCode === "formal" && top && visiblePattern(top) && [layer, bottomIndex >= 0 ? items[bottomIndex] : void 0].some((item) => item && visiblePattern(item)))
    return "formal_patterned_tailoring_requires_quiet_base";
  if (context.dressCode === "formal" && layer && bottomIndex >= 0) {
    const layerPattern = patternOf(layer), bottomPattern = patternOf(items[bottomIndex]);
    if (visiblePattern(layer) && visiblePattern(items[bottomIndex]) && layerPattern !== bottomPattern)
      return "formal_tailoring_patterns_conflict";
    if (layerPattern === "stripe" && bottomPattern !== "stripe")
      return "formal_striped_layer_requires_matching_trousers";
  }
  const refined = ["semi_classic", "classic", "old_money", "business_casual"].includes(context.style) || context.dressCode === "smart" || context.dressCode === "formal";
  if (refined && top && layer) {
    const a = describeGarment(top), b = describeGarment(layer);
    if (a.subtype === "field-shirt" && b.subtype === "blazer")
      return "utility_base_with_tailored_blazer";
  }
  if (refined && bottomIndex >= 0) {
    const outer = descriptors.find((d) => d.slot === "outerwear");
    const outerText = layer ? `${layer.name || ""} ${layer.description || ""}`.toLowerCase() : "";
    if (outer?.bulk === "heavy" && /\bbiker\b/.test(outerText) && descriptors[bottomIndex].structured)
      return "bulky_biker_with_refined_tailoring";
    if (/\b(work|office|business)\b/.test(context.occasion) && /\b(biker|trucker|racer)\b/.test(outerText) && shoeIndex >= 0 && descriptors[shoeIndex].formality === "formal")
      return "workwear_jacket_with_office_dress_footwear";
  }
  if (bottomIndex >= 0 && shoeIndex >= 0) {
    if (descriptors[bottomIndex].athletic && descriptors[shoeIndex].formality === "formal")
      return "formal_footwear_with_athletic_bottom";
    if (descriptors[bottomIndex].subtype === "knit-trouser" && descriptors[shoeIndex].formality === "formal")
      return "dress_footwear_with_unstructured_knit_bottom";
  }
  const refinedWork = ["classic", "old_money", "semi_classic"].includes(context.style) && /\b(work|office|business)\b/.test(context.occasion);
  if (refinedWork) {
    const shoeDesc = shoeIndex >= 0 ? descriptors[shoeIndex] : void 0;
    if (shoeDesc && (shoeDesc.athletic || shoeDesc.formality === "athletic" || ["sneaker", "sport-sneaker", "minimal-sneaker"].includes(shoeDesc.subtype)))
      return "classic_work_requires_non_athletic_footwear";
    if (bottomIndex >= 0 && descriptors[bottomIndex].athletic)
      return "classic_work_requires_non_athletic_bottom";
    const vacationCue = (item) => !!item && /\b(resort|vacation|holiday|seaside|beach|tropical|hawaiian|camp[ -]?collar)\b/i.test(
      `${item.name || ""} ${item.description || ""}`
    );
    if (vacationCue(layer) || bottomIndex >= 0 && vacationCue(items[bottomIndex]))
      return "classic_work_rejects_vacation_pieces";
  }
  if (context.dressCode === "formal" && ["old_money", "classic"].includes(context.style)) {
    if (top && patternOf(top) === "check")
      return "formal_checked_base_breaks_dress_shirt_line";
    const layerText = layer ? `${layer.name || ""} ${layer.description || ""}`.toLowerCase() : "";
    if (layer && /\b(leisure|unstructured|casual|jersey|knit|sweatshirt|hooded|soft[ -]?shoulder)\b/.test(layerText))
      return "formal_leisure_layer_breaks_structured_tailoring";
  }
  return null;
}
function outfitCoherence(items, context) {
  const descriptors = items.map(describeGarment), bySlot = (s) => descriptors.find((d) => d.slot === s);
  const top = items.find((i) => i.macroCategory === "top"), layer = items.find((i) => i.macroCategory === "outerwear");
  const reasons = [], evidence = [];
  const dimensions = {
    layer: 0.7,
    dressCode: 0.7,
    palette: 0.7,
    proportion: 0.7,
    texture: 0.7,
    footwear: 0.7,
    style: 0.7
  };
  if (top && layer) {
    const pair = layerCompatibility(top, layer, context);
    dimensions.layer = pair.score;
    reasons.push(...pair.reasons);
    evidence.push(...pair.evidence);
  }
  const bottom = bySlot("bottom"), shoe = bySlot("shoes");
  if (bySlot("outerwear")?.insulated && shoe && ["espadrille", "sandal", "driver"].includes(shoe.subtype))
    reasons.push("insulated_layer_with_summer_footwear");
  for (const item of items) {
    const reason = garmentContextReason(item, context);
    if (reason) reasons.push(reason);
  }
  if (context.dressCode === "formal")
    dimensions.dressCode = reasons.some((r) => r.startsWith("formal_")) ? 0 : 1;
  if (context.dressCode !== "formal") {
    dimensions.dressCode = items.reduce((sum, item) => sum + scoreGarmentForContext(item, context), 0) / Math.max(1, items.length);
    if (descriptors.some((d) => d.evening) && !/black[ -]?tie|tuxedo|gala|смокинг/i.test(`${context.occasion} ${context.request}`)) {
      dimensions.dressCode = Math.min(dimensions.dressCode, 0.2);
      evidence.push("evening_tailoring_overdresses_request");
    }
  }
  if (bottom && shoe) {
    if (bottom.structured && ["oxford", "derby", "monk", "loafer", "chelsea", "dress-boot"].includes(shoe.subtype)) {
      dimensions.footwear = 1;
      evidence.push("dress_footwear_with_tailored_bottom");
    }
    if (context.dressCode === "casual") {
      if (["minimal-sneaker", "sneaker"].includes(shoe.subtype) && ["chino", "jeans", "trouser"].includes(bottom.subtype)) {
        dimensions.footwear = 1;
        evidence.push("casual_footwear_with_casual_bottom");
      }
      if (["oxford", "monk"].includes(shoe.subtype) && bySlot("outerwear")?.subtype !== "blazer") {
        dimensions.footwear = 0.45;
        evidence.push("formal_footwear_overdresses_casual_layers");
      }
    }
    if (bottom.structured && shoe.rugged) dimensions.footwear = 0.15;
    if (bottom.structured && shoe.casualSole && ["smart", "formal"].includes(context.dressCode)) {
      dimensions.footwear = Math.min(dimensions.footwear, 0.35);
      evidence.push("casual_sole_breaks_tailored_footwear_line");
    }
    if (bottom.volume === "wide" && shoe.subtype === "driver") {
      dimensions.proportion = 0.45;
      evidence.push("wide_hem_over_delicate_driver");
    }
    if (shoe.subtype === "sport-sneaker" && bySlot("outerwear") && ["blazer", "coat"].includes(bySlot("outerwear").subtype)) {
      dimensions.footwear = Math.min(dimensions.footwear, 0.4);
      evidence.push("sport_sneaker_with_tailored_layer");
    }
  }
  const volumes = descriptors.filter((d) => d.volume === "wide").length;
  if (volumes >= 3) {
    dimensions.proportion = 0.5;
    evidence.push("multiple_high_volume_pieces");
  }
  const outer = bySlot("outerwear");
  if (outer?.structured && bottom?.structured && ["smart", "formal"].includes(context.dressCode)) {
    if (outer.volume === "slim" && bottom.volume === "wide" || outer.volume === "wide" && bottom.volume === "slim") {
      dimensions.proportion = Math.min(dimensions.proportion, 0.45);
      evidence.push("tailoring_volume_contrast");
    } else if (outer.volume !== "unknown" && outer.volume === bottom.volume && volumes < 3) {
      dimensions.proportion = 0.9;
      evidence.push("consistent_tailoring_volume");
    }
    if (outer.materials.some((material) => ["wool", "linen", "cotton"].includes(material) && bottom.materials.includes(material))) {
      dimensions.texture = 0.9;
      evidence.push("compatible_tailoring_fabrics");
    }
  }
  const heavy = descriptors.some((d) => d.bulk === "heavy"), airy = descriptors.some((d) => d.materials.includes("linen"));
  if (heavy && airy) {
    dimensions.texture = 0.4;
    evidence.push("heavy_and_airy_texture_contrast");
  }
  const harmony = scoreColorHarmony(
    items.map((item, index) => ({
      colors: colorTokensOf(item),
      colorFamily: colorFamilyOf(item),
      pattern: descriptors[index] ? descriptors[index].pattern : "unknown"
    }))
  );
  dimensions.palette = harmony.score;
  for (const reason of harmony.reasons) evidence.push(`palette_${reason}`);
  evidence.push(...harmony.evidence);
  const colors = new Set(descriptors.map((d) => d.color).filter((c) => c !== "unknown"));
  if (context.style === "minimalist") {
    const noisy = descriptors.filter((d) => /graphic|logo|floral|camo/.test(d.pattern)).length;
    const patternUnknown = descriptors.some((d) => d.pattern === "unknown");
    dimensions.style = noisy ? 0.35 : patternUnknown ? 0.7 : 0.9;
    if (colors.size <= 2 && colors.size > 0) {
      dimensions.style = noisy ? 0.35 : patternUnknown ? 0.8 : 1;
      evidence.push("restrained_minimalist_palette");
    }
  } else if (context.style === "old_money") {
    const classic = descriptors.filter(
      (d) => [
        "blazer",
        "cardigan",
        "polo",
        "shirt",
        "dress-shirt",
        "chino",
        "tailored-trouser",
        "loafer"
      ].includes(d.subtype)
    ).length;
    dimensions.style = 0.5 + classic * 0.125;
  } else if (context.style === "semi_classic")
    dimensions.style = descriptors.filter((d) => d.structured).length >= 2 ? 0.95 : 0.65;
  else if (context.style === "casual")
    dimensions.style = descriptors.some(
      (d) => ["overshirt", "jeans", "sneaker", "minimal-sneaker"].includes(d.subtype)
    ) ? 0.95 : 0.7;
  dimensions.style = (dimensions.style + items.reduce((sum, item) => sum + scoreGarmentForContext(item, context), 0) / Math.max(1, items.length)) / 2;
  const officePenalty = items.reduce((total, item, i) => total + officeRefinementPenalty(item, context, descriptors[i]), 0);
  if (officePenalty > 0) {
    dimensions.style = Math.min(dimensions.style, Math.max(0.3, 1 - officePenalty));
    evidence.push("casual_construction_weakens_office_refinement");
  }
  if (context.dressCode === "smart" && /\b(work|office|business)\b/.test(context.occasion) && top && layer && ["stripe", "check", "print"].includes(bySlot("top").pattern) && ["stripe", "check", "print"].includes(outer.pattern)) {
    dimensions.palette = Math.min(dimensions.palette, 0.55);
    evidence.push("competing_office_upper_patterns");
  }
  const registerClash = wholeOutfitRegisterClash(items, context);
  if (registerClash) reasons.push(registerClash);
  const valid = !reasons.length;
  return {
    valid,
    score: valid ? Object.values(dimensions).reduce((s, n) => s + n, 0) / 7 : 0,
    reasons: [...new Set(reasons)],
    evidence: [...new Set(evidence)],
    dimensions
  };
}

// src/domain/validatedQuartets.ts
function validatedQuartets(items, context, accepts, options = {}) {
  const required = options.requiredIds || [], maximum = options.maximum ?? 64, limit = options.searchLimit ?? 6e3;
  const reserveLimit = Math.min(256, maximum * Math.min(4, Math.max(1, options.reserveMultiplier ?? 4)));
  const perSlot = Math.min(32, Math.max(3, options.perSlot ?? 24));
  const slots = ["top", "outerwear", "bottom", "shoes"];
  const pools = slots.map((slot) => {
    const all = items.filter(
      (item) => item.macroCategory === slot && !garmentContextReason(item, context)
    ), locked = all.filter((item) => required.includes(item.id));
    return locked.length ? locked : all.map((item, index) => ({ item, index, score: scoreGarmentForContext(item, context) })).sort((a, b) => b.score - a.score || a.index - b.index).slice(0, perSlot).map((entry) => entry.item);
  });
  if (pools.some((pool) => !pool.length)) return [];
  const pairs = [];
  for (const top of pools[0])
    for (const layer of pools[1]) {
      const pair = layerCompatibility(top, layer, context);
      if (pair.valid) pairs.push({ items: [top, layer], score: pair.score, ordinal: pairs.length });
    }
  pairs.sort((a, b) => b.score - a.score || a.ordinal - b.ordinal);
  const usedTop = /* @__PURE__ */ new Set(), usedLayer = /* @__PURE__ */ new Set();
  const novel = pairs.filter((pair) => {
    const t = pair.items[0].id, l = pair.items[1].id;
    if (usedTop.has(t) && usedLayer.has(l)) return false;
    usedTop.add(t);
    usedLayer.add(l);
    return true;
  });
  const beam = [...novel, ...pairs.filter((p) => !novel.includes(p))].slice(
    0,
    Math.min(96, pairs.length)
  );
  const result = [];
  const signatures = /* @__PURE__ */ new Set();
  const identities = new Map(items.map((item) => [item, productIdentityKeys(item)]));
  let examined = 0;
  for (let step = 0; step < pools[2].length * pools[3].length && examined < limit; step++) {
    for (let p = 0; p < beam.length && examined < limit; p++) {
      examined++;
      const parts = [
        ...beam[p].items,
        pools[2][(step + p) % pools[2].length],
        pools[3][(Math.floor(step / pools[2].length) + p) % pools[3].length]
      ];
      const ids = new Set(parts.map((i) => i.id));
      if (ids.size !== 4 || required.some((id) => !ids.has(id))) continue;
      const keys = /* @__PURE__ */ new Set();
      let repeated = false;
      for (const part of parts) {
        const own = identities.get(part);
        if (own.some((k) => keys.has(k))) repeated = true;
        own.forEach((k) => keys.add(k));
      }
      if (repeated) continue;
      const signature = [...ids].sort().join("|");
      if (signatures.has(signature)) continue;
      signatures.add(signature);
      if (!accepts(parts)) continue;
      const quality = outfitCoherence(parts, context);
      if (!quality.valid) continue;
      result.push({ id: "", items: parts, score: quality.score, order: result.length });
      if (result.length >= reserveLimit) break;
    }
    if (result.length >= reserveLimit) break;
  }
  result.sort((a, b) => b.score - a.score || a.order - b.order);
  const reserve = [], seen = /* @__PURE__ */ new Set();
  for (const candidate of result) {
    const companions = candidate.items.filter((i) => !required.includes(i.id));
    if (companions.some((i) => !seen.has(i.id))) {
      reserve.push(candidate);
      companions.forEach((i) => seen.add(i.id));
    }
    if (reserve.length >= maximum) break;
  }
  const chosen = [...reserve, ...result.filter((r) => !reserve.includes(r))].slice(0, maximum);
  return chosen.map((c, i) => ({ id: `look${String(i + 1).padStart(3, "0")}`, items: c.items }));
}

// src/domain/outfitComposition.ts
var LAYERED_SLOTS = ["outerwear", "top", "bottom", "shoes"];
function resolveOutfitComposition(_style, _weather, _prompt) {
  return { layered: true, activeSlots: [...LAYERED_SLOTS] };
}
function resolveStylistWeather(prompt, weather) {
  const text3 = prompt || "";
  const explicit = text3.match(/(-?\d{1,2}(?:\.\d+)?)\s*(?:°\s*C|C\b|degrees?\s*(?:C|celsius)|celsius)\b/i);
  if (explicit) return { temp: Number(explicit[1]), condition: "requested temperature" };
  if (/\b(freezing|snow|winter|sub[ -]?zero)\b|мороз|зим[ауы]/i.test(text3)) return { condition: "requested winter season" };
  if (/\bsummer\b|лет[оа]/i.test(text3)) return { condition: "requested summer season" };
  return weather || void 0;
}
function needsLayering(style, weather, prompt) {
  return resolveOutfitComposition(style, weather, prompt).layered;
}

// src/domain/styleInference.ts
var MATERIAL_WORDS = [
  "linen",
  "wool",
  "cashmere",
  "merino",
  "silk",
  "satin",
  "cotton",
  "poplin",
  "oxford",
  "denim",
  "leather",
  "suede",
  "corduroy",
  "tweed",
  "velvet",
  "cashmere blend",
  "wool blend",
  "canvas",
  "nylon",
  "polyester",
  "mesh",
  "fleece",
  "terry",
  "jersey",
  "knit",
  "cashmere-blend",
  "wool-blend"
];
var PATTERN_WORDS = [
  "striped",
  "stripes",
  "checked",
  "check",
  "plaid",
  "houndstooth",
  "pinstripe",
  "graphic",
  "logo",
  "print",
  "printed",
  "tie-dye",
  "washed",
  "distressed",
  "ripped",
  "embroidered",
  "floral",
  "camo",
  "animal print",
  "monogram",
  "colorblock",
  "color-block"
];
var NEUTRAL_COLOR_WORDS = [
  "beige",
  "cream",
  "ivory",
  "ecru",
  "stone",
  "sand",
  "camel",
  "tan",
  "khaki",
  "brown",
  "chocolate",
  "taupe",
  "white",
  "off-white",
  "oatmeal",
  "oat",
  "black",
  "charcoal",
  "grey",
  "gray",
  "navy",
  "forest",
  "olive",
  "burgundy"
];
var BOLD_COLOR_WORDS = [
  "neon",
  "pink",
  "fuchsia",
  "magenta",
  "lime",
  "yellow",
  "orange",
  "turquoise",
  "aqua",
  "metallic",
  "silver",
  "gold",
  "iridescent",
  "holographic",
  "rhinestone",
  "glitter",
  "sequin"
];
function kw(word, weight) {
  return [new RegExp(`\\b${word.replace(/[-/\\^$*+?.()|[\]{}]/g, "\\$&")}\\b`, "i"), weight];
}
var STYLE_SIGNALS = {
  old_money: [
    kw("cashmere", 4),
    kw("merino", 3),
    kw("wool", 3),
    kw("linen", 3),
    kw("silk", 2),
    kw("cotton", 1),
    kw("blazer", 4),
    kw("loafer", 5),
    kw("loafers", 5),
    kw("oxford", 3),
    kw("knit polo", 4),
    kw("polo", 2),
    kw("cardigan", 3),
    kw("chino", 3),
    kw("chinos", 3),
    kw("trouser", 3),
    kw("trousers", 3),
    kw("tailored", 3),
    kw("pleated", 3),
    kw("pinstripe", 2),
    kw("houndstooth", 3),
    kw("herringbone", 3),
    kw("tweed", 3),
    kw("flannel", 3),
    kw("suit", 3),
    kw("suiting", 3),
    kw("knit sweater", 3),
    kw("pullover", 2),
    kw("turtleneck", 3),
    kw("relaxed-fit", 1),
    kw("classic", 2),
    kw("regular-fit", 1),
    kw("suit trousers", 3),
    kw("linen suit", 2),
    kw("cream", 2),
    kw("camel", 3),
    kw("navy", 2),
    kw("beige", 2),
    kw("burgundy", 2),
    kw("forest", 2),
    kw("tan", 2),
    kw("ralph lauren", 5),
    kw("brunello", 5),
    kw("loro piana", 5),
    kw("massimo dutti", 3),
    kw("arket", 2),
    kw("brooks", 3),
    kw("brooks brothers", 5),
    // Shoes-specific positives (loafers, penny loafers, bit loafers, boat shoes,
    // derbies, dress shoes, leather shoes are all classic Old Money footwear)
    kw("penny loafer", 5),
    kw("penny loafers", 5),
    kw("bit loafer", 5),
    kw("bit loafers", 5),
    kw("metal bit", 4),
    kw("boat shoe", 4),
    kw("boat shoes", 4),
    kw("derby", 4),
    kw("dress shoes", 4),
    kw("dress shoe", 4),
    kw("leather shoes", 3),
    kw("leather loafer", 5),
    kw("leather loafers", 5),
    kw("moc toe", 2),
    kw("double buckle", 3),
    kw("raised seam", 1),
    kw("suede", 3),
    // Negatives
    kw("hoodie", -4),
    kw("graphic", -3),
    kw("logo", -2),
    kw("oversized", -2),
    kw("cargo", -3),
    kw("track pants", -4),
    kw("sweatpants", -3),
    kw("ripped", -3),
    kw("distressed", -3),
    kw("neon", -4),
    kw("sequin", -4),
    kw("rhinestone", -4),
    kw("square-toe", -4),
    kw("square toe", -4),
    kw("low-rise", -3),
    kw("low rise", -3),
    kw("skinny", -2),
    kw("backpack", -2),
    // Footwear to AVOID for old_money. Driving moccasins sit a tier below
    // loafers (soft sole, tie laces) — Riviera-casual, never with tailoring.
    kw("moccasin", -3),
    kw("moccasins", -3),
    kw("driving", -2),
    kw("gommino", -3),
    kw("mule", -2),
    kw("mules", -2),
    kw("chunky sneaker", -4),
    kw("chunky sneakers", -4),
    kw("basketball", -5),
    kw("skate sneaker", -4),
    kw("skate sneakers", -4),
    kw("thick-soled", -3),
    kw("thick soled", -3),
    kw("retro sneaker", -2),
    kw("retro sneakers", -2),
    kw("rope lace", -2),
    // Street prints have no place in quiet luxury — souvenir graphics,
    // florals and lettering read as loud as logos.
    kw("souvenir", -4),
    kw("souvenirs", -4),
    kw("floral", -3),
    kw("florals", -3),
    kw("print", -2),
    kw("prints", -2),
    kw("printed", -2),
    kw("paisley", -3),
    // IMPORTANT: plain "sneaker" is mildly negative (clean whites are ok) and
    // "chunky" alone is bad regardless of what it qualifies.
    kw("sneaker", -1),
    kw("sneakers", -1),
    kw("chunky", -3),
    // NEVER pair shorts with old_money - they're fundamentally incompatible
    kw("shorts", -5),
    kw("short", -5),
    kw("bermuda", -5),
    kw("bermudas", -5),
    // Blue jeans read as casual streetwear, not old money - exclude them
    // the same way shorts are excluded (was previously unscored/neutral,
    // letting jeans slip into "old money" outfits alongside dinner shirts).
    kw("jeans", -4),
    kw("denim", -3),
    kw("jean", -4)
  ],
  semi_classic: [
    kw("blazer", 3),
    kw("cardigan", 3),
    kw("knit polo", 3),
    kw("turtleneck", 2),
    kw("chino", 3),
    kw("chinos", 3),
    kw("slack", 2),
    kw("slacks", 2),
    kw("loafer", 3),
    kw("loafers", 3),
    kw("suede", 3),
    kw("desert boot", 3),
    kw("merino", 2),
    kw("cotton", 1),
    kw("linen", 2),
    kw("poplin", 2),
    kw("tailored", 2),
    kw("structured", 2),
    kw("regular-fit", 2),
    kw("relaxed-fit", 1),
    kw("navy", 2),
    kw("cream", 2),
    kw("beige", 2),
    kw("camel", 2),
    kw("olive", 2),
    kw("tan", 2),
    kw("burgundy", 2),
    kw("white", 1),
    kw("massimo dutti", 3),
    kw("arket", 3),
    kw("cos", 2),
    kw("sneaker", 1),
    kw("sneakers", 1),
    kw("jeans", 1),
    kw("denim", 1),
    kw("sweater", 2),
    kw("pullover", 1),
    kw("vest", 1),
    // Negatives
    kw("hoodie", -2),
    kw("graphic", -2),
    kw("cargo", -3),
    kw("sweatpants", -3),
    kw("joggers", -3),
    kw("track", -2),
    kw("neon", -4),
    kw("sequin", -4),
    kw("rhinestone", -4),
    kw("oversized", -2),
    kw("baggy", -3),
    kw("ripped", -2)
  ],
  minimalist: [
    kw("minimal", 5),
    kw("clean", 2),
    kw("seamless", 3),
    kw("structured", 2),
    kw("essential", 3),
    kw("basic", 2),
    kw("plain", 3),
    kw("solid", 2),
    kw("cos", 5),
    kw("uniqlo", 4),
    kw("acne", 4),
    kw("jil sander", 5),
    kw("the row", 5),
    kw("everlane", 4),
    kw("arket", 3),
    kw("crew neck", 2),
    kw("turtleneck", 2),
    kw("mock neck", 2),
    kw("knit polo", 3),
    kw("merino", 3),
    kw("wool", 2),
    kw("cotton", 1),
    kw("black", 2),
    kw("white", 2),
    kw("grey", 2),
    kw("gray", 2),
    kw("beige", 2),
    kw("navy", 1),
    kw("stone", 2),
    kw("cream", 2),
    kw("sand", 2),
    kw("suede", 2),
    kw("poplin", 2),
    kw("linen", 2),
    // Minimalist footwear: clean leather sneakers, minimal loafers, plain boots
    kw("minimal sneaker", 5),
    kw("minimal sneakers", 5),
    kw("leather sneaker", 3),
    kw("leather sneakers", 3),
    kw("leather loafer", 3),
    kw("leather loafers", 3),
    kw("loafer", 2),
    kw("loafers", 2),
    // Negatives
    kw("graphic", -4),
    kw("logo", -3),
    kw("print", -3),
    kw("neon", -5),
    kw("sequin", -5),
    kw("rhinestone", -5),
    kw("floral", -3),
    kw("colorblock", -3),
    kw("color-block", -3),
    kw("chunky", -3),
    kw("basketball", -4),
    kw("retro sneaker", -2),
    // Minimalism takes no pleats (DNA).
    kw("pleated", -3),
    kw("pleats", -3)
  ],
  business_casual: [
    kw("blazer", 4),
    kw("chino", 4),
    kw("chinos", 4),
    kw("trouser", 4),
    kw("trousers", 4),
    kw("oxford", 4),
    kw("button-down", 3),
    kw("button down", 3),
    kw("loafer", 4),
    kw("loafers", 4),
    kw("dress shirt", 4),
    kw("dress shoes", 4),
    kw("dress shoe", 4),
    kw("derby", 4),
    kw("poplin", 3),
    kw("tailored", 4),
    kw("knit polo", 3),
    kw("turtleneck", 3),
    kw("cardigan", 3),
    kw("flannel", 3),
    kw("herringbone", 3),
    kw("tweed", 3),
    kw("suede", 3),
    kw("slim-fit", 2),
    kw("slim fit", 2),
    kw("straight-fit", 1),
    kw("shirt", 2),
    kw("polo", 2),
    kw("penny loafer", 4),
    kw("penny loafers", 4),
    kw("bit loafer", 4),
    kw("bit loafers", 4),
    kw("leather shoes", 3),
    kw("leather loafer", 4),
    kw("leather loafers", 4),
    kw("navy", 2),
    kw("charcoal", 2),
    kw("tan", 2),
    kw("white", 1),
    kw("cream", 2),
    kw("brown", 2),
    kw("burgundy", 2),
    kw("light blue", 2),
    kw("hugo boss", 4),
    kw("theory", 4),
    kw("everlane", 2),
    // Negatives
    kw("hoodie", -4),
    kw("graphic", -3),
    kw("cargo", -4),
    kw("ripped", -4),
    kw("sweatpants", -4),
    kw("sequin", -5),
    kw("neon", -5),
    kw("crop", -2),
    kw("chunky", -3),
    kw("basketball", -5),
    kw("skate", -3),
    kw("low-rise", -3),
    kw("low rise", -3),
    kw("skinny", -2),
    kw("square-toe", -4),
    kw("square toe", -4),
    // Shorts are inappropriate for business_casual
    kw("shorts", -5),
    kw("short", -5),
    kw("bermuda", -5),
    kw("bermudas", -5),
    // Jeans read as casual, not business casual
    kw("jeans", -3),
    kw("denim", -2),
    kw("jean", -3),
    // Driving moccasins never pair with tailoring; souvenir/print layers
    // underdress the office.
    kw("moccasin", -3),
    kw("moccasins", -3),
    kw("driving", -2),
    kw("gommino", -3),
    kw("mule", -2),
    kw("mules", -2),
    kw("souvenir", -4),
    kw("souvenirs", -4),
    kw("floral", -3),
    kw("print", -2),
    kw("prints", -2),
    kw("printed", -2)
  ],
  casual: [
    kw("t-shirt", 2),
    kw("tee", 2),
    kw("jeans", 2),
    kw("jean", 1),
    kw("sweater", 2),
    kw("cardigan", 1),
    kw("sneaker", 2),
    kw("sneakers", 2),
    kw("polo", 1),
    kw("shorts", 1),
    kw("denim", 2),
    kw("regular-fit", 1),
    kw("knit", 2),
    kw("slack", 2),
    kw("slacks", 2),
    kw("loafer", 2),
    kw("loafers", 2),
    kw("chino", 2),
    kw("chinos", 2),
    kw("canvas", 1),
    kw("suede", 2),
    kw("basketball", -3),
    kw("graphic hoodie", -3),
    kw("sweatpants", -2),
    // Refined Casual DNA: knit polos standard; hoodies and athletic
    // sneakers belong to the gym. (Precise phrases only — bare 'chunky'
    // would demote chunky-knit sweaters, which are fine.)
    kw("hoodie", -2),
    kw("hoodies", -2),
    kw("hooded", -2),
    kw("chunky sneaker", -3),
    kw("chunky sneakers", -3),
    kw("athletic", -2),
    kw("running shoe", -2),
    kw("running shoes", -2)
  ],
  classic: [
    // Core tailored pieces
    kw("suit", 5),
    kw("blazer", 4),
    kw("sport coat", 4),
    kw("sport jacket", 4),
    kw("tailored", 4),
    kw("trouser", 4),
    kw("trousers", 4),
    kw("pleated", 3),
    kw("slack", 3),
    kw("slacks", 3),
    kw("oxford", 4),
    kw("button-down", 3),
    kw("button down", 3),
    kw("dress shirt", 5),
    kw("cardigan", 3),
    kw("turtleneck", 4),
    kw("knit polo", 4),
    // Fabrics
    kw("flannel", 4),
    kw("tweed", 4),
    kw("herringbone", 4),
    kw("cashmere", 4),
    kw("merino", 3),
    kw("wool", 3),
    kw("linen", 2),
    kw("silk", 2),
    kw("poplin", 3),
    kw("oxford cloth", 3),
    kw("suede", 3),
    // Colors
    kw("navy", 3),
    kw("charcoal", 3),
    kw("cream", 2),
    kw("burgundy", 3),
    kw("camel", 3),
    kw("beige", 2),
    kw("tan", 2),
    kw("forest", 2),
    // Shoes
    kw("loafer", 4),
    kw("loafers", 4),
    kw("oxford shoe", 5),
    kw("oxford shoes", 5),
    kw("derby", 4),
    kw("dress shoes", 5),
    kw("dress shoe", 5),
    kw("leather shoes", 3),
    kw("suede shoes", 3),
    kw("penny loafer", 5),
    kw("penny loafers", 5),
    kw("bit loafer", 5),
    kw("bit loafers", 5),
    kw("brogue", 4),
    kw("brogues", 4),
    kw("monk strap", 4),
    kw("chelsea boot", 3),
    kw("chelsea boots", 3),
    // Brands
    kw("brooks brothers", 5),
    kw("ralph lauren", 4),
    kw("brunello", 5),
    kw("loro piana", 5),
    kw("massimo dutti", 3),
    kw("arket", 2),
    // Accessories
    kw("tie", 3),
    kw("pocket square", 3),
    kw("scarf", 2),
    // Negatives
    kw("hoodie", -4),
    kw("graphic", -3),
    kw("logo", -2),
    kw("cargo", -4),
    kw("sweatpants", -4),
    kw("joggers", -4),
    kw("track pants", -4),
    kw("shorts", -5),
    kw("short", -5),
    kw("bermuda", -5),
    kw("bermudas", -5),
    kw("puffer", -3),
    kw("bomber", -3),
    kw("denim jacket", -2),
    kw("chunky sneaker", -4),
    kw("chunky sneakers", -4),
    kw("athletic", -3),
    kw("square-toe", -4),
    kw("square toe", -4),
    kw("neon", -4),
    kw("sequin", -4),
    kw("ripped", -3),
    kw("low-rise", -3),
    kw("low rise", -3),
    kw("skinny", -2),
    kw("oversized", -2),
    kw("backpack", -2),
    // Jeans read as casual, not classic tailoring
    kw("jeans", -3),
    kw("denim", -2),
    kw("jean", -3),
    // Driving moccasins never pair with tailoring; souvenir/print layers
    // break Savile-Row discipline.
    kw("moccasin", -3),
    kw("moccasins", -3),
    kw("driving", -2),
    kw("gommino", -3),
    kw("mule", -2),
    kw("mules", -2),
    kw("souvenir", -4),
    kw("souvenirs", -4),
    kw("floral", -3),
    kw("print", -2),
    kw("prints", -2),
    kw("printed", -2)
  ]
};
function buildSearchBlob(item) {
  return [
    item.name,
    item.description,
    item.brand,
    item.color,
    item.type,
    item.category,
    item.macroCategory,
    item.style
  ].filter(Boolean).join(" ").toLowerCase();
}
function extractMatchingWords(blob, dictionary) {
  const out = [];
  for (const word of dictionary) {
    if (blob.includes(word)) out.push(word);
  }
  return out;
}
function scoreStyle(blob, signals) {
  let score = 0;
  for (const [re, weight] of signals) {
    if (re.test(blob)) score += weight;
  }
  return score;
}
function clamp012(n) {
  if (n < 0) return 0;
  if (n > 1) return 1;
  return n;
}
function estimateFormality(blob) {
  let score = 0.5;
  const bumps = [
    [/\b(suit|blazer|tailored|trouser|trousers|oxford|dress shirt|loafer|pinstripe|tweed|herringbone|cashmere|wool|silk)\b/, 0.1],
    [/\b(chino|chinos|polo|cardigan|linen|button-down|button down|poplin|merino)\b/, 0.06],
    [/\b(jeans|t-shirt|tee|sneaker|sneakers|hoodie|sweatpants|joggers|track|cargo|graphic|printed|ripped)\b/, -0.1],
    [/\b(baggy|oversized|distressed|athletic|workout|running|fleece|terry)\b/, -0.06]
  ];
  for (const [re, delta] of bumps) {
    if (re.test(blob)) score += delta;
  }
  return clamp012(score);
}
function inferItemAttributes(item) {
  const blob = buildSearchBlob(item);
  const materials = extractMatchingWords(blob, MATERIAL_WORDS);
  const patterns = extractMatchingWords(blob, PATTERN_WORDS);
  const styleScores = Object.keys(STYLE_SIGNALS).map((style) => ({
    style,
    score: scoreStyle(blob, STYLE_SIGNALS[style])
  }));
  styleScores.sort((a, b) => b.score - a.score);
  const styleTags = styleScores.filter((s) => s.score > 0).map((s) => s.style);
  const oldMoneyScore = styleScores.find((s) => s.style === "old_money")?.score ?? 0;
  const semiClassicScore = styleScores.find((s) => s.style === "semi_classic")?.score ?? 0;
  const semiClassicIndex = styleTags.indexOf("semi_classic");
  if (semiClassicIndex >= 0 && oldMoneyScore >= semiClassicScore + 2) {
    styleTags.splice(semiClassicIndex, 1);
  }
  if (styleTags.length === 0) styleTags.push("casual");
  const neutralHits = extractMatchingWords(blob, NEUTRAL_COLOR_WORDS).length;
  const boldHits = extractMatchingWords(blob, BOLD_COLOR_WORDS).length;
  const isNeutralPalette = neutralHits > 0 && neutralHits >= boldHits;
  return {
    styleTags,
    materials,
    patterns,
    formality: estimateFormality(blob),
    isNeutralPalette
  };
}
function rawStyleScore(item, style) {
  return scoreStyle(buildSearchBlob(item), STYLE_SIGNALS[style] || []);
}
function scoreItemForStyle(item, style) {
  const rawScore = rawStyleScore(item, style);
  const normalized = (rawScore + 4) / 16;
  return clamp012(normalized);
}
function rankItemsForStyle(items, style, opts = {}) {
  const { minKeep = 8, dropThreshold = -3, perCategoryFloor = 3 } = opts;
  const scored = items.filter((item) => !opts.eligibility || opts.eligibility(item)).map((item) => ({
    item,
    raw: rawStyleScore(item, style),
    macro: (item.macroCategory || "").toLowerCase()
  }));
  scored.sort((a, b) => b.raw - a.raw);
  const passesThreshold = /* @__PURE__ */ new Set();
  scored.forEach((s, idx) => {
    if (s.raw > dropThreshold) passesThreshold.add(idx);
  });
  if (perCategoryFloor > 0) {
    const perCatCount = /* @__PURE__ */ new Map();
    scored.forEach((s, idx) => {
      if (!s.macro) return;
      const count = perCatCount.get(s.macro) || 0;
      if (count < perCategoryFloor) {
        passesThreshold.add(idx);
        perCatCount.set(s.macro, count + 1);
      }
    });
  }
  const filtered = scored.filter((_, idx) => passesThreshold.has(idx));
  const final = filtered.length >= minKeep ? filtered : scored;
  return final.map((s) => s.item);
}
function normalizeStyleId(raw) {
  if (!raw) return "casual";
  const key = raw.trim().toLowerCase().replace(/[\s-]+/g, "_");
  if (key === "old_money" || key === "semi_classic" || key === "minimalist" || key === "business_casual" || key === "casual" || key === "classic") {
    return key;
  }
  return "casual";
}

// src/domain/stylingContext.ts
function stylingContext(input) {
  const occasion = (input.occasion || "").toLowerCase().trim();
  const request = input.request || "";
  const athletic = /\b(workout|gym|training|fitness|sport)\b|трениров|спортзал/.test(occasion);
  const formal = /\bformal\b|black[ -]?tie|формаль|торжеств/.test(occasion);
  const smart = /\b(work|office|date|business)\b|работ|свидан/.test(occasion);
  const weather = input.weather && (Number.isFinite(input.weather.temp) || !!input.weather.condition) ? {
    condition: input.weather.condition || void 0,
    temp: Number.isFinite(input.weather.temp) ? input.weather.temp : void 0
  } : null;
  return {
    ruleVersion: "stylist-quality-v1",
    style: normalizeStyleId(input.style || "casual"),
    occasion,
    dressCode: athletic ? "athletic" : formal ? "formal" : smart ? "smart" : "casual",
    activity: occasion,
    request,
    weather,
    anchorId: input.anchorId || null,
    sourceMode: input.sourceMode || "mixed",
    budget: input.budget,
    preferences: input.preferences ? {
      ...input.preferences,
      avoidColors: input.preferences.avoidColors ? [...input.preferences.avoidColors] : void 0,
      dislikedPatterns: input.preferences.dislikedPatterns ? [...input.preferences.dislikedPatterns] : void 0
    } : void 0,
    uncertainty: [
      ...!weather ? ["weather_unknown"] : [],
      ...!occasion ? ["occasion_unknown"] : []
    ]
  };
}

// src/domain/outfitChoices.ts
function resolveOutfitChoices(selections, choices) {
  if (!selections?.length) return null;
  const byId = new Map(choices.map((choice) => [choice.id, choice]));
  const seen = /* @__PURE__ */ new Set(), result = [];
  for (const selection of selections) {
    const id = selection.items?.length === 1 ? selection.items[0]?.id : null;
    if (typeof id !== "string" || seen.has(id) || !byId.has(id)) return null;
    seen.add(id);
    const { id: _modelId, ...metadata } = selection;
    result.push({ ...metadata, items: byId.get(id).items });
  }
  return result;
}
function parseRankedOutfitChoices(raw) {
  try {
    const parsed = JSON.parse(raw.trim().replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, ""));
    const ids = parsed?.rankedLookIds;
    if (!Array.isArray(ids) || !ids.length || ids.length > 100 || ids.some((id) => typeof id !== "string" || !id.trim())) return null;
    return ids.map((id) => ({ items: [{ id: id.trim() }] }));
  } catch {
    return null;
  }
}

// src/domain/outfitGenerationRequest.ts
var MAX_GENERATION_BODY_BYTES = 16 * 1024 * 1024;
var OutfitRequestError = class extends Error {
  constructor(status = 400) {
    super("invalid_generation_request");
    this.status = status;
  }
  status;
};
var object = (value) => !!value && typeof value === "object" && !Array.isArray(value);
function text2(value, max) {
  if (value !== void 0 && (typeof value !== "string" || value.length > max)) throw new OutfitRequestError();
}
function list(value, count, max) {
  if (value === void 0) return;
  if (!Array.isArray(value) || value.length > count || value.some((item) => typeof item !== "string" || !item.trim() || item.length > max)) throw new OutfitRequestError();
}
function validateOutfitGenerationRequest(body) {
  if (!object(body)) throw new OutfitRequestError();
  text2(body.prompt, 1e3);
  text2(body.stylePreferences, 80);
  text2(body.occasion, 100);
  text2(body.conversationContext, 4e3);
  text2(body.anchorItemId, 200);
  if (body.limit !== void 0 && (!Number.isInteger(body.limit) || body.limit < 1 || body.limit > 15)) throw new OutfitRequestError();
  if (body.deckTarget !== void 0 && (!Number.isInteger(body.deckTarget) || body.deckTarget < 1 || body.deckTarget > Math.min(15, body.limit ?? 3))) throw new OutfitRequestError();
  list(body.recentLookSignatures, 36, 1e3);
  list(body.selectedItemIds, 200, 200);
  list(body.excludeItemIds, 200, 200);
  list(body.excludeProductKeys, 450, 2048);
  list(body.previousDeckProductKeys, 60, 2048);
  list(body.olderRecentProductKeys, 450, 2048);
  list(body.previousShoeProductKeys, 60, 2048);
  list(body.recentShoeProductKeys, 450, 2048);
  text2(body.rotationScope, 500);
  text2(body.generationRequestId, 100);
  for (const key of ["rotationDecks", "shoeRotationDecks"]) if (body[key] !== void 0) {
    if (!Array.isArray(body[key]) || body[key].length > 12) throw new OutfitRequestError();
    for (const deck of body[key]) {
      if (!object(deck)) throw new OutfitRequestError();
      text2(deck.requestId, 100);
      list(deck.productKeys, 60, 2048);
    }
  }
  for (const key of ["wardrobeOnly", "useProvidedWardrobeOnly", "catalogOnly"]) if (body[key] !== void 0 && typeof body[key] !== "boolean") throw new OutfitRequestError();
  if (body.weather !== void 0 && body.weather !== null) {
    if (!object(body.weather)) throw new OutfitRequestError();
    if (body.weather.temp !== void 0 && (typeof body.weather.temp !== "number" || !Number.isFinite(body.weather.temp) || body.weather.temp < -100 || body.weather.temp > 100)) throw new OutfitRequestError();
    text2(body.weather.condition, 100);
  }
  if (body.preferences !== void 0 && body.preferences !== null) {
    if (!object(body.preferences)) throw new OutfitRequestError();
    for (const key of ["favoriteColors", "avoidColors", "likedPatterns", "dislikedPatterns", "styleGoals"]) list(body.preferences[key] ?? void 0, 20, 100);
    text2(body.preferences.fitPreference ?? void 0, 100);
    text2(body.preferences.budgetRatesId, 200);
  }
  if (body.wardrobeItems !== void 0) {
    if (!Array.isArray(body.wardrobeItems) || body.wardrobeItems.length > 200) throw new OutfitRequestError();
    for (const item of body.wardrobeItems) {
      if (!object(item) || typeof item.id !== "string" || !item.id.trim()) throw new OutfitRequestError();
      text2(item.id, 200);
      text2(item.name ?? void 0, 300);
      text2(item.description ?? void 0, 2e3);
      for (const key of ["brand", "color", "primary_color", "pattern", "type", "category", "macroCategory"]) text2(item[key] ?? void 0, 150);
      list(item.styleTags ?? void 0, 20, 100);
      if (Array.isArray(item.material)) list(item.material, 20, 100);
      else text2(item.material ?? void 0, 500);
      for (const key of ["imageUrl", "image", "thumbnailUrl"]) if (item[key] !== void 0 && typeof item[key] !== "string") throw new OutfitRequestError();
    }
  }
  return body;
}

// backend/supabase/functions/_shared/outfitGenerationRequest.ts
async function readOutfitGenerationRequest(request) {
  const declared = request.headers.get("content-length");
  if (declared !== null && Number(declared) > MAX_GENERATION_BODY_BYTES) throw new OutfitRequestError(413);
  const reader = request.body?.getReader();
  if (!reader) throw new OutfitRequestError();
  let timer;
  const deadline = new Promise((_, reject) => {
    timer = setTimeout(() => {
      reject(new OutfitRequestError(408));
      void reader.cancel().catch(() => void 0);
    }, 15e3);
  });
  const chunks = [];
  let bytes = 0;
  try {
    for (; ; ) {
      const part = await Promise.race([reader.read(), deadline]);
      if (part.done) break;
      bytes += part.value.byteLength;
      if (bytes > MAX_GENERATION_BODY_BYTES) throw new OutfitRequestError(413);
      chunks.push(part.value);
    }
    const buffer = new Uint8Array(bytes);
    let offset = 0;
    for (const chunk of chunks) {
      buffer.set(chunk, offset);
      offset += chunk.byteLength;
    }
    let body;
    try {
      body = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(buffer));
    } catch {
      throw new OutfitRequestError();
    }
    return validateOutfitGenerationRequest(body);
  } finally {
    clearTimeout(timer);
    void reader.cancel().catch(() => void 0);
    reader.releaseLock();
  }
}

// src/domain/outfitBudgetText.ts
var currencies = { "$": "USD", usd: "USD", dollar: "USD", dollars: "USD", "\u20AC": "EUR", eur: "EUR", euro: "EUR", euros: "EUR", "\xA3": "GBP", gbp: "GBP", pound: "GBP", pounds: "GBP", uzs: "UZS" };
var words = "(?:usd|eur|gbp|uzs|dollars?|euros?|pounds?)\\b";
var currency = `([$\u20AC\xA3]|${words})`;
var amount = "(\\d{1,3}(?:[ ,.\\u00a0\\u202f]\\d{3})+(?:[.,]\\d{1,2})?|\\d+(?:[.,]\\d{1,2})?)";
var units = /^(?:\s*(?:°\s*)?(?:c|f|celsius|fahrenheit|cm|kg|lbs?|items?|pieces?|degrees?)\b)/i;
function parseAmount(raw, code) {
  const compact = raw.replace(/[\s\u00a0\u202f]/g, "");
  const decimal = /[.,]\d{1,2}$/.exec(compact);
  const normalized = decimal ? compact.slice(0, decimal.index).replace(/[.,]/g, "") + "." + decimal[0].slice(1) : compact.replace(/[.,]/g, "");
  const value = Number(normalized), limit = code === "UZS" ? 1e12 : 1e5;
  return Number.isFinite(value) && value >= 0 && value <= limit ? Math.round(value * 100) / 100 : null;
}
function scan(text3) {
  const src = String(text3 || ""), budgets = [], spans = [];
  const add = (match2, raw, token, after, extra = 0) => {
    if (units.test(after)) return;
    const code = currencies[token.trim().toLowerCase()] || currencies[token.trim()] || "USD";
    const value = parseAmount(raw, code);
    if (value === null) return;
    budgets.push({ amount: value, currency: code });
    spans.push([match2.index, match2.index + match2[0].length + extra]);
  };
  let match;
  const prefix = new RegExp(`([$\u20AC\xA3])\\s*${amount}`, "g");
  while ((match = prefix.exec(src)) !== null) add(match, match[2], match[1], src.slice(match.index + match[0].length));
  const suffix = new RegExp(`${amount}\\s*${currency}`, "gi");
  while ((match = suffix.exec(src)) !== null) add(match, match[1], match[2], src.slice(match.index + match[0].length));
  const keyword = new RegExp(`\\b(?:budget(?:\\s*(?:of|is|:))?|under|below|within|up\\s*to|at\\s*most|no\\s+more\\s+than|less\\s+than|max(?:imum)?(?:\\s+of)?)\\s*${currency}?\\s*${amount}`, "gi");
  while ((match = keyword.exec(src)) !== null) {
    const after = src.slice(match.index + match[0].length), trailing = !match[1] ? new RegExp(`^\\s*${currency}`, "i").exec(after) : null;
    add(match, match[2], match[1] || trailing?.[1] || "", after, trailing?.[0].length || 0);
  }
  return { budgets, spans };
}
function budgetCandidatesFromText(text3) {
  return scan(text3).budgets;
}

// src/domain/budgetFx.ts
var BUDGET_CURRENCIES = ["USD", "EUR", "GBP", "UZS"];
var BUDGET_FX_TTL_MS = 60 * 60 * 1e3;
var MAX_REFERENCE_AGE_DAYS = 7;
var BudgetFxError = class extends Error {
  constructor() {
    super("budget_rates_unavailable");
    this.name = "BudgetFxError";
  }
};
function uzbekDateKey(now) {
  return new Date(now + 5 * 60 * 60 * 1e3).toISOString().slice(0, 10);
}
function validDate(value) {
  return /^\d{4}-\d{2}-\d{2}$/.test(value) && Number.isFinite(Date.parse(value)) && new Date(value).toISOString().slice(0, 10) === value;
}
function snapshotId(asOf, rates) {
  return `CBU:${asOf}:${rates.USD}:${rates.EUR}:${rates.GBP}`;
}
function validateBudgetFxSnapshot(value, now = Date.now()) {
  const row = value;
  if (!row || row.source !== "CBU" || typeof row.asOf !== "string" || !validDate(row.asOf) || typeof row.fetchedAt !== "string") throw new BudgetFxError();
  const referenceAge = (Date.parse(uzbekDateKey(now)) - Date.parse(row.asOf)) / 864e5;
  const fetchedAge = now - Date.parse(row.fetchedAt);
  if (!Number.isFinite(fetchedAge) || fetchedAge < -6e4 || fetchedAge > BUDGET_FX_TTL_MS || referenceAge < 0 || referenceAge > MAX_REFERENCE_AGE_DAYS) throw new BudgetFxError();
  const rates = row.uzsPerUnit;
  if (!rates || rates.UZS !== 1 || !BUDGET_CURRENCIES.every((currency2) => typeof rates[currency2] === "number" && Number.isFinite(rates[currency2]) && rates[currency2] > 0 && rates[currency2] <= 1e12)) throw new BudgetFxError();
  if (row.id !== snapshotId(row.asOf, rates)) throw new BudgetFxError();
  return Object.freeze({ id: row.id, source: "CBU", asOf: row.asOf, fetchedAt: row.fetchedAt, uzsPerUnit: Object.freeze({ USD: rates.USD, EUR: rates.EUR, GBP: rates.GBP, UZS: 1 }) });
}
function parseCbuBudgetRates(payload, now = Date.now()) {
  if (!Array.isArray(payload) || payload.length > 1e3) throw new BudgetFxError();
  const rates = { UZS: 1 };
  let asOf = "";
  for (const currency2 of ["USD", "EUR", "GBP"]) {
    const rows = payload.filter((row2) => row2 && row2.Ccy === currency2);
    if (rows.length !== 1) throw new BudgetFxError();
    const row = rows[0];
    if (typeof row.Date !== "string" || !/^\d{2}\.\d{2}\.\d{4}$/.test(row.Date)) throw new BudgetFxError();
    const [day, month, year] = row.Date.split("."), date = `${year}-${month}-${day}`;
    if (asOf && date !== asOf) throw new BudgetFxError();
    asOf = date;
    if (typeof row.Rate !== "string" || typeof row.Nominal !== "string" || !/^\d+(?:\.\d+)?$/.test(row.Rate) || !/^\d+(?:\.\d+)?$/.test(row.Nominal)) throw new BudgetFxError();
    const nominal = Number(row.Nominal), rate = Number(row.Rate);
    if (!Number.isFinite(nominal) || nominal <= 0 || !Number.isFinite(rate) || rate <= 0) throw new BudgetFxError();
    rates[currency2] = rate / nominal;
  }
  return validateBudgetFxSnapshot({ id: snapshotId(asOf, rates), source: "CBU", asOf, fetchedAt: new Date(now).toISOString(), uzsPerUnit: rates }, now);
}
function referenceAmount(amount2, from, to, snapshot, now = Date.now()) {
  const source = from.trim().toUpperCase(), target = to.trim().toUpperCase();
  if (!Number.isFinite(amount2) || amount2 < 0 || !BUDGET_CURRENCIES.includes(source) || !BUDGET_CURRENCIES.includes(target)) return Infinity;
  if (source === target) return amount2;
  const rates = validateBudgetFxSnapshot(snapshot, now).uzsPerUnit;
  const converted = amount2 * rates[source] / rates[target];
  return Number.isFinite(converted) ? converted : Infinity;
}
function budgetFxContext(snapshot, now = Date.now()) {
  return Object.freeze({ snapshot: validateBudgetFxSnapshot(snapshot, now), checkedAt: now });
}
function fitsBudgetAmount(spend, cap) {
  if (!Number.isFinite(spend) || !Number.isFinite(cap) || spend < 0 || cap < 0) return false;
  const noise = Math.min(1e-6, Number.EPSILON * Math.max(1, spend, cap) * 8);
  return spend <= cap + noise;
}

// backend/supabase/functions/_shared/budgetFxCache.ts
var refresh = null;
async function bounded(work, ms, parent) {
  const controller = new AbortController();
  const abort = () => controller.abort();
  let timer;
  let rejectAbort;
  const cancelled = new Promise((_, reject) => {
    rejectAbort = () => reject(new BudgetFxError());
    controller.signal.addEventListener("abort", rejectAbort, { once: true });
  });
  parent?.addEventListener("abort", abort, { once: true });
  if (parent?.aborted) controller.abort();
  timer = setTimeout(() => controller.abort(), ms);
  try {
    return await Promise.race([Promise.resolve().then(() => {
      if (controller.signal.aborted) throw new BudgetFxError();
      return work(controller.signal);
    }), cancelled]);
  } finally {
    clearTimeout(timer);
    parent?.removeEventListener("abort", abort);
    if (rejectAbort) controller.signal.removeEventListener("abort", rejectAbort);
    controller.abort();
  }
}
async function cachedSnapshot(admin, now, signal) {
  const result = await bounded((s) => admin.from("budget_exchange_rates").select("snapshot").order("fetched_at", { ascending: false }).limit(1).abortSignal(s).maybeSingle(), 5e3, signal);
  if (result.error) throw new BudgetFxError();
  if (result.data?.snapshot) {
    try {
      return validateBudgetFxSnapshot(result.data.snapshot, now);
    } catch {
    }
  }
  return null;
}
async function leaseRpc(admin, name, args, signal, ms = 2e3) {
  const result = await bounded((s) => admin.rpc(name, args).abortSignal(s), ms, signal);
  if (result.error) throw new BudgetFxError();
  return result.data;
}
async function waitForOwner(ms, signal) {
  await bounded((s) => new Promise((resolve, reject) => {
    const timer = setTimeout(resolve, ms);
    s.addEventListener("abort", () => {
      clearTimeout(timer);
      reject(new BudgetFxError());
    }, { once: true });
  }), ms + 100, signal);
}
async function currentBudgetRates(admin, now = Date.now(), request = fetch) {
  if (refresh) return refresh;
  const pending = (async () => {
    let token = null, stage = "cache_read";
    try {
      return await bounded(async (signal) => {
        let delay = 100;
        while (!signal.aborted) {
          stage = "cache_read";
          const cached = await cachedSnapshot(admin, now, signal);
          if (cached) return cached;
          stage = "lease_claim";
          const candidate = crypto.randomUUID();
          const state = await leaseRpc(admin, "claim_budget_rate_refresh", { p_token: candidate }, signal);
          if (state === "cooldown") throw new BudgetFxError();
          if (state === "busy") {
            await waitForOwner(delay, signal);
            delay = Math.min(1e3, delay * 2);
            continue;
          }
          if (state !== "acquired") throw new BudgetFxError();
          token = candidate;
          stage = "cache_recheck";
          let snapshot = await cachedSnapshot(admin, now, signal);
          if (!snapshot) {
            snapshot = await bounded(async (upstreamSignal) => {
              stage = "upstream_fetch";
              const url = `https://cbu.uz/uz/arkhiv-kursov-valyut/json/all/${uzbekDateKey(now)}/`;
              const response = await request(url, { signal: upstreamSignal, redirect: "error", headers: { Accept: "application/json" } });
              if (upstreamSignal.aborted || !response.ok || Number(response.headers.get("content-length") || 0) > 262144) throw new BudgetFxError();
              stage = "upstream_body";
              const reader = response.body?.getReader();
              if (!reader) throw new BudgetFxError();
              const chunks = [];
              let size = 0;
              try {
                for (; ; ) {
                  const { done, value } = await reader.read();
                  if (upstreamSignal.aborted) throw new BudgetFxError();
                  if (done) break;
                  size += value.byteLength;
                  if (size > 262144) {
                    await reader.cancel();
                    throw new BudgetFxError();
                  }
                  chunks.push(value);
                }
              } finally {
                reader.releaseLock();
              }
              const bytes = new Uint8Array(size);
              let offset = 0;
              for (const chunk of chunks) {
                bytes.set(chunk, offset);
                offset += chunk.byteLength;
              }
              stage = "upstream_validation";
              return parseCbuBudgetRates(JSON.parse(new TextDecoder().decode(bytes)), now);
            }, 8e3, signal);
          }
          if (signal.aborted) throw new BudgetFxError();
          stage = "fenced_publish";
          const saved = await leaseRpc(admin, "finish_budget_rate_refresh", { p_token: token, p_snapshot: snapshot }, signal);
          if (saved !== true || signal.aborted) throw new BudgetFxError();
          token = null;
          return snapshot;
        }
        throw new BudgetFxError();
      }, 12e3);
    } catch {
      console.warn("[budget-rates] refresh_failed", { stage });
      if (token) {
        try {
          await leaseRpc(admin, "finish_budget_rate_refresh", { p_token: token, p_snapshot: null }, void 0, 1e3);
        } catch {
        }
      }
      throw new BudgetFxError();
    }
  })();
  refresh = pending;
  try {
    return await pending;
  } finally {
    if (refresh === pending) refresh = null;
  }
}
async function budgetRatesById(admin, id, now = Date.now()) {
  if (typeof id !== "string" || !id.startsWith("CBU:") || id.length > 200) throw new BudgetFxError();
  const controller = new AbortController();
  let timer;
  try {
    const query = admin.from("budget_exchange_rates").select("snapshot").eq("id", id).abortSignal(controller.signal).maybeSingle();
    const result = await Promise.race([query, new Promise((_, reject) => {
      timer = setTimeout(() => {
        controller.abort();
        reject(new BudgetFxError());
      }, 5e3);
    })]);
    if (result.error || !result.data?.snapshot) throw new BudgetFxError();
    const snapshot = validateBudgetFxSnapshot(result.data.snapshot, now);
    if (snapshot.id !== id) throw new BudgetFxError();
    return snapshot;
  } finally {
    if (timer) clearTimeout(timer);
  }
}

// backend/supabase/functions/_shared/outfitBudgetFeasibility.ts
function minimumSlotSpend(items, slots, costOf) {
  if (!slots.length) return null;
  const minimum = /* @__PURE__ */ new Map();
  for (const item of items) {
    const slot = item.macroCategory || "";
    if (!slots.includes(slot)) continue;
    const cost = costOf(item);
    if (cost === null || !Number.isFinite(cost) || cost < 0) return null;
    minimum.set(slot, Math.min(minimum.get(slot) ?? Infinity, cost));
  }
  if (slots.some((slot) => !minimum.has(slot))) return null;
  const total = [...new Set(slots)].reduce((sum, slot) => sum + minimum.get(slot), 0);
  return Number.isFinite(total) ? total : null;
}

// src/domain/outfitBudgetSelection.ts
function strictestOutfitBudget(selected, parsed, convert) {
  const validate = (value) => {
    if (value == null) return null;
    if (typeof value !== "object" || Array.isArray(value)) throw new Error("invalid_outfit_budget");
    const candidate = value;
    const currency2 = typeof candidate.currency === "string" ? candidate.currency.trim().toUpperCase() : "";
    if (typeof candidate.amount !== "number" || !Number.isFinite(candidate.amount) || candidate.amount < 0 || !["USD", "EUR", "GBP", "UZS"].includes(currency2)) throw new Error("invalid_outfit_budget");
    return { amount: candidate.amount, currency: currency2 };
  };
  const explicit = validate(selected), text3 = validate(parsed);
  if (!explicit) return text3;
  if (!text3) return explicit;
  const comparable = validate(convert(text3, explicit.currency));
  if (!comparable || comparable.currency !== explicit.currency) throw new Error("invalid_outfit_budget");
  return { amount: Math.min(explicit.amount, comparable.amount), currency: explicit.currency };
}

// backend/supabase/functions/_shared/outfitBudgetFeedback.ts
function outfitBudgetFeedback(outfits, items, budget, convert, eligible) {
  if (!budget || !Number.isFinite(budget.amount) || budget.amount <= 0) return "";
  const currency2 = String(budget.currency || "").toUpperCase();
  const price = (item) => {
    if (!item) return null;
    if (item.isWardrobe === true && !item.isShopItem) return 0;
    if (!item.isShopItem && item.isWardrobe !== false && !String(item.id || "").startsWith("shop_")) return 0;
    if (!Number.isFinite(item.price) || item.price <= 0) return null;
    const source = String(item.currency || "").toUpperCase();
    const cost = source === currency2 ? item.price : convert?.(item.price, source, currency2);
    return typeof cost === "number" && Number.isFinite(cost) ? cost : null;
  };
  const totals = outfits.map((outfit) => {
    const costs = (outfit.items || []).map((item) => price(items.get(String(typeof item === "string" ? item : item?.id))));
    return costs.length && costs.every((cost) => cost !== null) ? costs.reduce((sum, cost) => sum + cost, 0) : null;
  });
  if (!totals.some((total) => total !== null && total > budget.amount)) return "";
  const counts = /* @__PURE__ */ new Map();
  const candidates = [...items.values()].filter((item) => price(item) !== null && (!eligible || eligible(item))).sort((a, b) => price(a) - price(b)).filter((item) => {
    const slot = String(item.macroCategory || item.category || "");
    const count = counts.get(slot) || 0;
    if (count >= 3) return false;
    counts.set(slot, count + 1);
    return true;
  }).map((item) => ({ id: String(item.id), slot: item.macroCategory || item.category, cost: price(item) }));
  const slots = new Set((outfits[0]?.items || []).map((item) => String(items.get(String(typeof item === "string" ? item : item?.id))?.macroCategory || "")));
  const ordinarySlots = [...slots].every((slot) => ["top", "bottom", "shoes", "outerwear"].includes(slot)) && ["top", "bottom", "shoes"].every((slot) => slots.has(slot));
  const example = ordinarySlots ? [...slots].map((slot) => candidates.find((item) => item.slot === slot)) : [];
  const completeExample = example.length >= 3 && example.every(Boolean);
  const exampleTotal = completeExample ? example.reduce((sum, item) => sum + Number(item.cost), 0) : Infinity;
  const guidance = exampleTotal <= budget.amount ? ` A complete candidate combination within the spending cap is ${JSON.stringify(example)}; total ${exampleTotal} ${currency2}. You may select it only if it also meets every style and occasion rule, or choose another valid combination.` : "";
  return `
BUDGET CORRECTION: Hard limit ${budget.amount} ${currency2} PER outfit. Previous catalog totals: ${JSON.stringify(totals)}. Add the prices before choosing IDs; every outfit must include all required slots and stay within the limit. Owned pieces cost zero. Lower-cost candidates (costs expressed in the budget currency): ${JSON.stringify(candidates)}.${guidance} Keep all category, occasion and style requirements; do not omit a required slot to reduce spending.`;
}

// backend/supabase/functions/_shared/modelOutfitResponse.ts
function normalizeModelOutfits(value) {
  if (!Array.isArray(value)) return [];
  return value.filter((outfit) => outfit && typeof outfit === "object" && !Array.isArray(outfit)).map((outfit) => ({
    ...outfit,
    items: Array.isArray(outfit.items) ? outfit.items.map((item) => {
      if (typeof item === "string") return { id: item.trim() };
      if (!item || typeof item !== "object" || Array.isArray(item)) return { id: "" };
      const row = item;
      const id = row.id ?? row.itemId ?? row.item_id;
      return { ...row, id: typeof id === "string" ? id.trim() : "" };
    }) : []
  }));
}

// backend/supabase/functions/_shared/rateLimit.ts
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.45.4";
async function enforceRateLimit(req, opts) {
  const windowSec = opts.windowSec ?? 60;
  const reply = (error, status, retryAfter) => new Response(JSON.stringify({ error, ...retryAfter ? { retry_after_sec: retryAfter } : {} }), {
    status,
    headers: {
      ...opts.headers,
      "content-type": "application/json",
      ...retryAfter ? { "retry-after": String(retryAfter) } : {}
    }
  });
  if (!Number.isSafeInteger(opts.limit) || opts.limit < 1 || !Number.isSafeInteger(windowSec) || windowSec < 1 || !opts.feature) {
    return reply("rate_limit_unavailable", 503);
  }
  const authHeader = req.headers.get("authorization") ?? "";
  const match = /^Bearer\s+(\S+)$/i.exec(authHeader.trim());
  if (!match) return reply("unauthorized", 401);
  const url = Deno.env.get("SUPABASE_URL") || Deno.env.get("SB_URL");
  const serviceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") || Deno.env.get("SB_SERVICE_KEY");
  if (!url || !serviceKey) return reply("rate_limit_unavailable", 503);
  try {
    const admin = createClient(url, serviceKey, {
      auth: { persistSession: false, autoRefreshToken: false }
    });
    const { data: userRes, error: authError } = await admin.auth.getUser(match[1]);
    if (authError) {
      return reply("unauthorized", Number(authError.status) >= 500 ? 503 : 401);
    }
    const userId = userRes?.user?.id;
    if (!userId) return reply("unauthorized", 401);
    const paidStop = (Deno.env.get("AIWARDROBE_PAID_WORK_DISABLED") || "").trim().toLowerCase();
    const disabledFeatures = (Deno.env.get("AIWARDROBE_DISABLED_PAID_FEATURES") || "").split(",").map((value) => value.trim().toLowerCase()).filter(Boolean);
    if (opts.meterPaidWork !== false && (paidStop !== "" && paidStop !== "false" || disabledFeatures.includes(opts.feature.toLowerCase()))) {
      return reply("feature_disabled", 503);
    }
    const nowMs = Date.now();
    const bucketMs = Math.floor(nowMs / (windowSec * 1e3)) * windowSec * 1e3;
    const { data, error } = await admin.rpc("increment_rate_bucket", {
      p_user: userId,
      p_feature: `rl:${opts.feature}:${bucketMs}`
    });
    if (error || !Number.isSafeInteger(data) || data < 1) {
      return reply("rate_limit_unavailable", 503);
    }
    if (data > opts.limit) {
      const retryAfter = Math.max(1, Math.ceil((bucketMs + windowSec * 1e3 - nowMs) / 1e3));
      return reply("rate_limited", 429, retryAfter);
    }
    if (opts.meterPaidWork === false) return null;
    const { data: admitted, error: quotaError } = await admin.rpc("consume_edge_quota", {
      p_user: userId,
      p_endpoint: opts.feature
    });
    if (quotaError || typeof admitted !== "boolean") return reply("rate_limit_unavailable", 503);
    if (!admitted) return reply("feature_limit_reached", 403);
    return null;
  } catch {
    return reply("rate_limit_unavailable", 503);
  }
}

// src/domain/outfitStructure.ts
function isDressGarment(item) {
  if (!item) return false;
  const name = `${item.name || ""} ${item.type || ""}`.toLowerCase();
  if (/\b(shoes?|loafers?|boots?|sneakers?|sandals?|heels?|trainers?|derbys?|derbies|moccasins?|mules?)\b/.test(name)) return false;
  const slot = String(item.macroCategory || "").toLowerCase();
  if (slot && slot !== "top" && slot !== "dress") return false;
  if (slot === "dress") return true;
  return /\bdress(?:es)?\b|плать[ея]/.test(name) && !/\bdress[ -](shirts?|shoes?|pants?|trousers?)\b/.test(name);
}
function isNonOutfitItem(item) {
  const name = typeof item?.name === "string" ? item.name.toLowerCase() : "";
  const type = typeof item?.type === "string" ? item.type.toLowerCase() : "";
  const accessory = /\b(watch(?:es)?|towels?|(?:suit|garment)[ -]+carriers?|key[ -]?rings?|keychains?|(?:hand|tote)?bags?|backpacks?|wallets?|card[ -]?holders?|purses?|belts?|scar(?:f|ves)|hats?|caps?|beanies?|jewell?ery|sunglasses|gloves?|cufflinks?|pocket[ -]?squares?|necklaces?|earrings?|bracelets?|rings?|umbrellas?|socks?|cases?|pouches?|perfumes?|fragrances?)\b/.test(`${name} ${type}`);
  const garment = /\b(t-?shirts?|tees?|shirts?|polos?|blouses?|dress(?:es)?|pants?|trousers?|jeans|shorts?|skirts?|chinos?|jackets?|coats?|blazers?|cardigans?|sweaters?|hoodies?|vests?|shoes?|loafers?|boots?|sneakers?|sandals?)\b/.test(name);
  return accessory && !garment;
}
function outfitStructureReason(items, requireLayer = false) {
  if (!Array.isArray(items) || !items.length) return "empty_outfit";
  const ids = /* @__PURE__ */ new Set();
  const slots = /* @__PURE__ */ new Map();
  let dresses = 0;
  for (const item of items) {
    if (!item || typeof item !== "object") return "invalid_item";
    if (isNonOutfitItem(item)) return "non_outfit_item";
    if (typeof item.id !== "string" || !item.id.trim() || /^(placeholder_|item_|local_)/.test(item.id)) return "invalid_item_id";
    if (ids.has(item.id)) return "duplicate_item";
    ids.add(item.id);
    const slot = item.macroCategory;
    if (!["top", "bottom", "shoes", "outerwear", "dress"].includes(slot)) return "invalid_slot";
    const dress = (slot === "dress" || slot === "top") && isDressGarment(item);
    if (dress) dresses++;
    const normalized = slot === "dress" ? "top" : slot;
    slots.set(normalized, (slots.get(normalized) || 0) + 1);
  }
  if ([...slots.values()].some((count) => count !== 1)) return "duplicate_slot";
  if (!slots.has("shoes") || !slots.has("top")) return "missing_slot";
  if (dresses ? slots.has("bottom") : !slots.has("bottom")) return dresses ? "dress_with_bottom" : "missing_slot";
  if (requireLayer && !slots.has("outerwear")) return "missing_layer";
  return null;
}
function isFourSlotCandidate(item) {
  const text3 = `${item.name || ""} ${item.type || ""}`.toLowerCase();
  const name = String(item.name || "").toLowerCase();
  if (isNonOutfitItem(item) || isDressGarment(item)) return false;
  if (item.macroCategory === "top") {
    if (/\bknit(?:ted|wear)?\b/.test(name) && !/\b(?:polos?|shirts?|t-?shirts?|tees?)\b/.test(name)) return false;
    if (/\b(sweaters?|pullovers?|jumpers?|cardigans?|hoodies?|jackets?|blazers?|coats?|vests?|sweatshirts?)\b|свитер|джемпер|кардиган|толстовк|куртк|пиджак/.test(text3)) return false;
    return /\b(t[ -]?shirts?|tees?|shirts?|polos?)\b|рубашк|футболк|поло/.test(text3);
  }
  if (item.macroCategory === "bottom") {
    if (/\b(t[ -]?shirts?|tees?|shirts?|polos?|blouses?|sweaters?|pullovers?|cardigans?|jackets?|blazers?|coats?)\b|рубашк|футболк|поло|свитер|куртк|пиджак/.test(text3)) return false;
    if (/\b(shorts?|bermudas?|skirts?|trunks?|swimwear)\b|шорт|юбк/.test(text3)) return false;
    return /\b(pants?|trousers?|jeans|chinos?|joggers?|slacks?)\b|брюк|джинс|чинос/.test(text3);
  }
  return item.macroCategory === "outerwear" || item.macroCategory === "shoes";
}
function fourSlotOutfitReason(items, _requireLayer = true) {
  const structural = outfitStructureReason(items, true);
  if (structural) return structural;
  const parts = items;
  const physicalKeys = /* @__PURE__ */ new Set();
  for (const item of parts) {
    const keys = productIdentityKeys(item);
    if (keys.some((key) => physicalKeys.has(key))) return "duplicate_physical_item";
    keys.forEach((key) => physicalKeys.add(key));
  }
  if (parts.length !== 4 || parts.some(isDressGarment)) return "requires_four_categories";
  if (!parts.every(isFourSlotCandidate)) return "invalid_required_category";
  const top = parts.find((item) => item.macroCategory === "top");
  const bottom = parts.find((item) => item.macroCategory === "bottom");
  const topName = `${top.name || ""} ${top.type || ""}`.toLowerCase();
  if (/\b(sweaters?|pullovers?|jumpers?|cardigans?|hoodies?|jackets?|blazers?|coats?|vests?)\b|свитер|джемпер|кардиган|толстовк|куртк|пиджак/.test(topName)) return "base_top_must_be_shirt";
  const bottomName = `${bottom.name || ""} ${bottom.type || ""}`.toLowerCase();
  if (/\b(shorts?|bermudas?|skirts?|trunks?|swimwear)\b|шорт|юбк/.test(bottomName)) return "bottom_must_be_pants";
  return null;
}

// src/domain/occasionRules.ts
function isWorkoutOccasion(value) {
  return /\b(gym|workout|fitness|training|sport|sports)\b|трениров|спортзал|фитнес/i.test(value || "");
}
function rejectForActivity(item, occasion) {
  if (!isWorkoutOccasion(occasion)) return null;
  const slot = macroOf(item);
  const text3 = blobOf(item);
  if (["top", "bottom", "shoes"].includes(slot) && !/\b(athletic|training|running|workout|performance|sportswear|activewear|gym)\b/.test(text3)) {
    return `workout requires verified athletic ${slot}`;
  }
  if (/\b(dress[ -]shirt|formal[ -]shirt|blazer|loafers?|suede)\b/.test(text3)) return "dress clothing is not workout clothing";
  return null;
}
function activityReason(items, occasion) {
  const text3 = occasion.toLowerCase();
  if (isWorkoutOccasion(text3)) {
    for (const slot of ["top", "bottom", "shoes"]) {
      const candidates = items.filter((it) => macroOf(it) === slot);
      if (!candidates.length || candidates.some((it) => !/\b(athletic|training|running|workout|performance|sportswear|activewear|gym)\b/.test(blobOf(it)))) {
        return `workout requires verified athletic ${slot}`;
      }
    }
    for (const item of items) {
      const reason = rejectForActivity(item, occasion);
      if (reason) return reason;
    }
  }
  if (/black[ -]?tie|блэк[ -]?тай|смокинг/.test(text3)) {
    const has = (slot, pattern) => items.some((it) => macroOf(it) === slot && pattern.test(blobOf(it)));
    if (!has("outerwear", /tuxedo|dinner jacket/) || !has("bottom", /tuxedo|dinner suit|black trousers|black pants|formal trousers/) || !has("top", /tuxedo|dress shirt|formal shirt|white shirt/)) return "black tie requires a tuxedo and formal shirt";
    if (!has("shoes", /black/) || !has("shoes", /patent|oxford|opera pump/) || items.some((it) => macroOf(it) === "shoes" && /brown|suede|loafer/.test(blobOf(it)))) return "black tie requires black formal evening shoes";
  }
  return null;
}
function rejectForRequestIntent(item, request, weather) {
  const slot = macroOf(item), text3 = blobOf(item);
  if (slot === "outerwear") {
    const noJacket = /\b(no|without|skip|exclude)\s+(?:a\s+|any\s+)?jackets?\b|без\s+(?:куртк|пиджак)/i.test(request);
    const noCoat = /\b(no|without|skip|exclude)\s+(?:a\s+|any\s+)?coats?\b|без\s+пальто/i.test(request);
    const softLayer = /\b(no|without|skip|exclude)\s+(?:a\s+|any\s+)?(?:layers?|outerwear)\b/i.test(request);
    if (noJacket && /\b(jackets?|blazers?|shackets?)\b/.test(text3)) return "requested a non-jacket layer";
    if (noCoat && /\b(coats?|overcoats?|topcoats?|parkas?|peacoats?)\b/.test(text3)) return "requested a non-coat layer";
    if (softLayer && !/\b(cardigans?|vests?|overshirts?)\b/.test(text3)) return "requested a light alternative layer";
  }
  if (weather?.temp != null && weather.temp < 5 || /\b(snow|freezing|sub[ -]?zero)\b|мороз/i.test(request)) {
    if (slot === "outerwear" && !/\b(puffer|parka|down|insulated|overcoat|topcoat|winter coat|shearling|quilted)\b/.test(text3)) return "cold request requires insulated coat";
    if (slot === "shoes" && !/\b(boots?|chelsea|winter|insulated|snow)\b/.test(text3)) return "cold request requires winter boots";
  }
  if (/black[ -]?tie|блэк[ -]?тай|смокинг/i.test(request)) {
    if (slot === "outerwear" && !/tuxedo|dinner jacket/.test(text3)) return "black tie requires dinner jacket";
    if (slot === "bottom" && !/tuxedo|dinner suit|black trousers|black pants|formal trousers/.test(text3)) return "black tie requires evening trousers";
    if (slot === "top" && !/tuxedo|dress shirt|formal shirt|white shirt/.test(text3)) return "black tie requires formal shirt";
    if (slot === "shoes" && (!/black/.test(text3) || !/patent|oxford|opera pump/.test(text3) || /brown|suede|loafer/.test(text3))) return "black tie requires black evening shoes";
  }
  return null;
}
function occasionFormality(occasionId) {
  const key = String(occasionId || "").toLowerCase().trim();
  if (/\b(work|office|professional|business|interview|date|formal|black[ -]?tie)\b|офис|работ|делов|свидани/.test(key)) return "formal";
  return "casual";
}
var FORMAL_SHOE_RE = /\b(sneakers?|trainers?|athletic|basketball|skate|sandals?|slides?|flip[\s-]?flops?|espadrilles?|boat\b|slippers?|clogs?|runners?|joggers?|mules?)\b|кроссовки|кеды|сандалии|шлёпанцы|шлепанцы|эспадрильи/;
var DRIVER_SHOE_RE = /\b(moccasins?|drivers?|driving|gommino|loafer[s]?\s+with\s+(?:tie|laces?|nubs?)|nubbed|rubber[\s-]?nubs?)\b/;
function isDrivingFootwear(item) {
  let text3 = normalizeGarmentText(`${item.name || ""} ${item.type || ""} ${item.description || ""}`);
  const structuredLoafer = /\b(penny|tassel|bit|structured)\b/.test(text3) && /\bloafer/.test(text3) && /\b(blake|leather outsoles?|leather soles?|structured)\b/.test(text3);
  if (structuredLoafer) text3 = text3.replace(/\bmoccasin[ -]+toe\b/g, "stitched toe");
  return DRIVER_SHOE_RE.test(text3);
}
var CLASSIC_UGLY_SHOE_COLOR_RE = /\b(olive|khaki|moss|sage|lime|chartreuse)\b/;
var MULE_RE = /\b(mules?|backless|slip[\s-]?ons?)\b/;
var MANDARIN_RE = /\b(mandarins?|stand[\s-]?ups?[\s-]?collars?|band[\s-]?collars?|grandad[\s-]?collars?)\b/;
var WORKISH_RE = /\b(work|office|interviews?|meetings?|business|corporate|conferences?)\b/;
var PRINT_TOP_RE = /\b(graphic|logo|jacquard|camo|tie[\s-]?dye|sequins?|rhinestones?)\b/;
var CASUAL_LAYER_RE = /\b(puffer|quilted|bombers?|hoodies?|sweatshirts?|windbreakers?|parkas?|anoraks?|fleece vest|ski\b)\b/;
var ZIP_CASUAL_LAYER_RE = /\b(zip[\s-]?up|zipped|total[\s-]?look|overshirt|shacket)\b/;
var RELAXED_FIT_RE = /\brelaxed[\s-]?fit\b/;
var PLAID_TOP_RE = /\bplaid\b/;
var CASUAL_LOAFER_RE = /\b(split[\s-]?leather|jute|cupsole|cup[\s-]?sole|crepe[\s-]?sole|gum[\s-]?sole|white[\s-]?sole|caramel[\s-]?sole|moc[\s-]?toe)\b/;
var SLOPPY_BOTTOM_RE = /\b(cargo|ripped|distressed|sweatpants?|joggers?|jogging|basketball|board\s*shorts?|drawstrings?|elasticated|\belastic(?:ated)?[\s-]?waist(?:band)?|co-?ords?|tracksuits?|track[\s-]?(?:pants?|trousers?|suits?)|balloon[\s-]?fit)\b/;
var FASHION_WORK_BOTTOM_RE = /\b(washed[\s-]?effect|bell[\s-]?bottoms?|splatter|uneven[\s-]?hem|twisted[\s-]?seams?|barrel[\s-]?fit)\b/;
var FASHION_WORK_LAYER_RE = /\b(washed[\s-]?effect)\b/;
var PIQUE_POLO_RE = /\bpiqu[eé]/i;
var MINIMALIST_NOISE_RE = /\b(graphic|logo|camo|neon|rhinestones?|sequins?|animal\s*print|monogram)\b/;
var LOUD_PRINT_RE = /\b(graphics?|logos?|florals?|paisleys?|prints?|printed|camos?|tie[\s-]?dyes?|neons?|sequins?|rhinestones?|souvenirs?|sukajans?|embroidered|embroidery|animal\s*prints?|lettering|slogans?|texts?|palms?|tropical|hawaiian)\b/;
var UNCLASSIC_FIT_RE = /\b(cropped|boxy|oversized|shrunken)\b/;
var SOCCER_RE = /\bsoccers?\b/;
var SLEEPWEAR_RE = /\b(pajamas?|pyjamas?|bathrobes?|slippers?|loungewear|sleepwear|nightgowns?|nightwear|dressing\s*gowns?)\b/;
var CLASSIC_STYLES = /* @__PURE__ */ new Set(["old_money", "classic", "business_casual"]);
var SUMMER_OPEN_SHOE_RE = /\b(espadrilles?|sandals?|slides?|flip[\s-]?flops?|jandals?)\b/;
var HEAVY_WINTER_LAYER_RE = /\b(puffer|parka|ski\b|down\s+jacket|overcoats?|topcoats?|peacoats?|duffle|greatcoats?)\b/;
var WARM_FIBER_RE = /\b(wool|merino|cashmere|flannel|tweed|felt|fleece|boiler)\b/;
var WARM_LAYER_RE = /\b(coat|jacket|blazer|cardigan|sweater|pullover|knit|trench|peacoat|hoodie|overshirt)\b/;
function isWarmWeatherOnlyFabric(item) {
  if (!item) return false;
  const blob = `${item.name || ""} ${item.type || ""} ${item.description || ""}`.toLowerCase();
  if (/\b(seersucker|voile|gauze|tropical\s+wool)\b/.test(blob)) return true;
  const linen = /\blinen\b/.test(blob);
  if (!linen) return false;
  if (/\b(knit|sweater|cardigan|pullover|jumper)\b/.test(blob)) return false;
  return !WARM_FIBER_RE.test(blob);
}
function rejectForClimate(item, weather) {
  if (!item || !weather || typeof weather.temp !== "number" || Number.isNaN(weather.temp)) return null;
  const temp = weather.temp;
  const blob = `${item.name || ""} ${item.type || ""} ${item.brand || ""} ${item.description || ""}`.toLowerCase();
  const blobSansSleeves = blob.replace(STRIP_SHORT_SLEEVE_RE, " ");
  const slot = String(item.macroCategory || "").toLowerCase();
  const cold = temp < 18;
  const hot = temp >= 26;
  if (cold && isWarmWeatherOnlyFabric(item)) {
    return `${temp}\xB0C is too cold for linen / summer cloth`;
  }
  if (cold && SUMMER_OPEN_SHOE_RE.test(blob)) {
    return `${temp}\xB0C is too cold for open summer shoes`;
  }
  if (cold && /\b(shorts?|bermudas?|trunks?|swim)\b/.test(blobSansSleeves) && (slot === "bottom" || slot === "")) {
    return `${temp}\xB0C is too cold for shorts`;
  }
  if (hot && slot === "outerwear" && HEAVY_WINTER_LAYER_RE.test(blob)) {
    return `${temp}\xB0C is too warm for heavy winter outerwear`;
  }
  if (hot && slot !== "shoes" && /\b(wool|merino|alpaca|cashmere|mohair|angora|lambswool|fleece|shearling|chunky|heavyweight|insulated|padded)\b/.test(blob) && !/\btropical\s+wool\b/.test(blob)) {
    return `${temp}\xB0C requires ${slot === "outerwear" ? "a light layer" : "light clothing"}, not warm winter fabric`;
  }
  if (hot && slot === "outerwear" && (/\b(leather|suede|corduroy)\b/.test(blob) || !/\b(lightweight|light[ -]weight|fine[ -]gauge|open[ -]knit|summer|linen|cotton|breathable|unlined)\b/.test(blob))) {
    return `${temp}\xB0C requires a documented lightweight layer`;
  }
  if (hot && slot === "outerwear" && /\b(sweaters?|pullovers?|hoodies?|sweatshirts?)\b|funnel[ -]neck/.test(blob) && !/\b(lightweight|light[ -]weight|fine[ -]gauge|open[ -]knit|summer|linen)\b/.test(blob)) {
    return `${temp}\xB0C requires a documented lightweight layer`;
  }
  if (temp >= 28 && /\b(tweed|flannel)\b/.test(blob) && slot !== "shoes") {
    return `${temp}\xB0C is too warm for tweed/flannel`;
  }
  return null;
}
function rejectOutfitClimate(items, weather) {
  if (!items || items.length === 0) return null;
  if (!weather || typeof weather.temp !== "number" || Number.isNaN(weather.temp)) return null;
  if (weather.temp >= 18) return null;
  const hasWarmLayer = items.some((it) => {
    const slot = String(it.macroCategory || "").toLowerCase();
    const blob = `${it.name || ""} ${it.type || ""} ${it.description || ""}`.toLowerCase();
    if (slot !== "outerwear" && slot !== "") return false;
    if (!WARM_LAYER_RE.test(blob) && slot !== "outerwear") return false;
    return !isWarmWeatherOnlyFabric(it);
  });
  if (!hasWarmLayer) return `${weather.temp}\xB0C needs a warm layer, not summer cloth`;
  if (weather.temp < 5) {
    const hasInsulated = items.some((it) => macroOf(it) === "outerwear" && /\b(puffer|parka|down|insulated|overcoat|topcoat|winter coat|shearling|quilted)\b/.test(blobOf(it)));
    if (!hasInsulated) return `${weather.temp}\xB0C requires an insulated coat`;
    const hasWinterBoots = items.some((it) => macroOf(it) === "shoes" && /\b(boots?|chelsea|winter|insulated|snow)\b/.test(blobOf(it)));
    if (!hasWinterBoots) return `${weather.temp}\xB0C requires winter boots`;
  }
  return null;
}
var SQUARE_TOE_RE = /\bsquare[\s-]?toes?\b/;
var EXOTIC_RE = /\b(crocodile|alligator|pythons?|stingray|ostrich|lizard)(\s*(leather|skin))?\b|\bexotic\s*leathers?\b/;
var BASIC_TEE_RE = /\bt-?shirts?\b|\btees?\b|\btanks?\b/;
function garmentTypeEvidence(item) {
  const name = normalizeGarmentText(item.name || "");
  const type = String(item.type || "");
  return /\b(polos?|shirts?)\b|поло/.test(name) && !BASIC_TEE_RE.test(name) && /^(tees?|t[ -]?shirts?)$/i.test(type.trim()) ? /\bpolos?\b/.test(name) ? "polo" : "shirt" : type;
}
function rejectForStyleOccasion(item, styleId, occasionId) {
  if (!item) return "empty item";
  const refinedDate = refinedDateProductReason(item, styleId, occasionId);
  if (refinedDate) return refinedDate;
  const blob = blobOf(item);
  if (UNDERWEAR_RE.test(blob)) return "underwear must never surface as visible clothing";
  if (/\b(iphone|phone[\s-]?cases?|airpods?)\b/.test(blob)) return "tech accessories are not outfit pieces";
  const style = isWorkoutOccasion(occasionId) ? "sport" : String(styleId || "").toLowerCase().replace(/[\s-]+/g, "_");
  const normalizedContext = stylingContext({ style: styleId, occasion: occasionId });
  const contextFailure = normalizedContext.dressCode === "smart" ? null : garmentContextReason(item, normalizedContext);
  if (contextFailure) return contextFailure;
  const formal = occasionFormality(occasionId) === "formal";
  const slot = String(item.macroCategory || "").toLowerCase();
  const isTopOrLayer = slot === "" || slot === "top" || slot === "outerwear";
  const isBottom = slot === "bottom";
  const occasion = String(occasionId || "").toLowerCase().trim();
  if (occasion === "travel") {
    if ((slot === "outerwear" || slot === "" || isTopOrLayer) && BOARDROOM_LAYER_RE.test(blob)) {
      return "boardroom suiting is not a travel look";
    }
    if ((slot === "outerwear" || slot === "") && /\b(tuxedos?|suit\s*jackets?)\b/.test(blob)) {
      return "suit jackets are not a travel look";
    }
  }
  if (formal) {
    if (FORMAL_SHOE_RE.test(blob)) return "beach/athletic shoes are not evening wear";
    if (isShoeish(item, blob) && /\bchunky\b/.test(blob)) return "chunky soles cheapen polished footwear";
    if (isDrivingFootwear(item)) return "driving moccasins are a tier below evening footwear";
    if (MULE_RE.test(blob)) return "mules and backless shoes are not evening wear";
    if (isTopOrLayer && MANDARIN_RE.test(blob)) return "mandarin collars cheapen polished lines";
    if (PRINT_TOP_RE.test(blob)) return "printed tops clash with a polished evening look";
    if ((isTopOrLayer || isBottom) && hasLoudPrint(blob)) return "loud prints clash with a polished evening look";
    if (isTopOrLayer && UNCLASSIC_FIT_RE.test(blob)) return "cropped/boxy fits break classic proportions";
    if (isTopOrLayer && SOCCER_RE.test(blob)) return "soccer kits are fan wear, not evening wear";
    if (isTopOrLayer && (CASUAL_LAYER_RE.test(blob) || ZIP_CASUAL_LAYER_RE.test(blob))) {
      return "casual outerwear underdresses the occasion";
    }
    if (slot === "top" && PLAID_TOP_RE.test(blob)) return "fashion plaid shirts underdress a polished look";
    if (isTopOrLayer && RELAXED_FIT_RE.test(blob)) return "relaxed-fit fashion tops break polished proportions";
    if (isBottom && RELAXED_FIT_RE.test(blob)) return "relaxed-fit trousers underdress a polished look";
    if ((slot === "top" || slot === "") && SHORT_SLEEVE_RE.test(blob) && !/\b(knit|polo)\b/.test(blob)) {
      return "short sleeves underdress a polished look";
    }
    if (isShoeish(item, blob) && CASUAL_LOAFER_RE.test(blob)) {
      return "split-leather / gum-sole loafers are not evening footwear";
    }
    if (SLOPPY_BOTTOM_RE.test(blob)) return "sloppy bottoms underdress the occasion";
    if (isBottom && FASHION_WORK_BOTTOM_RE.test(blob)) return "fashion denim/splatter bottoms underdress work";
    if (isTopOrLayer && FASHION_WORK_LAYER_RE.test(blob)) return "washed fashion jackets underdress the occasion";
    if (WORKISH_RE.test(occasion) && (slot === "top" || slot === "") && PIQUE_POLO_RE.test(blob) && /\bpolo\b/.test(blob)) {
      return "piqu\xE9 polos are sporty, not workwear";
    }
    if (SLEEPWEAR_RE.test(blob)) return "sleepwear is not daywear for a polished occasion";
    if (isBottom && /\bjeans?\b/.test(blob)) return "denim bottoms underdress a formal look";
  }
  const workish = WORKISH_RE.test(String(occasionId || "").toLowerCase());
  if ((CLASSIC_STYLES.has(style) || workish) && isBottom && /\bjeans?\b/.test(blob)) {
    return "denim bottoms never belong in classic or work decks";
  }
  if (workish && isTopOrLayer && /\bdenim\b/.test(blob) && /\bshirts?\b/.test(blob)) {
    return "denim shirts underdress a work look";
  }
  if (workish && (slot === "outerwear" || slot === "") && isKnitOnlyLayerBlob(blob)) {
    return "knit sweaters are not work outerwear";
  }
  if (CLASSIC_STYLES.has(style)) {
    const casualActivity = normalizedContext.dressCode === "casual" && /\b(casual|travel|trip)\b/.test(occasion);
    const practicalClassicSneaker = casualActivity && /\b(leather)\b/.test(blob) && /\b(sneakers?|trainers?)\b/.test(blob) && !/\b(chunky|athletic|running|training|basketball|skate|mesh|logo|graphic)\b/.test(blob);
    const practicalClassicLayer = casualActivity && slot === "outerwear" && /\b(overshirts?|shackets?)\b/.test(blob) && /\b(cotton|linen|lightweight|unlined)\b/.test(blob) && !/\b(bulky|heavyweight|padded|cargo|military)\b/.test(blob);
    if (isShoeish(item, blob) && /\bchunky\b/.test(blob)) return "chunky soles break classic discipline";
    if (SLEEPWEAR_RE.test(blob)) return "sleepwear breaks classic discipline";
    if (isDrivingFootwear(item)) return "driving moccasins never pair with classic tailoring";
    if (isShoeish(item, blob) && FORMAL_SHOE_RE.test(blob) && !practicalClassicSneaker) {
      return "boat/athletic shoes never pair with classic tailoring";
    }
    if (isShoeish(item, blob) && CLASSIC_UGLY_SHOE_COLOR_RE.test(blob)) {
      return "olive/khaki casual shoes clash with classic tailoring";
    }
    if (isShoeish(item, blob) && CASUAL_LOAFER_RE.test(blob)) {
      return "split-leather / gum-sole loafers never pair with classic tailoring";
    }
    if (MULE_RE.test(blob)) return "mules never pair with classic tailoring";
    if (isTopOrLayer && (CASUAL_LAYER_RE.test(blob) || ZIP_CASUAL_LAYER_RE.test(blob)) && !practicalClassicLayer) {
      return "casual outerwear breaks classic discipline";
    }
    if (isTopOrLayer && RELAXED_FIT_RE.test(blob)) return "relaxed-fit fashion tops break classic proportions";
    if (isBottom && /\b(ripped|distressed|sweatpants?|joggers?|jogging|track|tracksuits?|velour|baggy|cargo|balloon[\s-]?fit)\b/.test(blob)) {
      return "sloppy/track bottoms break classic discipline";
    }
    if (isBottom && FASHION_WORK_BOTTOM_RE.test(blob) && formal) {
      return "fashion denim/splatter bottoms break classic discipline";
    }
    if ((macroOf(item) === "top" || macroOf(item) === "") && BASIC_TEE_RE.test(blob) && !/\bknit\b/.test(blob)) {
      return "basic tees break classic discipline";
    }
    if (isTopOrLayer && hasLoudPrint(blob)) return "loud prints break classic discipline";
    if (isTopOrLayer && UNCLASSIC_FIT_RE.test(blob)) return "cropped/boxy fits break classic proportions";
    if (isTopOrLayer && SOCCER_RE.test(blob)) return "soccer kits break classic discipline";
    if (isTopOrLayer && MANDARIN_RE.test(blob)) return "mandarin collars break classic lines";
  }
  if (detectSlotMismatch(item.macroCategory, item.name, item.type) !== null) {
    return "garment name and outfit slot disagree (mislabeled catalog row)";
  }
  if (style === "minimalist" && MINIMALIST_NOISE_RE.test(blob)) {
    return "loud piece breaks minimalist discipline";
  }
  if (style === "minimalist" && isTopOrLayer && hasLoudPrint(blob)) {
    return "loud prints break minimalist discipline";
  }
  if (style === "minimalist" && /\bpleats?\b|\bpleated\b/.test(blob)) {
    return "minimalism takes no pleats";
  }
  if (style === "minimalist" && isBottom && /\b(ripped|distressed|joggers?|jogging|baggy|track|tracksuits?|balloon[\s-]?fit)\b/.test(blob)) {
    return "ripped/baggy bottoms break minimalist discipline";
  }
  if (style === "semi_classic") {
    if (isShoeish(item, blob) && /\bchunky\b/.test(blob)) return "chunky soles break semi-classic polish";
    if (isBottom && /\b(ripped|distressed|sweatpants?|joggers?|jogging|tracksuits?|track[\s-]?(?:pants?|trousers?)|baggy|cargo|oversized|balloon[\s-]?fit)\b/.test(blob)) {
      return "sloppy/track bottoms break semi-classic discipline";
    }
  }
  if (style === "casual" && formal) {
    if (/\b(chunky|athletic|basketball|running|skate|fabric|mesh)\b/.test(blob) && SNEAKER_RE.test(blob)) {
      return "refined casual takes only minimal leather sneakers";
    }
    if (/\bhoodies?|hooded\b/.test(blob)) return "refined casual never wears hoodies";
    if (macroOf(item) !== "shoes" && BASIC_TEE_RE.test(blob) && !/\bknit\b/.test(blob)) {
      return "refined casual wears knit polos, not basic cotton tees";
    }
  }
  if (formal && (macroOf(item) === "top" || macroOf(item) === "") && BASIC_TEE_RE.test(blob) && !/\bknit\b/.test(blob)) {
    return "basic tees are underdressed for polished occasions";
  }
  if (SQUARE_TOE_RE.test(blob)) return "square toes cheapen any outfit";
  if ((CLASSIC_STYLES.has(style) || formal) && EXOTIC_RE.test(blob)) return "exotic leathers read vulgar with classics";
  return null;
}
function hasLoudPrint(blob) {
  if (!LOUD_PRINT_RE.test(blob)) return false;
  return true;
}
var FORMAL_LAYER_RE = /\b(blazers?|suit\s*jackets?|sport\s*coats?|overcoats?|top\s*coats?|topcoats?|trenchs?|peacoats?|pea\s*coats?|tuxedos?)\b/;
var BOARDROOM_LAYER_RE = /\b(double[\s-]?breasted|suit\s*(jacket|blazer)|cool wool suit)\b/;
function isKnitOnlyLayerBlob(blob) {
  if (/\b(blazer|jacket|coat|trench|peacoat|bomber)\b/.test(blob)) return false;
  return /\b(sweater|pullover|jumper|cardigan)\b/.test(blob);
}
var DRESS_SHIRT_RE = /\b(oxford|button[\s-]?downs?|dress\s*shirts?)\b/;
var SNEAKER_RE = /\b(?:sneakers?|trainers?)\b|кроссовки|кеды|кеда/;
var BLACK_RE = /\bblack\b/;
var WHITE_RE = /\bwhite\b/;
var STRIPED_LAYER_RE = /\bstripes?|pinstripes?\b/;
function blobOf(item) {
  return normalizeGarmentText(`${item.name || ""} ${garmentTypeEvidence(item)} ${item.description || ""} ${item.color || ""}`);
}
function macroOf(item) {
  return String(item.macroCategory || "").toLowerCase();
}
function isShoeish(item, blob) {
  const m = macroOf(item);
  if (m === "shoes" || m === "shoe" || m === "footwear" || m === "foot_wear") return true;
  return /\b(loafers?|shoes?|moccasins?|derb(?:y|ies)|oxfords?|sneakers?|trainers?|boots?)\b/.test(blob);
}
var TAILORED_PANTS_RE = /\b(?:suit|tailored|pinstriped?|flannel|total[\s-]?look)\b[\w\s-]{0,30}\b(?:trousers?|pants?|slacks?)\b|\b(?:trousers?|pants?|slacks?)\b[\w\s-]{0,30}\b(?:suit|tailored|pinstriped?|flannel|total[\s-]?look)\b/;
var OUTER_LAYER_CLASS_RE = /\b(?:blazers?|suit\s*jackets?|sport\s*coats?|overcoats?|top\s*coats?|trench(?:es)?|peacoats?|puffers?|parkas?|bombers?|gilets?|vests?|windbreakers?|coats?|jackets?|outerwears?)\b|пальто|куртк|жилет|жакет|парк/i;
var PULLOVER_KNIT_CLASS_RE = /\b(?:sweaters?|pullovers?|jumpers?|cardigans?|knit(?:ted)?|mock[\s-]?necks?|polos?)\b|свитт?ер|пуловер|кардиган|водолазк/i;
var ATHLETIC_SOLE_RE = /\b(?:sneakers?|trainers?|hybrids?)\b|кроссовки|кеды|кеда/;
function outfitPaletteReason(items) {
  const vivid = /* @__PURE__ */ new Set();
  for (const item of items) {
    const color = normalizeGarmentText(String(item.color || ""));
    for (const component of color.split(/[,/;&+]|\band\b/)) {
      if (/\b(navy|olive|khaki|sage|muted|dusty|pastel|pale|light|dark|burgundy|maroon|mustard|rust|forest)\b/.test(component)) continue;
      for (const hue of ["red", "orange", "yellow", "green", "blue", "purple", "pink"]) {
        if (new RegExp(`\\b${hue}\\b`).test(component)) vivid.add(hue);
      }
    }
  }
  if (vivid.size >= 3) return "too many competing vivid colors";
  if ([["red", "green"], ["blue", "orange"], ["yellow", "purple"]].some((pair) => pair.every((hue) => vivid.has(hue)))) {
    return "competing saturated complementary colors";
  }
  return null;
}
function rejectOutfitIncoherence(items, styleId, occasionId) {
  if (!items || items.length === 0) return null;
  const style = isWorkoutOccasion(occasionId) ? "sport" : String(styleId || "").toLowerCase().replace(/[\s-]+/g, "_");
  const formal = occasionFormality(occasionId) === "formal";
  const classic = CLASSIC_STYLES.has(style);
  const palette = outfitPaletteReason(items);
  if (palette) return palette;
  const hasAthleticSole = items.some(
    (it) => (macroOf(it) === "shoes" || macroOf(it) === "") && ATHLETIC_SOLE_RE.test(blobOf(it))
  );
  const hasTailoredPants = items.some(
    (it) => (macroOf(it) === "bottom" || macroOf(it) === "") && TAILORED_PANTS_RE.test(blobOf(it))
  );
  if (hasAthleticSole && hasTailoredPants) {
    return "sneakers never pair with tailored trousers";
  }
  const tops = items.filter((it) => {
    const m = macroOf(it);
    return m === "top" || m === "outerwear" || m === "";
  });
  for (let i = 0; i < tops.length; i++) {
    for (let j = i + 1; j < tops.length; j++) {
      const a = blobOf(tops[i]);
      const b = blobOf(tops[j]);
      if (OUTER_LAYER_CLASS_RE.test(a) && OUTER_LAYER_CLASS_RE.test(b)) {
        return "two outer coats never layer together";
      }
      const openCardigan = (item, text3) => macroOf(item) === "outerwear" && /\bcardigans?\b/.test(text3) && !/\bpullover\b/.test(text3);
      if (PULLOVER_KNIT_CLASS_RE.test(a) && PULLOVER_KNIT_CLASS_RE.test(b) && !openCardigan(tops[i], a) && !openCardigan(tops[j], b)) {
        return "two pullover knits never layer together";
      }
    }
  }
  if (items.some(
    (it) => (macroOf(it) === "top" || macroOf(it) === "bottom" || macroOf(it) === "outerwear") && isAccessoryPropItem(it)
  )) {
    return "accessory prop never fills a garment slot";
  }
  const registerClash = wholeOutfitRegisterClash(
    items,
    stylingContext({ style, occasion: occasionId })
  );
  if (registerClash) {
    return registerClash;
  }
  if (!formal && !classic) return null;
  const hasFormalLayer = items.some(
    (it) => (macroOf(it) === "outerwear" || macroOf(it) === "") && FORMAL_LAYER_RE.test(blobOf(it))
  );
  const hasDressShirt = items.some(
    (it) => (macroOf(it) === "top" || macroOf(it) === "") && DRESS_SHIRT_RE.test(blobOf(it))
  );
  const hasSneakers = items.some(
    (it) => (macroOf(it) === "shoes" || macroOf(it) === "") && SNEAKER_RE.test(blobOf(it))
  );
  if (hasSneakers && (hasFormalLayer || hasDressShirt)) {
    return "sneakers never pair with tailoring or dress shirts";
  }
  const hasBasicTee = items.some((it) => {
    if (macroOf(it) !== "top" && macroOf(it) !== "") return false;
    const b = blobOf(it);
    return BASIC_TEE_RE.test(b) && !/\bknit\b/.test(b);
  });
  if (hasBasicTee && hasFormalLayer) {
    return "basic tees never go under tailoring (knit polos do)";
  }
  const hasPiqueTop = items.some((it) => {
    if (macroOf(it) !== "top" && macroOf(it) !== "") return false;
    return /\bpiqu[eé]/i.test(blobOf(it));
  });
  if (hasPiqueTop && hasFormalLayer) {
    return "piqu\xE9 polos are sporty \u2014 knit polos go under tailoring";
  }
  const hasCheapShortSleeve = items.some((it) => {
    if (macroOf(it) !== "top" && macroOf(it) !== "") return false;
    const b = blobOf(it);
    return SHORT_SLEEVE_RE.test(b) && (/\bshirts?\b/.test(b) || BASIC_TEE_RE.test(b)) && !/\bhawaiian|camp[\s-]?collar|cuban\b/.test(b);
  });
  if (hasCheapShortSleeve && (classic || formal)) {
    return "classic short-sleeve shirts read cheap (Hawaiian camp excepted)";
  }
  const hasSuedeShoes = items.some(
    (it) => (macroOf(it) === "shoes" || macroOf(it) === "") && /\bsuede\b/.test(blobOf(it))
  );
  const hasSuitPiece = items.some((it) => /\bsuits?\b/.test(blobOf(it)));
  if (hasSuedeShoes && hasSuitPiece) {
    return "suede stays in smart casual \u2014 suits take leather";
  }
  const hasWhiteShirt = items.some((it) => {
    if (macroOf(it) !== "top" && macroOf(it) !== "") return false;
    const b = blobOf(it);
    return /\bwhite\b/.test(b) && /\bshirts?|oxford\b/.test(b);
  });
  const hasJeans = items.some((it) => {
    if (macroOf(it) !== "bottom" && macroOf(it) !== "") return false;
    return /\bjeans?|denim\b/.test(blobOf(it));
  });
  if (hasWhiteShirt && hasJeans) {
    return "a white shirt never goes with jeans";
  }
  const workish = WORKISH_RE.test(String(occasionId || "").toLowerCase());
  if ((classic || workish) && hasJeans) {
    return "denim never belongs in classic or work decks";
  }
  if (classic || formal && style !== "minimalist") {
    const hasBlackCloth = items.some((it) => {
      const m = macroOf(it);
      return (m === "top" || m === "outerwear" || m === "bottom" || m === "") && BLACK_RE.test(blobOf(it));
    });
    const hasWhite = items.some((it) => WHITE_RE.test(blobOf(it)));
    if (hasBlackCloth && !hasWhite) return "black pairs with nothing but white";
  }
  const stripedLayer = items.find((it) => {
    const m = macroOf(it);
    const looksOuter = m === "outerwear" || m === "" && /\b(jacket|coat|blazer|sport\s*coat)\b/.test(blobOf(it));
    return looksOuter && STRIPED_LAYER_RE.test(blobOf(it));
  });
  if (stripedLayer && (classic || formal)) {
    const bottoms = items.filter((it) => macroOf(it) === "bottom");
    const layerColor = String(stripedLayer.color || "").toLowerCase().trim();
    const matched = bottoms.some((b) => {
      const bBlob = blobOf(b);
      if (!STRIPED_LAYER_RE.test(bBlob)) return false;
      const bColor = String(b.color || "").toLowerCase().trim();
      return layerColor !== "" && bColor !== "" && layerColor === bColor;
    });
    if (!matched) return "striped jackets only go with their matching trousers";
  }
  return null;
}
var TOP_WORDS_RE = /\bt-?shirts?\b|\btees?\b|\bpolos?\b|\bblouses?\b|\btanks?\b|\bhenleys?\b|\bshirts?\b|\boxfords?\b|\bbutton[\s-]?downs?\b/;
function isOxfordFootwearName(name) {
  const text3 = normalizeGarmentText(name);
  return /\boxfords?\b/.test(text3) && !/\b(shirts?|blouses?|polos?|jackets?|coats?|blazers?|trousers?|pants?|shorts?|skirts?|dresses?)\b/.test(text3);
}
var OUTER_WORDS_RE = /\b(?:jackets?|coats?|blazers?|hoodies?|cardigans?|sweaters?|pullovers?|puffers?|bombers?|vests?|outerwears?|trenchs?|peacoats?|sweatshirts?|knitwears?|jumpers?|fleece|overshirts?|shackets?|total[\s-]?look)\b/;
var BOTTOM_WORDS_RE = /\bpants?|trousers?|jeans?|bottoms?|shorts|bermudas?|skirts?|chinos?|slacks?|joggers?|jogging|trunks?|swim/;
var SHOE_WORDS_RE = /\bshoes?|sneakers?|boots?|loafers?|sandals?|heels?|trainers?|derbys?|derbies|moccasins?|drivers?|mules?|clogs?|espadrilles?\b|кроссовки|кеды|туфли|ботинки|лоферы|сапоги|мокасины|сандалии/;
var UNDERWEAR_RE = /\bboxers?|briefs?|underwears?|panties?|thongs?|lingerie|boxer[\s-]?shorts?|nightwear|pyjamas?|pajamas?\b/;
var ACCESSORY_PROP_RE = /\b(sunglasses?|eyewear|glasses|card[\s-]?holders?|wallets?|belts?|bags?|totes?|scar(?:f|ves)|hats?|caps?|beanies?|watch(?:es)?|jewelry|earrings?|necklaces?|ties?|neckties?|key[\s-]?rings?|keyrings?|towels?)\b|брелок|полотен/;
var STRIP_SHORT_SLEEVE_RE = /\bshort[\s-]?sleeve(?:s|d)?\b/gi;
function isAccessoryPropItem(item) {
  if (!item) return false;
  const macro = String(item.macroCategory || "").toLowerCase();
  if (macro === "accessory" || macro === "accessories") return true;
  return ACCESSORY_PROP_RE.test(`${item.name || ""} ${item.type || ""} ${item.brand || ""} ${item.description || ""}`.toLowerCase());
}
function detectSlotMismatch(macro, name, type) {
  const raw = String(macro || "").toLowerCase().trim();
  const alias = {
    top: "top",
    tops: "top",
    upper_body: "top",
    "upper-body": "top",
    upperbody: "top",
    dress: "top",
    dresses: "top",
    bottom: "bottom",
    bottoms: "bottom",
    lower_body: "bottom",
    "lower-body": "bottom",
    lowerbody: "bottom",
    pant: "bottom",
    pants: "bottom",
    outerwear: "outerwear",
    outer: "outerwear",
    shoe: "shoes",
    shoes: "shoes",
    footwear: "shoes"
  };
  const claimed = alias[raw];
  if (!claimed) {
    return null;
  }
  const blob = String(name || "").toLowerCase();
  const blobSansSleeves = blob.replace(STRIP_SHORT_SLEEVE_RE, " ");
  const oxfordFootwear = isOxfordFootwearName(blob);
  const hasTop = TOP_WORDS_RE.test(blob) && !oxfordFootwear;
  const hasOuter = OUTER_WORDS_RE.test(blob);
  const hasBottom = BOTTOM_WORDS_RE.test(blobSansSleeves);
  const hasShoe = SHOE_WORDS_RE.test(blob) || oxfordFootwear;
  if (SHORT_SLEEVE_RE.test(blob)) {
    if (claimed === "top") return null;
    return "top";
  }
  if ((claimed === "top" || claimed === "bottom") && hasOuter && !hasTop && /\b(jacket|coat|blazer|trench|peacoat|bomber|puffer)\b/.test(blob)) {
    return "outerwear";
  }
  const supportsClaim = claimed === "top" && hasTop || claimed === "outerwear" && hasOuter || claimed === "bottom" && hasBottom || claimed === "shoes" && hasShoe;
  if (supportsClaim) return null;
  if (hasShoe) return "shoes";
  if (hasBottom) return "bottom";
  if (hasOuter) return "outerwear";
  if (hasTop) return "top";
  return null;
}
function styleJudgeReason(items, styleId, occasionId, weather, lockedItemId, preferences) {
  if (!items || items.length === 0) return "empty outfit";
  const activity = activityReason(items, occasionId || "");
  if (activity) return activity;
  if (isWorkoutOccasion(occasionId)) styleId = "sport";
  for (const it of items) {
    const r = rejectForStyleOccasion(it, lockedItemId && String(it.id) === lockedItemId ? "casual" : styleId, occasionId);
    if (r !== null) return `item "${String(it?.name || "?").slice(0, 40)}": ${r}`;
  }
  for (const it of items) {
    const intent = rejectForRequestIntent(it, occasionId || "", weather);
    if (intent) return `request: ${intent}`;
    const r = rejectForClimate(it, weather);
    if (r !== null) return `climate "${String(it?.name || "?").slice(0, 40)}": ${r}`;
  }
  const oc = rejectOutfitClimate(items, weather);
  if (oc !== null) return `outfit climate: ${oc}`;
  return rejectOutfitIncoherence(items, styleId, occasionId) || assessOutfitQuality(items, styleId, occasionId, weather, lockedItemId, preferences).reasons[0] || null;
}
function isOldMoneyDate(style, occasion) {
  return String(style || "").toLowerCase().replace(/[\s-]+/g, "_") === "old_money" && /\bdate\b|свидани/i.test(occasion || "");
}
function refinedDateProductReason(item, style, occasion) {
  if (!isOldMoneyDate(style, occasion)) return null;
  const text3 = blobOf(item), slot = macroOf(item);
  if (slot === "top" && /\bwestern[ -]shirts?\b|\bshirts?\b.{0,25}\bwestern\b/.test(text3)) return "Western shirt construction conflicts with refined evening styling";
  if (slot === "top" && /\bripstop\b/.test(text3) && /\bshirts?\b/.test(text3)) return "utility ripstop shirt construction underdresses the date";
  if (slot === "outerwear" && /\bdenim\b/.test(text3) && /\b(blazers?|jackets?)\b/.test(text3)) return "denim tailoring remains too casual for this refined date";
  if (slot === "outerwear" && /\btechnical\b/.test(text3) && /\b(leisure|elastic|sports?|running)\b/.test(text3)) return "technical leisure construction conflicts with refined evening tailoring";
  if (slot === "bottom" && /\b(sports?|athletic|training|performance)\b/.test(text3) && /\b(pants?|trousers?)\b/.test(text3)) return "athletic trouser construction is not refined evening wear";
  if (slot === "top" && /\brelaxed\b/.test(text3) && /\b(shirt|button[ -]down|poplin|pinpoint)\b/.test(text3)) return "relaxed shirt silhouette underdresses this polished date";
  if (["top", "outerwear", "bottom"].includes(slot) && /\b(acid[ -]washed|stone[ -]washed|destroyed|distressed)\b/.test(text3)) return "distressed surface treatment conflicts with understated evening clothing";
  if (slot === "top" && /\b(terry|towelling|toweling|towel)\b/.test(text3) && /\b(polo|shirt|knit)\b/.test(text3)) return "towelling resort texture underdresses a refined evening";
  if (slot === "outerwear" && /\b(flight|racer|racing|waxed|technical|track)\b/.test(text3) && /\b(jacket|cotton|nylon|zip|pocket)\b/.test(text3)) return "sport/utility construction underdresses the refined date";
  if (slot === "outerwear" && /\b(nylon|contrasting bands|contrast panels)\b/.test(text3) && /\b(zip|zipped|jacket|elastic)\b/.test(text3)) return "technical or athletic outerwear conflicts with refined evening garments";
  if (slot === "outerwear" && /\bjacket\b/.test(text3) && !/\b(blazer|tailored|single[ -]breasted|double[ -]breasted|suit|wool|cashmere|unstructured|tweed|herringbone|linen)\b/.test(text3)) return "jacket lacks evidence of refined tailoring for this date";
  if (slot === "top" && /\bdenim\b/.test(text3) && /\b(shirts?|western)\b/.test(text3)) return "denim/Western shirt underdresses a refined date";
  if (slot === "outerwear" && /\b(biker|motorcycle|moto|trucker)\b/.test(text3) && /\b(leather|suede|shearling|jacket|asymmetric|hardware)\b/.test(text3)) return "riding/workwear jacket dominates the refined date silhouette";
  if (slot === "outerwear" && /\b(deck|field|military|utility)\b/.test(text3) && /\b(jacket|cargo|pockets|epaulettes)\b/.test(text3)) return "utility outerwear is too casual for this refined date";
  if (slot === "shoes" && /\b(rugged|hiking|combat|workboot)\b/.test(text3) && /\b(lug|anti-slip|reinforced|rugged|workboot)\b/.test(text3)) return "rugged lug footwear conflicts with refined evening clothing";
  if (slot === "bottom" && /\b(drawstring|jogger|track|interlock|jersey)\b/.test(text3) && /\b(trousers?|pants?|elasticated|jersey)\b/.test(text3)) return "sporting trouser construction underdresses the date";
  return null;
}
function qualityColor(item) {
  const text3 = normalizeGarmentText(`${item.color || ""} ${item.name || ""}`);
  const groups = [["white", /\b(white|ivory|ecru|cream|offwhite)\b/], ["navy", /\b(navy|indigo|midnight)\b/], ["black", /\bblack\b/], ["grey", /\b(grey|gray|charcoal|silver)\b/], ["earth", /\b(brown|tan|camel|beige|taupe|sand|cognac|brandy|chocolate)\b/], ["green", /\b(olive|sage|khaki|forest|green)\b/], ["red", /\b(burgundy|maroon|red)\b/], ["blue", /\bblue\b/], ["yellow", /\b(yellow|mustard)\b/], ["orange", /\borange\b/], ["purple", /\b(purple|lilac)\b/], ["pink", /\bpink\b/]];
  const explicit = normalizeGarmentText(String(item.color || ""));
  return groups.find(([, pattern]) => pattern.test(explicit))?.[0] || groups.find(([, pattern]) => pattern.test(text3))?.[0] || "unknown";
}
function assessOutfitQuality(items, style, occasion, weather, lockedItemId, preferences) {
  const dimensions = { styleFit: 1, occasionFit: 1, colorHarmony: 1, materialHarmony: 1, formalityFit: 1, weatherFit: 1 };
  const reasons = [], evidence = [];
  const relevant = isOldMoneyDate(style, occasion);
  for (const item of items) {
    const locked = !!lockedItemId && String(item.id) === lockedItemId;
    const reason = locked ? null : refinedDateProductReason(item, style, occasion);
    if (locked && refinedDateProductReason(item, style, occasion)) evidence.push("owned anchor balanced by refined companions");
    if (reason) {
      reasons.push(reason);
      dimensions.styleFit = 0;
      dimensions.occasionFit = 0;
      dimensions.formalityFit = 0;
    }
    if (rejectForClimate(item, weather)) dimensions.weatherFit = 0;
  }
  const bySlot = (slot) => items.find((item) => macroOf(item) === slot);
  const pairs = [["outerwear", "top"], ["top", "bottom"], ["bottom", "shoes"], ["outerwear", "shoes"]];
  let conflictingPairs = 0;
  for (const [a, b] of pairs) {
    const first = bySlot(a), second = bySlot(b);
    if (!first || !second) continue;
    const x = qualityColor(first), y = qualityColor(second);
    const vivid = (item) => /\b(bright|neon|electric|vivid|fluorescent)\b/.test(blobOf(item));
    const complement = [["red", "green"], ["blue", "orange"], ["yellow", "purple"]].some((p) => p.includes(x) && p.includes(y) && x !== y);
    if (complement && vivid(first) && vivid(second)) {
      conflictingPairs++;
      evidence.push(`${a}/${b}: competing vivid complementary colors`);
    }
  }
  if (conflictingPairs) {
    dimensions.colorHarmony = 0;
    reasons.push("competing vivid complementary colors across outfit pairs");
  }
  const palette = outfitPaletteReason(items);
  if (palette) {
    dimensions.colorHarmony = 0;
    reasons.push(palette);
  }
  if (relevant) {
    const layer = bySlot("outerwear"), top = bySlot("top"), bottom = bySlot("bottom");
    const upper = [layer, top].filter(Boolean).map((item) => blobOf(item)).join(" ");
    const lower = bottom ? blobOf(bottom) : "";
    const layerEvidence = layer ? blobOf(layer) : "";
    const winterCoat = /\b(coat|trench|overcoat|peacoat)\b/.test(layerEvidence) && /\b(wool|cashmere|winter|insulated|double[ -]breasted)\b/.test(layerEvidence) && !/\b(lightweight|tropical|summer|unlined)\b/.test(layerEvidence);
    if ((winterCoat || /\b(shearling|alpaca|mohair|fleece|chunky|heavyweight|rugged|padded)\b/.test(upper)) && /\b(linen|seersucker|gauze)\b/.test(`${lower} ${top ? blobOf(top) : ""}`)) {
      dimensions.materialHarmony = 0;
      reasons.push("heavy rugged upper textures conflict with airy summer trousers");
    }
    const utility = items.filter((item) => /\b(western|military|utility|workwear|leisure)\b/.test(blobOf(item)));
    if (utility.length >= 2) {
      dimensions.styleFit = 0;
      reasons.push("multiple utility/leisure pieces dominate the refined date");
    }
    if (utility.length === 1) {
      dimensions.styleFit = 0.75;
      evidence.push("one casual detail balanced by refined companions");
    }
    const layerText = layer ? blobOf(layer) : "";
    if (/\b(leather|suede)\b/.test(layerText) && !reasons.length) {
      dimensions.styleFit = Math.min(dimensions.styleFit, 0.75);
      evidence.push("understated leather layer: smart evening rather than classic tailoring");
    }
  }
  const connected = outfitCoherence(items.map((item) => ({ ...item, id: item.id == null ? void 0 : String(item.id), name: item.name || void 0, type: item.type || void 0, description: item.description || void 0, macroCategory: item.macroCategory || void 0, color: item.color || void 0 })), stylingContext({ style, occasion, weather, anchorId: lockedItemId, preferences }));
  reasons.push(...connected.reasons);
  evidence.push(...connected.evidence);
  const score = Object.values(dimensions).reduce((sum, n) => sum + n, 0) / 6 * 0.5 + connected.score * 0.5;
  return { dimensions, score, reasons: [...new Set(reasons)], evidence };
}
function createQualityMemo(maxEntries = 2e3) {
  const cache = /* @__PURE__ */ new Map();
  const signature = (item) => {
    const blob = `${item?.id ?? ""}~${item?.macroCategory ?? ""}~${item?.color ?? ""}~${item?.material ?? ""}~${item?.pattern ?? ""}~${item?.name ?? ""}`;
    return blob.length > 160 ? blob.slice(0, 160) : blob;
  };
  const stable = (value) => {
    if (value === null || value === void 0) return "~";
    if (typeof value !== "object") return String(value);
    try {
      return JSON.stringify(value);
    } catch {
      return "~";
    }
  };
  return (items, style, occasion, weather, lockedItemId, preferences) => {
    const key = `${(items || []).map(signature).sort().join("|")}@${stable(style)}@${stable(occasion)}@${stable(weather)}@${stable(lockedItemId)}@${stable(preferences)}`;
    const hit = cache.get(key);
    if (hit) return hit;
    const verdict = assessOutfitQuality(items, style, occasion, weather, lockedItemId, preferences);
    if (cache.size >= maxEntries) {
      const oldest = cache.keys().next();
      if (!oldest.done) cache.delete(oldest.value);
    }
    cache.set(key, verdict);
    return verdict;
  };
}
function spreadQualityFamilies(items, familyOf) {
  const families = /* @__PURE__ */ new Map();
  for (const item of items) {
    const key = familyOf(item);
    const group = families.get(key) || [];
    group.push(item);
    families.set(key, group);
  }
  const result = [];
  for (let i = 0; result.length < items.length; i++) for (const group of families.values()) if (i < group.length) result.push(group[i]);
  return result;
}

// src/domain/outfitRotation.ts
function wasRecentlyShown(item, history) {
  const id = String(item?.id || "");
  return history.has(id) || history.has(id.replace(/^shop_/, "")) || productIdentityKeys({ ...item, imageUrl: item?.imageUrl || item?.image_url || item?.image }).some((key) => history.has(key));
}
function rotationPool(items, history, target, lockedIds = /* @__PURE__ */ new Set()) {
  const recycledSlots = [];
  const kept = [];
  for (const slot of ["outerwear", "top", "bottom", "shoes"]) {
    const pool = items.filter((item) => diversitySlotOf(item) === slot);
    const fresh = pool.filter((item) => !wasRecentlyShown(item, history) || lockedIds.has(String(item.id)));
    const seen = pool.filter((item) => wasRecentlyShown(item, history) && !lockedIds.has(String(item.id)));
    kept.push(...fresh);
    if (countDistinctProducts(fresh) < target && seen.length) {
      recycledSlots.push(slot);
      kept.push(...seen);
    }
  }
  return { items: kept, recycledSlots };
}
function freshProductCount(items, history, anchorId) {
  return countDistinctProducts(items.filter((item) => String(item.id) !== anchorId && !wasRecentlyShown(item, history)));
}
function createRotationPriority(history, previous, decks, anchorId) {
  const deckSets = decks.map((deck) => new Set(deck.productKeys));
  const cache = /* @__PURE__ */ new Map();
  return (items) => {
    let total = 0;
    const used = /* @__PURE__ */ new Set();
    for (const item of items) {
      const id = String(item.id);
      if (id === anchorId) continue;
      let entry = cache.get(id);
      if (!entry) {
        const keys = productIdentityKeys(item);
        let last = -1;
        deckSets.forEach((set, index) => {
          if (wasRecentlyShown(item, set)) last = index;
        });
        entry = { keys, score: (!wasRecentlyShown(item, history) ? 1e5 : 0) + (!wasRecentlyShown(item, previous) ? 1e3 : 0) + (last < 0 ? 0 : decks.length - last) };
        cache.set(id, entry);
      }
      if (entry.keys.some((key) => used.has(key))) continue;
      entry.keys.forEach((key) => used.add(key));
      total += entry.score;
    }
    return total;
  };
}
function compareRotationOrder(a, b) {
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    const delta = (a[i] || 0) - (b[i] || 0);
    if (delta) return delta;
  }
  return 0;
}
function createShoeRotationPriority(history, previous, decks, anchorId, shoeHistory) {
  const sets = decks.map((deck) => new Set(deck.productKeys));
  const shoeSets = shoeHistory?.decks.map((deck) => new Set(deck.productKeys)) || sets;
  const legacy = createRotationPriority(history, previous, decks, anchorId);
  const cache = /* @__PURE__ */ new Map();
  return (items) => {
    let shoeOverlap = 0, otherOverlap = 0;
    const ages = Array(shoeSets.length + 1).fill(0), counts = Array(shoeSets.length + 1).fill(0);
    const used = /* @__PURE__ */ new Set(), clothing = [];
    for (const item of items) {
      const id = String(item.id);
      if (id === anchorId) continue;
      const isShoe = diversitySlotOf(item) === "shoes";
      const cacheKey = `${id}|${isShoe ? "shoe" : "clothing"}`;
      let entry = cache.get(cacheKey);
      if (!entry) {
        let last = -1, count = 0;
        (isShoe ? shoeSets : sets).forEach((set, i) => {
          if (wasRecentlyShown(item, set)) {
            last = i;
            count++;
          }
        });
        entry = { keys: productIdentityKeys(item), previous: wasRecentlyShown(item, isShoe ? shoeHistory?.previous || previous : previous), last, count };
        cache.set(cacheKey, entry);
      }
      if (entry.keys.some((key) => used.has(key))) continue;
      entry.keys.forEach((key) => used.add(key));
      if (diversitySlotOf(item) === "shoes") {
        shoeOverlap += Number(entry.previous);
        ages[entry.last + 1]++;
        counts[entry.count]++;
      } else {
        otherOverlap += Number(entry.previous);
        clothing.push(item);
      }
    }
    return [-shoeOverlap, -otherOverlap, ...ages, ...counts, legacy(clothing)];
  };
}
function assignRotatingShoes(looks, shoes, accepts, priority, quality, anchorId) {
  const anchor = shoes.find((shoe) => String(shoe.id) === anchorId);
  const choices = looks.map((look) => (anchor ? [anchor] : shoes).map((shoe) => {
    const items = look.items.map((item) => diversitySlotOf(item) === "shoes" ? shoe : item);
    return accepts(items) ? { look: { ...look, items }, keys: productIdentityKeys(shoe), score: quality(items) } : null;
  }).filter((choice) => choice !== null));
  const requireUnique = !anchor && countDistinctProducts(shoes) >= looks.length;
  let best = [], bestOrder = [], bestQuality = -Infinity, minimum = null;
  const MAX_SHOE_ASSIGNMENT_COMBINATIONS = 2e4;
  const totalCombinations = choices.reduce((product, list2) => product * Math.max(1, list2.length), 1);
  const greedyAssign = () => {
    const order = choices.map((list2, index) => ({ list: list2, index })).sort((a, b) => a.list.length - b.list.length || a.index - b.index);
    const picked = new Array(choices.length).fill(null);
    const used = /* @__PURE__ */ new Set();
    let pickedScore = 0;
    for (const { list: list2, index } of order) {
      let winner = null;
      for (const choice of list2) {
        if (requireUnique && choice.keys.some((key) => used.has(key))) continue;
        if (!winner) {
          winner = choice;
          continue;
        }
        const byOrder = compareRotationOrder(
          priority([...picked.filter(Boolean).map((look) => look.items).flat(), ...choice.look.items]),
          priority([...picked.filter(Boolean).map((look) => look.items).flat(), ...winner.look.items])
        );
        if (byOrder > 0 || byOrder === 0 && choice.score > winner.score) winner = choice;
      }
      if (!winner) {
        best = [];
        return;
      }
      picked[index] = winner.look;
      pickedScore += winner.score;
      winner.keys.forEach((key) => used.add(key));
    }
    best = picked;
    if (best.length) {
      const finalOrder = priority(best.flatMap((look) => look.items));
      minimum = -finalOrder[0];
      bestOrder = finalOrder;
      bestQuality = pickedScore;
    }
  };
  const walk = (index, selected, used, score) => {
    if (index === choices.length) {
      const order = priority(selected.flatMap((look) => look.items));
      minimum = Math.min(minimum ?? Infinity, -order[0]);
      const comparison = best.length ? compareRotationOrder(order, bestOrder) : 1;
      if (comparison > 0 || comparison === 0 && score > bestQuality) {
        best = [...selected];
        bestOrder = order;
        bestQuality = score;
      }
      return;
    }
    for (const choice of choices[index]) {
      if (requireUnique && choice.keys.some((key) => used.has(key))) continue;
      walk(index + 1, [...selected, choice.look], /* @__PURE__ */ new Set([...used, ...choice.keys]), score + choice.score);
    }
  };
  if (looks.length) {
    if (totalCombinations > MAX_SHOE_ASSIGNMENT_COMBINATIONS) greedyAssign();
    else walk(0, [], /* @__PURE__ */ new Set(), 0);
  }
  return { deck: best, feasibleMinimumOverlap: minimum };
}

// backend/supabase/functions/generate-outfits/index.ts
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient as createClient2 } from "https://esm.sh/@supabase/supabase-js@2";
var corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type"
};
var GENERATION_VERSION = "stylist-runtime-repair-v3-cpu-bounded";
var catalogPages = /* @__PURE__ */ new Map();
var CATALOG_PAGE_TTL = 5 * 60 * 1e3;
var MAX_LEGACY_PAGES_PER_SLOT = 6;
function parseBudgetFromText(text3, fx) {
  const found = budgetCandidatesFromText(text3);
  if (!found.length || !fx && new Set(found.map((b) => b.currency)).size > 1) return null;
  return found.reduce((best, cur) => convertShopAmountEdge(cur.amount, cur.currency, best.currency, fx) < best.amount ? cur : best);
}
function isShopPricedItem(item) {
  if (!item) return false;
  if (item.isWardrobe === true && !item.isShopItem) return false;
  const id = String(item.id || "");
  return Boolean(item.isShopItem || item.isWardrobe === false || /^(shop_|md-|zara-|classic-m-)/i.test(id));
}
function convertShopAmountEdge(amount2, from, to, fx) {
  const source = String(from || "").trim().toUpperCase(), target = String(to).trim().toUpperCase();
  if (source === target) return Number.isFinite(amount2) && amount2 >= 0 ? amount2 : Infinity;
  return fx ? referenceAmount(amount2, source, target, fx.snapshot, fx.checkedAt) : Infinity;
}
function shopPriceOf(item, targetCurrency, fx) {
  if (!isShopPricedItem(item)) return 0;
  const n = Number(item?.price);
  return Number.isFinite(n) && n > 0 ? targetCurrency ? convertShopAmountEdge(n, item.currency, targetCurrency, fx) : n : targetCurrency ? Number.POSITIVE_INFINITY : 0;
}
function outfitShopSpend(items, itemMap, targetCurrency, fx) {
  const currency2 = targetCurrency || items.map((it) => (itemMap?.get(String(it?.id)) || it)?.currency).find(Boolean);
  return (items || []).reduce((sum, it) => {
    const src = itemMap?.get(String(it?.id)) || it;
    return sum + shopPriceOf({
      ...src,
      ...it,
      price: src?.price,
      currency: src?.currency,
      isShopItem: src?.isShopItem,
      isWardrobe: src?.isWardrobe,
      id: src?.id
    }, currency2, fx);
  }, 0);
}
function budgetInCurrencyEdge(budget, targetCurrency) {
  if (!budget) return null;
  const to = String(targetCurrency || "").trim().toUpperCase();
  const from = String(budget.currency || "USD").trim().toUpperCase();
  if (!to || to === from) return { ...budget, currency: from };
  const converted = convertShopAmountEdge(budget.amount, from, to, budget.fx);
  return Number.isFinite(converted) ? { amount: converted, currency: to } : null;
}
function fitOutfitToBudget(parts, pool, budget) {
  if (!budget || !Array.isArray(parts) || parts.length === 0) return parts;
  const cap = budget.amount;
  const slotOf = (p) => String(p?.macroCategory || "").toLowerCase();
  const cheaper = (slot, exclude, maxPrice) => {
    return pool.filter((p) => isShopPricedItem(p) && slotOf(p) === slot && !exclude.has(String(p.id))).filter((p) => {
      const pr = shopPriceOf(p, budget.currency, budget.fx);
      return pr > 0 && pr <= maxPrice;
    }).sort((a, b) => shopPriceOf(a, budget.currency, budget.fx) - shopPriceOf(b, budget.currency, budget.fx))[0];
  };
  const next = parts.map((p) => ({ ...p }));
  for (let guard = 0; guard < 8 && !fitsBudgetAmount(outfitShopSpend(next, void 0, budget.currency, budget.fx), cap); guard++) {
    let worstIdx = -1;
    let worstPrice = -1;
    next.forEach((p, i) => {
      const pr = shopPriceOf(p, budget.currency, budget.fx);
      if (pr > worstPrice) {
        worstPrice = pr;
        worstIdx = i;
      }
    });
    if (worstIdx < 0) break;
    const used = new Set(next.map((p) => String(p.id)));
    const room = cap - (outfitShopSpend(next, void 0, budget.currency, budget.fx) - shopPriceOf(next[worstIdx], budget.currency, budget.fx));
    const repl = cheaper(slotOf(next[worstIdx]), used, room);
    if (repl) {
      next[worstIdx] = { ...repl, recommendation: next[worstIdx].recommendation || "Suggested from shop" };
    } else {
      break;
    }
  }
  return next;
}
function isShopOnlyDeck(style, occasion) {
  const s = (style || "").toLowerCase().replace(/[\s-]+/g, "_");
  if (s === "business_casual") return true;
  const o = (occasion || "").toLowerCase();
  if (o.includes("team") && o.includes("collab")) return true;
  if (o.trim() === "team collaboration") return true;
  return false;
}
function isShopRow(item) {
  if (!item) return false;
  return Boolean(item.isShopItem) || String(item.id || "").startsWith("shop_");
}
function macroCategory(category, type, name) {
  const garmentName = String(name || "").toLowerCase();
  if (/кроссовк|кеды|туфл|ботинк|лофер|сапог|мокасин|сандал/.test(garmentName)) return "shoes";
  if (/жилет|куртк|пальто|пиджак|жакет|кардиган|свитер|худи|толстовк/.test(garmentName)) return "outerwear";
  if (/джинс|брюк|шорт|юбк|леггинс/.test(garmentName)) return "bottom";
  if (/футболк|рубашк|блуз|майк|платье/.test(garmentName)) return "top";
  if (/\b(pants?|trousers?|jeans|chinos?|slacks?|joggers?)\b/.test(garmentName) && !/\b(shirts?|jackets?|coats?|blazers?|cardigans?)\b/.test(garmentName)) return "bottom";
  const t = normalizeGarmentText(`${category} ${type} ${name || ""}`);
  let result = "other";
  if (t.match(/\b(blazer|overcoat|topcoat|peacoat|trench(?:es)?|parka|puffer|windbreaker|bomber)s?\b/) || t.match(/\b(coat|jacket|cardigan|sweater|hoodie|vest|pullover|fleece)s?\b/) || t.match(/jacket|coat|blazer|hoodie|cardigan|sweater|pullover|vest|puffer|parka|trench|outerwear/)) {
    result = "outerwear";
  } else if (t.match(/\b(pants?|trousers?|jeans|shorts?|skirts?|chinos?|slacks?|joggers?|sweatpants?)\b/) || t.match(/pant|trouser|jeans|bottom|shorts|skirt|lower[_\s-]?body/)) {
    result = "bottom";
  } else if (t.match(/\b(shoes?|sneakers?|boots?|loafers?|sandals?|heels?|trainers?|derbys?|derbies|mules?)\b/) || t.match(/\boxfords?\b/) && !t.match(/\bshirts?\b/) || t.match(/\boxfords?\s*(shoes?|flats?|lace|brogues?|derbys?|derbies)\b/) || t.match(/shoe|sneaker|boot|loafer|sandal|heel|footwear|trainer/)) {
    result = "shoes";
  } else if (t.match(/\b(t-shirt|tshirt|tee|polo|blouse|shirt|dress(?:es)?)s?\b/) || t.match(/shirt|t-shirt|tshirt|tee|blouse|polo|tops|top|upper[_\s-]?body/) || t.match(/dress/)) {
    result = "top";
  }
  return result;
}
function isDressItem(item) {
  return isDressGarment(item);
}
function needsLayering2(style, weather, prompt) {
  return needsLayering(style, weather, prompt);
}
function isFormalLayerItem(item) {
  const blob = `${item?.type || ""} ${item?.name || ""} ${item?.sub_category || ""} ${item?.subCategory || ""} ${item?.description || ""}`.toLowerCase();
  const macro = String(item?.macroCategory || "").toLowerCase();
  const isOuter = macro === "outerwear" || /jacket|coat|blazer|vest|outerwear/.test(blob);
  if (!isOuter) return false;
  return /\b(blazer|suit\s*jacket|sport\s*coat|sports\s*coat|overcoat|top\s*coat|topcoat|trench|peacoat|pea\s*coat|tuxedo)\b/.test(blob);
}
function isShortsItem(item) {
  const blob = `${item?.type || ""} ${item?.name || ""} ${item?.sub_category || ""} ${item?.subCategory || ""} ${item?.description || ""}`.toLowerCase().replace(/\bshort[\s-]?sleeve(?:s|d)?\b/g, "");
  const macro = String(item?.macroCategory || "").toLowerCase();
  const isBottom = macro === "bottom" || /pant|trouser|jeans|bottom|shorts?|skirt|trunks?|swim|lower[_\s-]?body/.test(blob);
  if (!isBottom) return false;
  return /\b(shorts?|bermudas?|trunks?|swim\w*|swimwear|board\s*shorts?|bathing\s*suit)\b/.test(blob);
}
function isSwimBottomItem(item) {
  const blob = `${item?.type || ""} ${item?.name || ""} ${item?.sub_category || ""} ${item?.subCategory || ""} ${item?.description || ""}`.toLowerCase();
  const macro = String(item?.macroCategory || "").toLowerCase();
  if (macro !== "bottom" && !/lower[_\s-]?body/.test(blob)) return false;
  return /\b(swim\w*|swimwear|trunks?|board\s*shorts?|bathing\s*suit)\b/.test(blob);
}
function isDriverShoeItem(item) {
  const blob = `${item?.type || ""} ${item?.name || ""} ${item?.sub_category || ""} ${item?.subCategory || ""} ${item?.description || ""}`.toLowerCase();
  const macro = String(item?.macroCategory || "").toLowerCase();
  if (macro !== "shoes" && !/shoe|sneaker|footwear|upper[_\s-]?body/.test(blob)) return false;
  return isDrivingFootwear(item) || /\bmules?\b/.test(blob);
}
function isWorkishOccasion(occasion) {
  return /work|business|office|meeting|interview|wedding|funeral|formal|conference|corporate/.test((occasion || "").toLowerCase());
}
function inferAttrsEdge(item) {
  const attrs = inferItemAttributes(item);
  return { styleTags: attrs.styleTags, materials: attrs.materials.join(", "), patterns: attrs.patterns.join(", "), formality: attrs.formality };
}
function rankItemsForStyleEdge(items, style, opts = {}) {
  return rankItemsForStyle(items, normalizeStyleId(style), { minKeep: 12, dropThreshold: -2, ...opts });
}
var SIM_COLOR_WORDS = new Set("black white offwhite ecru cream beige stone sand taupe camel khaki olive green sage mint blue navy indigo brown chocolate grey gray silver charcoal red burgundy maroon pink purple lilac yellow mustard orange rust".split(" "));
var SIM_LOW_WORDS = new Set("basic classic essential premium soft touch washed faded heavyweight mediumweight lightweight regularfit relaxedfit slimfit loosefit textured structured comfort interlock stretch cotton viscose polyester polyamide elastane lyocell modal jersey fit detail details edition men man zara suit shoe".split(" "));
function simKeyEdge(name, slot) {
  const toks = [];
  const seen = /* @__PURE__ */ new Set();
  const words2 = String(name || "").toLowerCase().replace(/https?:\/\//g, "").replace(/[^a-z0-9]+/g, " ").split(" ");
  for (let raw of words2) {
    if (!raw || /^\d+$/.test(raw) || raw.length < 2) continue;
    if (raw.endsWith("ies") && raw.length > 4) raw = raw.slice(0, -3) + "y";
    else if (raw.endsWith("s") && raw.length > 3 && !raw.endsWith("ss")) raw = raw.slice(0, -1);
    if (raw === "tshirt" || raw === "tee" || raw === "tees") raw = "tee";
    if (raw === "trainer" || raw === "trainers") raw = "sneaker";
    if (raw === "sweater" || raw === "jumper") raw = "knit";
    if (SIM_COLOR_WORDS.has(raw) || SIM_LOW_WORDS.has(raw) || seen.has(raw)) continue;
    seen.add(raw);
    toks.push(raw);
  }
  return `${slot}:${toks.join(" ") || slot}`;
}
function spreadBySimEdge(items, keyFn) {
  if (items.length <= 2) return items;
  const remaining = items.map((item) => ({ item, key: keyFn(item) }));
  const ordered = [];
  const recent = [];
  while (remaining.length > 0) {
    let idx = remaining.findIndex((c) => !recent.includes(c.key));
    if (idx === -1) {
      const last = recent[recent.length - 1];
      idx = remaining.findIndex((c) => c.key !== last);
    }
    if (idx === -1) idx = 0;
    const [next] = remaining.splice(idx, 1);
    ordered.push(next);
    recent.push(next.key);
    if (recent.length > 2) recent.shift();
  }
  return ordered.map((o) => o.item);
}
function collapseBySimEdge(items, keyFn) {
  const seen = /* @__PURE__ */ new Set();
  const out = [];
  for (const item of items) {
    const key = keyFn(item);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(item);
  }
  return out;
}
function retainBudgetFloorCandidates(source, ranked, cost, eligible, anchor, perSlotChoices = 1) {
  const cheapest = /* @__PURE__ */ new Map();
  for (const item of source) {
    if (!eligible(item)) continue;
    const slot = isDressGarment(item) ? "dress" : String(item.macroCategory || "").toLowerCase();
    if (!["top", "bottom", "shoes", "outerwear", "dress"].includes(slot)) continue;
    const value = cost(item);
    if (!Number.isFinite(value) || value < 0) continue;
    const prior = cheapest.get(slot);
    if (!prior || value < prior.cost) cheapest.set(slot, { item, cost: value });
  }
  const representatives = Array.from(cheapest.values(), (row) => row.item);
  const representedIds = new Set(representatives.map((item) => String(item.id)));
  const slotCounts = new Map(Array.from(cheapest.keys(), (slot) => [slot, 1]));
  for (const item of [...ranked, ...source]) {
    if (!eligible(item) || representedIds.has(String(item.id))) continue;
    const slot = isDressGarment(item) ? "dress" : String(item.macroCategory || "").toLowerCase();
    if (!cheapest.has(slot) || (slotCounts.get(slot) || 0) >= perSlotChoices) continue;
    representedIds.add(String(item.id));
    slotCounts.set(slot, (slotCounts.get(slot) || 0) + 1);
    representatives.push(item);
  }
  const combined = [...anchor ? [anchor] : [], ...representatives, ...ranked];
  const ids = /* @__PURE__ */ new Set();
  return collapseBySimEdge(combined.filter((item) => {
    const id = String(item.id);
    if (ids.has(id)) return false;
    ids.add(id);
    return true;
  }), (item) => canonicalProductKey(item)).slice(0, 40);
}
function deckHashEdge(style, occasion) {
  const key = `${style}::${occasion}`.toLowerCase();
  let hash = 0;
  for (let i = 0; i < key.length; i++) hash = Math.imul(31, hash) + key.charCodeAt(i) | 0;
  return Math.abs(hash);
}
function deckSeedEdge(style, occasion) {
  return deckHashEdge(style, occasion) % 32;
}
function deckReverseEdge(style, occasion) {
  return (deckHashEdge(style, occasion) >> 5 & 1) === 1;
}
var STYLING_TIPS_LAYERED = [
  "Layer the base top under the outerwear for depth",
  "Keep the palette tonal for a refined finish",
  "Match your socks to your trousers, not your shoes",
  "Leave the bottom jacket button undone",
  "Show 1-2cm of shirt cuff beyond the jacket sleeve",
  "Texture elevates: linen, flannel or suede over flat cotton"
];
var STYLING_TIPS_FLAT = [
  "Add accessories to personalize",
  "Roll sleeves or cuffs for a lived-in feel",
  "Match your socks to your trousers, not your shoes",
  "Keep it to three colors \u2014 white doesn't count",
  "Dark top + light bottom (or the reverse) reads most masculine",
  "Trousers sit at mid-rise; low-rise breaks the proportions"
];
function pickStylingTips(layered) {
  const pool = layered ? STYLING_TIPS_LAYERED : STYLING_TIPS_FLAT;
  const a = Math.floor(Math.random() * pool.length);
  let b = Math.floor(Math.random() * pool.length);
  if (b === a) b = (b + 1) % pool.length;
  return [pool[a], pool[b]];
}
var STYLE_KNOWLEDGE = `
UNIVERSAL STYLE PRINCIPLES (apply to every outfit regardless of style):
1. FIT IS KING: Medium-tapered silhouettes always beat skinny/oversized for classic looks. No pulling buttons, no excess fabric. Shoulders must align.
2. HIGH/MID-RISE TROUSERS: Always prefer mid-to-high rise. Low-rise breaks body proportions and looks sloppy. Bottom hem width ~19-20cm for classic; slightly narrower for smart casual.
3. NEVER MIX STYLES: Do not pair sportswear with tailored pieces in one outfit. A suit jacket over a sports tee or sneakers with a blazer violates formality coherence.
4. BLACK IS NOT UNIVERSAL: Black only pairs with white. For classic/smart-casual wardrobes, replace black with navy, charcoal, dark brown, or burgundy \u2014 all far more versatile.
5. GOLDEN 8 COLORS: white, blue, light blue, brown, green, cream/milk, burgundy, grey. These 8 intermix freely: blue+brown, blue+green, green+brown always work.
6. TEXTURE ELEVATES: Same color in cheap cotton looks like underwear; in textured linen, cashmere, or suede it reads as luxury. Favor texture over trend.
7. SEASONAL FABRICS: Linen/tropical wool/silk-cotton for summer; flannel/tweed/heavy wool for winter. Accessories follow seasons too (linen ties in summer, wool in winter).
8. KNIT OVER WOVEN: For polos and t-shirts, knit (fine-gauge) reads as elegant and versatile; piqu\xE9 reads as sporty; basic woven cotton reads as cheap.
9. COLLAR SIZE MATTERS: Large shirt collars frame the face and tuck under jacket lapels cleanly. Small collars, mandarin/stand collars cheapen any look.
10. LAPEL WIDTH: Minimum 8.5cm. Narrow lapels cheapen the look. Lapel should cover ~half the distance from neck to shoulder seam.
11. FOOTWEAR RULES: Loafers (penny/bit/tassel), oxfords, and derbies for smart and classic \u2014 all pair with tailoring. Driving moccasins sit a full tier BELOW loafers (soft unstructured sole, tie laces): Riviera-casual with chinos or linen, NEVER with suits or at work. Clean minimal sneakers only for casual; never athletic trainers with tailored clothing. Suede is more versatile than leather for smart-casual.
12. ACCESSORIES COMPLETE: Pocket squares, ties, seasonal scarves, watches elevate any outfit from ordinary to polished.
13. FORMALITY COHERENCE: Every item in one outfit should sit within 1-2 formality tiers of each other. A formal blazer + gym shorts = hard clash.
14. QUALITY OVER QUANTITY: One well-made piece in a good fabric outperforms three cheap ones. Natural fibers (wool, linen, cotton, silk) always look more expensive than synthetics.
15. AVOID: puffer jackets with suits, square-toe shoes, backpacks with tailored clothing, sports watches with formal outfits, graphic tees in any non-casual look.
16. SOCKS MATCH TROUSERS, NOT SHOES: Grey, navy, and burgundy socks are the foundation. Black socks are disallowed \u2014 they show wear faster and pair with nothing.
17. CONTRAST & COLOR BALANCE: Dark top + light bottom (or vice versa) creates more masculine, harmonious outfits. Limit to 3 colors per outfit; white doesn't count and can be a 4th. Monochromatic looks need different textures/shades to avoid looking flat.
18. JACKET FIT DETAILS: Jacket must cover the buttocks. 1-2cm of shirt cuff must show beyond the jacket sleeve. Never button the bottom vest button. Tie width should be 7.5-9.5cm.
19. CITY BANS: No shorts, sandals/mules, mesh/see-through tees, or mandarin/stand collars in urban settings. No popped polo collars. Always button polo buttons (all or leave top one open). No exotic leather shoes (crocodile, python) \u2014 quality calf or suede only.
20. PATTERNED SPORT JACKETS: Glen check, Prince of Wales, Gun Club are the most versatile patterns \u2014 they contain multiple colors for easy coordination. Avoid smooth solid-color sport jackets (look cheap); choose textured or patterned fabrics. Mixed fabric compositions (wool-linen-silk blends) are more textured and practical than pure fibers.`;
var WORKOUT_STYLE_KNOWLEDGE = `WORKOUT REQUIREMENTS:
- Use verified athletic or performance clothing in every required slot, with activity-appropriate athletic shoes.
- Match the user's fit, color and pattern preferences without substituting dress clothing or tailored footwear.
- Performance synthetics and breathable fabrics are appropriate; prioritize movement and comfort.
- Keep all weather, required-slot and total-budget constraints. Never omit shoes or a required layer.`;
var STYLE_CONTEXT = {
  sport: {
    vibe: "Functional workout clothing with coordinated, understated colors.",
    must: ["Verified athletic top, training bottom and activity-appropriate athletic shoes. Activity requirements take priority over aesthetic tailoring."],
    reject: ["NEVER include: dress shirts, blazers, loafers, suede shoes"],
    palette: ["navy", "grey", "black", "white"],
    fabrics: ["performance fabric", "breathable cotton", "stretch jersey"]
  },
  old_money: {
    vibe: `Old Money / Quiet Luxury \u2014 the wardrobe of someone who summers in the Hamptons, sails in Capri, and owns a Brunello Cucinelli cardigan. Think Ralph Lauren Purple Label, Loro Piana, Brooks Brothers. Nothing flashy, everything expensive. The outfit should feel understated, tonal, and timeless. Medium-tapered fit, never skinny. High-rise trousers. Texture over trend.`,
    must: [
      "Use only tonal / monochromatic / analogous palettes (e.g. camel + cream + white; navy + cream + brown; blue + green + brown).",
      "Prefer tailored silhouettes with medium taper: blazers, cardigans, oxford shirts, knit polos, chinos, pleated trousers, slacks. NEVER skinny/super-slim fit.",
      "Trousers must be mid-to-high rise. Low-rise is disallowed for this style.",
      "Footwear must be loafers (penny, bit, tassel), oxfords, derbies, boat shoes, or minimal clean white/cream leather sneakers.",
      "Prefer knit polos and knit tees over piqu\xE9 or basic cotton \u2014 they read as more elegant and versatile.",
      "Shirts must have large collars that tuck cleanly under jacket lapels. Small collars or mandarin collars are disallowed.",
      "Blazer lapels must be \u22658.5cm wide. Narrow lapels cheapen the look.",
      "Favor textured fabrics: flannel, tweed, herringbone, cashmere, merino, linen blends. Smooth flat fabrics look cheaper.",
      "Every piece should look like it could be from Ralph Lauren, Brunello Cucinelli, Loro Piana, Massimo Dutti, or Brooks Brothers.",
      "Pocket square is mandatory with any blazer or suit outfit \u2014 its absence is a major style error."
    ],
    reject: [
      "NEVER include: graphic, logo, print, printed, patterned, floral, paisley, camo, tie-dye, pattern, patterns, souvenir jackets, pajama, pyjama, sleepwear, loungewear, t-shirt.",
      "NEVER include: hoodie, hoodies, hooded, windbreaker, windbreakers, puffer, puffers, zip-up, zip-ups, bomber, bombers, track jacket, track jackets, graphic sweatshirts.",
      "NEVER include: cargo shorts, cargo pants, athletic shorts, basketball shorts, track pants, sweatpants, joggers, jogging trousers, shorts/bermudas, drawstring pants, elastic-waist trousers.",
      "NEVER include: washed-effect jackets, bell-bottom or splatter pants, fashion denim, twisted-seam jeans, piqu\xE9 polos for work.",
      "NEVER include: driving moccasins, loafers with tie laces, chunky sneakers, high-tops, platform shoes, athletic trainers, running shoes, basketball shoes, square-toe shoes.",
      "NEVER include: black as a primary color \u2014 replace with navy, charcoal, or dark brown. Black only for formal eveningwear.",
      "NEVER include: exotic leather shoes (crocodile, python, stingray) \u2014 they look vulgar. Quality calf leather or suede only.",
      "NEVER pair: formal outerwear with shorts, backpacks with tailored clothing, sports watches with formal outfits."
    ],
    palette: ["navy", "cream", "ivory", "beige", "camel", "chocolate", "forest green", "burgundy", "white", "charcoal", "midnight blue", "tan", "olive"],
    fabrics: ["cashmere", "wool", "merino", "linen", "silk", "cotton", "poplin", "tweed", "flannel", "herringbone", "suede", "tropical wool"]
  },
  semi_classic: {
    vibe: `Semi-Classic \u2014 refined everyday elegance that bridges the gap between casual and formal. Think Massimo Dutti, Arket, COS. Tailored touches with relaxed comfort \u2014 structured but not stiff, polished but not overdressed. Cardigans over tees, chinos with loafers, knit polos with slacks.`,
    must: [
      "Blazers and structured cardigans are key layering pieces \u2014 more relaxed than formal but more polished than hoodies.",
      "Knit polos, fine-gauge tees, and turtlenecks as base layers. Piqu\xE9 polos and basic cotton tees are too sporty.",
      "Chinos, slacks, and well-cut jeans. Mid-rise, medium taper. No skinny, no baggy.",
      "Loafers (suede or leather), desert boots, clean minimal sneakers, or Chelsea boots.",
      "Stick to the golden 8 colors: white, blue, light blue, brown, green, cream, burgundy, grey \u2014 they all intermix.",
      "Favor textured fabrics: merino, cotton blends, linen, poplin, suede. Texture elevates simple outfits.",
      "Lightweight cardigans (cotton/merino) are the signature semi-classic layer \u2014 more relaxed than a blazer but more polished than a hoodie. Can be tied around the neck as a style accent when not worn.",
      "Socks must match trousers, not shoes. Grey, navy, and burgundy socks are the foundation."
    ],
    reject: [
      "Avoid: hoodie, hoodies, hooded, graphic, logo, print, printed, cargo, sweatpants, joggers, pajama, pyjama, sleepwear, t-shirt.",
      "Avoid: chunky sneakers, athletic trainers, basketball shoes.",
      "Avoid: neon, rhinestone, sequin, metallic, or tie-dye pieces.",
      "Avoid: oversized/baggy fits, ripped/distressed denim.",
      "Avoid: low-rise trousers \u2014 they break the refined proportions."
    ],
    palette: ["navy", "cream", "beige", "camel", "olive", "tan", "burgundy", "white", "charcoal", "brown", "light blue"],
    fabrics: ["merino", "cotton", "linen", "poplin", "suede", "knit", "denim", "wool", "tropical wool"]
  },
  minimalist: {
    vibe: `Minimalist \u2014 quiet, clean, intentional. Think COS, Uniqlo U, Acne Studios, The Row, Jil Sander. Every piece should feel essential. Texture and cut do the talking; color stays restrained.`,
    must: [
      "Stick to a tight palette: black, white, grey, beige, navy, stone. Monochromatic outfits are ideal.",
      "Silhouettes must be simple and clean; no busy patterns. Medium taper, never skinny.",
      "Prefer structured, tailored fits over trendy oversized looks.",
      "Use texture to add depth within monochromatic looks: ribbed knits, brushed wool, matte leather, raw linen.",
      "Knit polos and fine-gauge tees over piqu\xE9 or basic cotton for a more polished minimal look."
    ],
    reject: [
      "NEVER include: graphic prints, logos, tie-dye, floral prints, neon colors, rhinestones, sequins, pleated trousers, pajama, pyjama, sleepwear.",
      "Avoid: color-blocking, chunky sneakers, puffers with branding."
    ],
    palette: ["black", "white", "grey", "charcoal", "beige", "navy", "stone", "cream", "sand"],
    fabrics: ["merino", "wool", "cotton", "linen", "cashmere", "poplin", "suede"]
  },
  business_casual: {
    vibe: `Modern Professional \u2014 sharp tailoring that still feels comfortable. Hugo Boss meets Everlane meets Theory. Smart Casual with a corporate edge. Blazers with texture, knit polos, well-cut trousers.`,
    must: [
      "Blazers (textured: flannel, tweed, herringbone) paired with chinos or tailored trousers. Oxford or poplin button-downs with large collars.",
      "Knit polos and fine-gauge tees are excellent under blazers \u2014 more elegant than piqu\xE9 or basic cotton.",
      "Polished shoes: leather loafers, oxfords, derbies, or minimal clean white leather sneakers. Dark brown shoes are as formal as black but more versatile.",
      "Trousers must be mid-to-high rise. Low-rise is disallowed.",
      "Blazer lapels \u22658.5cm. Narrow lapels cheapen the professional look.",
      "Palette: navy, charcoal, tan, white, light blue, cream, brown, burgundy.",
      "Favor tonal/analogous combinations: navy + cream + brown; charcoal + light blue + tan.",
      "Navy blazer with metal buttons is the most versatile starter jacket \u2014 pairs with nearly any trouser color.",
      "Pocket square (white or tonal) is expected with any blazer \u2014 its absence is a noticeable gap.",
      "Socks must match trousers, not shoes. Grey, navy, and burgundy are the only acceptable sock colors."
    ],
    reject: [
      "Avoid: hoodie, hoodies, hooded, windbreaker, windbreakers, puffer, puffers, zip-up, zip-ups, bomber, bombers, track jacket, track jackets, souvenir jackets.",
      "Avoid: graphic, logo, print, printed, patterned, floral, paisley, camo, tie-dye, pattern, patterns, neon, rhinestones, sequins, pajama, pyjama, sleepwear, loungewear, t-shirt.",
      "Avoid: cargo pants, cargo shorts, ripped jeans, sweatpants, joggers, shorts, drawstring pants, elastic-waist trousers.",
      "Avoid: black as primary color \u2014 use navy or charcoal instead.",
      "Avoid: chunky sneakers, platform shoes, athletic trainers, running shoes, basketball shoes, square-toe shoes, driving moccasins, mules.",
      "Avoid: skinny/super-slim fit, low-rise trousers, small shirt collars, narrow lapels."
    ],
    palette: ["navy", "charcoal", "tan", "white", "light blue", "grey", "cream", "brown", "burgundy", "beige"],
    fabrics: ["wool", "cotton", "poplin", "oxford cloth", "linen", "flannel", "tweed", "merino", "cashmere", "tropical wool", "glen check", "prince of wales"]
  },
  casual: {
    vibe: `Smart Casual \u2014 relaxed but intentional. Well-fitted basics, clean color combinations. The bridge between sport and classic. Knit tees, slacks, loafers or clean sneakers.`,
    must: [
      "Well-fitted knit tees or knit polos with chinos, slacks, or well-cut jeans. Medium taper, not skinny.",
      "Clean minimal sneakers, loafers (suede or leather), or desert boots.",
      "Trousers at mid-rise. Low-rise breaks proportions.",
      "Stick to the golden 8 colors: white, blue, light blue, brown, green, cream, burgundy, grey \u2014 they all intermix.",
      "Add texture: ribbed knits, brushed cotton, linen blends elevate simple outfits.",
      "Lightweight cardigans (cotton/silk) are excellent smart-casual layers \u2014 more relaxed than a blazer but more polished than a hoodie. Can be tied around the neck as a style accent.",
      "Socks should match trousers or another outfit item, not shoes."
    ],
    reject: [
      "Avoid: sweatpants, basketball shorts, graphic hoodies, hoodies, hooded layers, pajama, pyjama, sleepwear.",
      "Avoid: athletic sneakers with trousers, piqu\xE9 polos with dressy bottoms.",
      "Avoid: mesh/see-through tees, popped polo collars, unbuttoned polo collars."
    ],
    palette: ["navy", "white", "grey", "beige", "olive", "cream", "brown", "light blue", "burgundy", "tan"],
    fabrics: ["cotton", "linen", "denim", "wool", "merino", "knit", "suede", "canvas", "tropical wool"]
  },
  classic: {
    vibe: `Classic Menswear \u2014 timeless elegance rooted in traditional tailoring. Think Savile Row, Italian sartoria, Alexander from Strokanor. Suits, sport jackets, high-rise trousers, knitwear, and proper shoes. Formality levels from business suit to smart-casual separates. Never trendy, always refined.`,
    must: [
      "Suits (same-fabric top+bottom) or sport-jacket separates with complementary trousers. Striped suits must only be worn as matching sets \u2014 never mix a striped jacket with different trousers.",
      "Trousers must be mid-to-high rise with a hem width of ~19-20cm for classic, slightly narrower for smart-casual. NEVER skinny/super-slim.",
      "Shirts must have large collars that tuck under lapels. Small collars, mandarin/stand collars are disallowed.",
      "Blazer/suit lapels must be \u22658.5cm wide. Narrow lapels are a style error.",
      "Knit polos, turtlenecks, and fine-gauge tees are the correct base layers under jackets. Piqu\xE9 polos and basic cotton tees are too sporty for classic looks.",
      "Footwear: oxfords and derbies with suits; loafers (leather for dressy, suede for smart-casual); clean minimal leather sneakers only with casual trousers. Dark brown shoes are as formal as black but more versatile.",
      "Suits with smooth/office fabric require a shirt + tie. To go tieless, the suit must be in a seasonal/textured fabric (flannel, tweed, linen).",
      "Favor the golden 8 colors: white, blue, light blue, brown, green, cream, burgundy, grey. Blue+brown, blue+green, green+brown are foolproof formulas.",
      "Favor textured fabrics: flannel, tweed, herringbone, cashmere, merino, linen blends. Texture = luxury. Mixed fabric compositions (wool-linen-silk blends) are more textured and practical than pure fibers.",
      "Accessories complete the look: pocket squares, ties, seasonal scarves, watches. Tie and pocket square must coordinate but NOT be identical fabric. Pocket square is mandatory with any jacket \u2014 its absence is one of the biggest classic style errors.",
      "Socks must match trousers, not shoes. Grey, navy, and burgundy socks are the foundation. Black socks are disallowed.",
      "Navy blazer with metal buttons is the most versatile sport jacket \u2014 pairs with almost any trousers except very dark navy or graphite. Next additions: Glen check, Prince of Wales, Gun Club patterns for multi-color coordination."
    ],
    reject: [
      "NEVER include: hoodie, hoodies, hooded, windbreaker, windbreakers, puffer, puffers, zip-up, zip-ups, bomber, bombers, track jacket, track jackets.",
      "NEVER include: graphic, logo, print, printed, patterned, floral, paisley, camo, tie-dye, pattern, patterns, neon, rhinestones, sequins, pajama, pyjama, sleepwear, loungewear, t-shirt.",
      "NEVER include: cargo pants, cargo shorts, athletic shorts, basketball shorts, track pants, sweatpants, joggers, shorts/bermudas, drawstring pants, elastic-waist trousers.",
      "NEVER include: chunky sneakers, platform shoes, athletic trainers, running shoes, basketball shoes, square-toe shoes, driving moccasins, mules.",
      "NEVER include: souvenir jackets, floral/paisley/printed layers.",
      "NEVER include: black as a primary everyday color \u2014 navy, charcoal, dark brown are more versatile and elegant.",
      "NEVER include: exotic leather shoes (crocodile, python, stingray) \u2014 they look vulgar. Quality calf leather or suede only.",
      "NEVER pair: formal outerwear with shorts, sportswear with tailored pieces, backpacks with suits.",
      "NEVER: mix a striped suit jacket with non-matching trousers. Striped jackets only go with their matching pants.",
      "NEVER: wear a bow tie with a business suit \u2014 bow ties are for tuxedos/evening jackets only.",
      "NEVER: identical tie + pocket square fabric. They must coordinate, not match exactly.",
      "NEVER: smooth solid-color sport jackets without texture or pattern \u2014 they look cheap. Choose textured or patterned fabrics."
    ],
    palette: ["navy", "charcoal", "cream", "white", "beige", "camel", "brown", "burgundy", "forest green", "light blue", "grey", "olive", "tan"],
    fabrics: ["wool", "flannel", "tweed", "herringbone", "cashmere", "merino", "linen", "silk", "cotton", "poplin", "oxford cloth", "suede", "tropical wool", "glen check", "prince of wales", "gun club"]
  }
};
function getStyleContext(style) {
  const key = style.toLowerCase().replace(/[\s-]+/g, "_");
  return STYLE_CONTEXT[key] || STYLE_CONTEXT.casual;
}
function buildSystemPrompt(limit, style, layered, preferences, occasion, budget) {
  const ctx = getStyleContext(style);
  const shopOnly = isShopOnlyDeck(style, occasion) && !preferences?.lockedAnchorId;
  const styleKnowledge = style === "sport" ? WORKOUT_STYLE_KNOWLEDGE : STYLE_KNOWLEDGE;
  const slotRules = layered ? `- EVERY outfit MUST contain exactly FOUR items, one in each category:
    (1) LAYER (macroCategory="outerwear" \u2014 cardigan, blazer, lightweight jacket, coat, vest or sweater)
    (2) BASE SHIRT (macroCategory="top" \u2014 shirt, T-shirt or polo shirt, never another sweater or jacket)
    (3) PANTS (macroCategory="bottom" \u2014 pants, full-length trousers, chinos or jeans; never shorts or skirts)
    (4) SHOES (macroCategory="shoes")
  Never omit a category, substitute a dress for shirt and pants, or count one garment twice.` : `- EVERY outfit MUST contain exactly THREE items, one in each category:
    (1) BASE SHIRT (macroCategory="top" \u2014 shirt, T-shirt or polo shirt)
    (2) PANTS (macroCategory="bottom" \u2014 pants, full-length trousers, chinos or jeans; never shorts or skirts)
    (3) SHOES (macroCategory="shoes")
  Do NOT include any outerwear/layer/jacket in this request. Never substitute a dress for shirt and pants, or count one garment twice.`;
  const ownershipRule = shopOnly ? `- THIS IS A SHOP-ONLY DECK (Team Collaboration). EVERY item in EVERY outfit MUST have an id starting with "shop_". NEVER use user wardrobe items. Build all ${limit} looks purely from shop candidates.` : `- PRIORITIZE USER-OWNED ITEMS. Items whose id starts with "shop_" are optional catalog suggestions; use them only to complete a missing/sparse slot or add necessary variety. Every outfit should contain owned pieces whenever any compatible owned pieces exist.`;
  const budgetRule = budget ? `- HARD BUDGET ${budget.amount} ${budget.currency}: for EACH outfit, the sum of budgetCost values (converted to the budget currency) MUST be <= ${budget.amount}. Owned wardrobe items (source:user wardrobe) cost 0. Never pick a shop item whose budgetCost alone exceeds remaining budget. Prefer a cheaper shop substitute over dropping a required slot.` : `- Shop items may include a price. If the user prompt names a budget, shop prices in each look MUST sum to that cap or less (owned = 0). Always cite prices for shop picks. Without an explicit budget, prefer lower verified prices among equally suitable alternatives in the same currency; do not assume a premium price improves styling and do not invent a budget ceiling.`;
  const uniquenessRule = shopOnly ? `- NEVER reuse the same actual product in two looks when another valid product is available. Two candidates with the same product name or the same photo are the SAME product; candidates whose names differ only by color are DIFFERENT products.
- Different looks MAY use the same garment types and the same outfit structure. Rotate tops, bottoms AND shoes so each look uses a different actual product in every slot.` : `- NEVER reuse the same actual non-anchor item in two looks when another valid product is available. An item is the SAME product when the id, the product name or the photo matches.
- Different looks MAY use the same garment types and the same outfit structure. Rotate tops, bottoms and shoes so each look uses a different actual product in every slot.
- Reusing one anchor item across all looks is allowed ONLY when an ANCHOR ITEM is locked; all other slots MUST still rotate.`;
  const directionRule = `- PRODUCT UNIQUENESS ACROSS THE ${limit} LOOKS:
  - Each non-anchor item should appear in only one look. When a slot has fewer candidates than looks, it is acceptable to reuse that slot's product rather than break style, occasion, weather or budget.
  - Different looks MAY use the same garment types or the same overall outfit structure. Three blazer + shirt + trousers + loafer looks are acceptable if every blazer, shirt, trouser and shoe is a different actual product.
  - Do not sacrifice style, occasion, weather or budget correctness just to force different silhouettes or formulas.
  - Prefer some visual variety when equally good alternatives exist, but product uniqueness is the hard diversity requirement.`;
  return `You are a world-class fashion stylist curating outfits for a specific aesthetic. You must obey the STYLE DIRECTION below as if your reputation depends on it.

TASK: Create ${limit} complete outfits using ONLY items from the provided wardrobe, referenced by their exact id.

GENERAL STYLING RULES:
${slotRules}
${ownershipRule}
${budgetRule}
${uniquenessRule}
${directionRule}
- If any required slot has no suitable item, omit that outfit. Return fewer outfits or an empty outfits array rather than relaxing the request.
- NEVER invent items. Use ONLY the exact ids listed in "Available wardrobe items".
- Pick items whose styleTags, name, description, or material indicate they FIT the requested style. If an item's tags or description clash with the style, do NOT use it, even when it is the only option for that slot.
- Favor color harmony (tonal, monochromatic, or analogous palettes from the style's preferred palette). When layering, the base top must color-coordinate with the outerwear (tonal or complementary \u2014 never clashing).
- PRODUCT ROTATION: spread usage across the available pool. Do not put the same top, bottom, or shoes in all ${limit} outfits when alternatives exist. If a slot has \u2265${limit} candidates, each outfit MUST use a different actual product. The same garment TYPE repeating across looks is fine; the same actual PRODUCT repeating is not.
- NEVER pair formal outerwear (blazer, suit jacket, sport coat, overcoat, topcoat, trench, peacoat, tuxedo) with shorts or bermudas. If the only available bottom is shorts and a formal layer is required, omit the outfit.
- Each outfit needs a vivid 1-2 sentence description and 2 actionable styling tips.

USER'S SPECIFIC PERSONAL STYLE SETTINGS:
You MUST strictly satisfy the following style, color, and pattern settings chosen by the user in onboarding:
${preferences && preferences.favoriteColors && preferences.favoriteColors.length > 0 ? `- FAVORITE COLORS: ${preferences.favoriteColors.join(", ")}. Actively prioritize and suggest outfits featuring these colors.` : ""}
${preferences && preferences.avoidColors && preferences.avoidColors.length > 0 ? `- AVOID COLORS: ${preferences.avoidColors.join(", ")}. NEVER include any clothing item in these colors. This is a strict constraint.` : ""}
${preferences && preferences.fitPreference ? `- FIT PREFERENCE: ${preferences.fitPreference}. Choose garments that match this silhouette.` : ""}
${preferences && preferences.likedPatterns && preferences.likedPatterns.length > 0 ? `- LIKED PATTERNS: ${preferences.likedPatterns.join(", ")}. Prefer these patterns.` : ""}
${preferences && preferences.dislikedPatterns && preferences.dislikedPatterns.length > 0 ? `- DISLIKED PATTERNS: ${preferences.dislikedPatterns.join(", ")}. NEVER include any item featuring these patterns.` : ""}
${preferences && preferences.prefersSustainable ? `- SUSTAINABILITY: The user prefers sustainable materials when possible.` : ""}
${preferences && preferences.styleGoals && preferences.styleGoals.length > 0 ? `- STYLE GOALS: ${preferences.styleGoals.join(", ")}.` : ""}

${styleKnowledge}

STYLE DIRECTION \u2014 ${style.toUpperCase()}:
${ctx.vibe}

MUST:
${ctx.must.map((r) => `  \u2022 ${r}`).join("\n")}

MUST NOT:
${ctx.reject.map((r) => `  \u2022 ${r}`).join("\n")}

PREFERRED PALETTE: ${ctx.palette.join(", ")}
PREFERRED FABRICS: ${ctx.fabrics.join(", ")}

Self-check BEFORE responding:
  1. Did I pick only items whose description/material/styleTags match ${style}?
  2. Is the color palette coherent with ${style}?
  3. Did I avoid every REJECT category?
  4. Does any non-anchor item (same id, same product name or same photo) appear in more than one outfit? If so, rebuild the overlapping outfit with a different actual product (a different blazer/shirt/trousers/shoes \u2014 the same garment type and the same formula are fine).
  5. Are style, occasion, weather and budget still correct for every outfit? Never trade correctness for variety.
If the answer to any of these is "no", rebuild the outfit.

Respond ONLY in pure JSON (no markdown, no code fences):
{"outfits":[{"description":"brief explanation","items":[{"id":"EXACT_DB_ID"}]}]}`;
}
function buildUserPrompt(items, prompt, style, occasion, weather, limit, preferences) {
  const itemSummary = items.map((i) => {
    const name = i.name || i.type || "item";
    const desc = (i.description || "").toString().slice(0, 80);
    const tags = Array.isArray(i.styleTags) && i.styleTags.length > 0 ? i.styleTags.join(",") : "";
    const material = i.material || "";
    const pattern = i.pattern || "";
    const formality = typeof i.formality === "number" ? ` formality:${i.formality.toFixed(2)}` : "";
    const parts = [
      `[${i.id}]`,
      `${name}`,
      i.type ? `type:${i.type}` : "",
      `color:${i.color || "neutral"}`,
      `cat:${i.macroCategory || "other"}`,
      i.brand ? `brand:${i.brand}` : "",
      i.isShopItem || String(i.id || "").startsWith("shop_") ? "source:shop suggestion" : "source:user wardrobe",
      i.isShopItem || String(i.id || "").startsWith("shop_") ? typeof i.price === "number" && i.price > 0 ? `price:${i.price}${i.currency ? ` ${i.currency}` : ""}` : "price:unknown" : "price:owned(0)",
      preferences?.maxShopBudget ? `budgetCost:${shopPriceOf(i, preferences.maxShopBudget.currency, preferences.maxShopBudget.fx)} ${preferences.maxShopBudget.currency}` : "",
      material ? `material:${material}` : "",
      pattern ? `pattern:${pattern}` : "",
      tags ? `styleTags:${tags}` : ""
    ].filter(Boolean).join(" | ");
    return `\u2022 ${parts}${formality}${desc ? `
    desc: "${desc}"` : ""}`;
  }).join("\n");
  let prefText = "";
  if (preferences) {
    const lines = [];
    if (preferences.favoriteColors?.length) lines.push(`- Highly favored colors: ${preferences.favoriteColors.join(", ")}`);
    if (preferences.avoidColors?.length) lines.push(`- strictly banned colors (Do not use): ${preferences.avoidColors.join(", ")}`);
    if (preferences.fitPreference) lines.push(`- Fit preference: ${preferences.fitPreference}`);
    if (preferences.likedPatterns?.length) lines.push(`- Patterns to favor: ${preferences.likedPatterns.join(", ")}`);
    if (preferences.dislikedPatterns?.length) lines.push(`- Patterns to strictly avoid: ${preferences.dislikedPatterns.join(", ")}`);
    if (preferences.maxShopBudget?.amount) {
      lines.push(`- HARD BUDGET: shop budgetCost values in each outfit must sum to <= ${preferences.maxShopBudget.amount} ${preferences.maxShopBudget.currency || ""}. Owned items cost 0.`);
    }
    if (lines.length > 0) {
      prefText = `

REMINDER OF USER PREFERENCES:
${lines.join("\n")}`;
    }
  }
  const temp = typeof weather?.temp === "number" ? weather.temp : null;
  let climateRules = "";
  if (temp != null && temp < 18) {
    climateRules = style === "sport" ? `CLIMATE HARD RULES: It is ${temp}\xB0C. Choose warm performance layers, training trousers and closed athletic shoes. Avoid shorts, sandals and summer-only fabrics. Include the required warm outer layer.
` : `CLIMATE HARD RULES: It is ${temp}\xB0C (${weather?.condition || "cool"}). Do NOT pick linen, seersucker, tropical wool, shorts, sandals, or espadrilles. Prefer wool, merino, cashmere, flannel, cotton oxfords, and leather shoes. EVERY outfit MUST include a warm layer (wool blazer, knit, coat, cardigan) \u2014 a summer shirt alone is invalid.
`;
  } else if (temp != null && temp >= 26) {
    climateRules = `CLIMATE HARD RULES: It is ${temp}\xB0C. Do NOT pick puffer/parka/overcoat. Skip heavy tweed and flannel. Linen, cotton, and tropical wool are appropriate.
`;
  }
  return `${prompt ? `User's specific request: "${prompt}"

` : ""}Requested style: ${style || "Casual"}
${occasion ? `Occasion: ${occasion}
` : ""}${weather ? `Weather: ${weather.temp}\xB0C, ${weather.condition}
` : ""}${climateRules}

Available wardrobe items (use ONLY these exact ids; every id is pre-filtered to be at least plausible for ${style}, but you MUST still reject any item whose name/description/styleTags clash with the style):
${itemSummary}${prefText}

Create ${limit} distinct, on-aesthetic outfits. Each outfit must feel like it was hand-picked by a ${style} stylist, not a random color-matching algorithm.`;
}
function validateOutfitAgainstStyle(outfit, style, itemMap, layered, preferences) {
  const ctx = getStyleContext(style);
  const items = Array.isArray(outfit.items) ? outfit.items : [];
  const includesDress = items.some((item) => isDressItem(itemMap.get(String(item?.id))));
  const minItems = includesDress ? layered ? 3 : 2 : layered ? 4 : 3;
  if (items.length < minItems) return { ok: false, reason: `too few items (${items.length}, need ${minItems})` };
  const seenIds = /* @__PURE__ */ new Set();
  for (const it of items) {
    const key = String(it?.id);
    if (!itemMap.has(key)) return { ok: false, reason: `unknown item id "${it?.id}" (not in candidate pool)` };
    if (seenIds.has(key)) return { ok: false, reason: `duplicate item id "${it?.id}" inside one outfit` };
    seenIds.add(key);
  }
  for (const it of items) {
    const src = itemMap.get(String(it?.id)) || {};
    const macro = String(src.macroCategory || it.macroCategory || "").toLowerCase();
    if (macro !== "top" && macro !== "outerwear" && macro !== "bottom" && macro !== "shoes") continue;
    const nameOnly = `${it?.name || ""} ${src.name || ""}`.toLowerCase();
    const nameIsTop = /\bt-?shirts?\b|\btees?\b|\bpolos?\b|\bblouses?\b|\btanks?\b|\bhenleys?\b|\bshirts?\b|\boxfords?\b|\bbutton[\s-]?downs?\b/.test(nameOnly);
    const nameIsOuter = /\bjackets?|coats?|blazers?|hoodies?|cardigans?|sweaters?|pullovers?|puffers?|bombers?|vests?|outerwears?|trenchs?|peacoats?|sweatshirts?|knitwears?|jumpers?|fleece\b/.test(nameOnly);
    const nameIsBottom = /\bpants?|trousers?|jeans?|bottoms?|shorts|bermudas?|skirts?|chinos?|slacks?|joggers?|trunks?|swim/.test(nameOnly);
    const nameIsShoe = isOxfordFootwearName(nameOnly) || /\bshoes?|sneakers?|boots?|loafers?|sandals?|heels?|trainers?|derbys?|derbies|moccasins?|drivers?|mules?\b/.test(nameOnly);
    const nameSlot = nameIsShoe ? "shoes" : nameIsBottom ? "bottom" : nameIsOuter ? "outerwear" : nameIsTop ? "top" : null;
    if (nameSlot !== null && nameSlot !== macro) {
      return { ok: false, reason: `item "${it?.name || it?.id}" is a ${nameSlot} sitting in the ${macro} slot (mislabeled catalog row)` };
    }
  }
  const UNDERWEAR_RE2 = /\b(boxers?|briefs?|panties|lingerie|underwear|jockstrap|bra\b)/;
  for (const it of items) {
    const src = itemMap.get(String(it.id)) || {};
    const blob = `${it.name || ""} ${src.name || ""} ${src.description || ""} ${it.type || ""} ${src.type || ""}`.toLowerCase();
    if (UNDERWEAR_RE2.test(blob)) return { ok: false, reason: `underwear item "${it.name || it.id}" can never be an outfit part` };
  }
  const macros = /* @__PURE__ */ new Set();
  let hasDress = false;
  for (const it of items) {
    const src = itemMap.get(it.id) || {};
    const macro = (src.macroCategory || it.macroCategory || "").toLowerCase();
    if (macro) macros.add(macro);
    if (isDressItem(src) || isDressItem(it)) hasDress = true;
  }
  if (!hasDress) {
    for (const m of macros) {
      if (m !== "top" && m !== "outerwear" && m !== "bottom" && m !== "shoes") {
        return { ok: false, reason: `non-slot macro "${m}" fills no outfit slot` };
      }
    }
  }
  const hasBaseTop = macros.has("top") || hasDress;
  const hasOuter = macros.has("outerwear");
  const hasBottom = macros.has("bottom") || hasDress;
  const hasShoes = macros.has("shoes");
  if (!hasShoes) return { ok: false, reason: `missing shoes slot (macros: ${Array.from(macros).join(",")})` };
  if (!hasBottom) return { ok: false, reason: `missing bottom slot (macros: ${Array.from(macros).join(",")})` };
  if (layered) {
    if (!hasOuter) return { ok: false, reason: `missing outerwear/main-top slot (macros: ${Array.from(macros).join(",")})` };
    if (!hasBaseTop && !hasDress) return { ok: false, reason: `missing base-top slot (macros: ${Array.from(macros).join(",")})` };
  } else {
    if (!hasBaseTop && !hasOuter) return { ok: false, reason: `missing top/outerwear slot (macros: ${Array.from(macros).join(",")})` };
  }
  {
    const slotCounts = /* @__PURE__ */ new Map();
    for (const it of items) {
      const src = itemMap.get(it.id) || {};
      const macro = String(src.macroCategory || it.macroCategory || "").toLowerCase();
      if (!macro) continue;
      slotCounts.set(macro, (slotCounts.get(macro) || 0) + 1);
    }
    const needSingle = hasDress ? ["shoes"] : ["top", "outerwear", "bottom", "shoes"];
    const singleSlots = layered ? needSingle : hasDress ? ["shoes"] : ["bottom", "shoes"];
    for (const slot of singleSlots) {
      const n = slotCounts.get(slot) || 0;
      if (n > 1) return { ok: false, reason: `duplicate "${slot}" slot x${n} (macros: ${Array.from(macros).join(",")}) \u2014 one ${slot} per outfit` };
    }
    if (!layered && !hasDress) {
      const uppers = (slotCounts.get("top") || 0) + (slotCounts.get("outerwear") || 0);
      if (uppers > 1) return { ok: false, reason: `duplicate upper slot x${uppers} (macros: ${Array.from(macros).join(",")}) \u2014 one top per non-layered outfit` };
    }
    if (hasDress) {
      const dressCount = items.filter((it) => isDressItem(itemMap.get(it.id) || {}) || isDressItem(it)).length;
      if (dressCount > 1) return { ok: false, reason: `duplicate dress x${dressCount} \u2014 one dress per outfit` };
      if ((slotCounts.get("bottom") || 0) > 0) return { ok: false, reason: `dress + bottom (macros: ${Array.from(macros).join(",")}) \u2014 a dress replaces the bottom slot` };
      const extraTops = items.filter((it) => {
        const src = itemMap.get(it.id) || {};
        if (String(src.macroCategory || it.macroCategory || "").toLowerCase() !== "top") return false;
        return !(isDressItem(src) || isDressItem(it));
      });
      if (extraTops.length > 0) return { ok: false, reason: `dress + extra top "${extraTops[0]?.name || extraTops[0]?.id}" \u2014 a dress replaces the base-top slot` };
      const dressTarget = layered ? 3 : 2;
      if (items.length > dressTarget) return { ok: false, reason: `too many items for a dress look (${items.length}, need ${dressTarget}) \u2014 duplicate slot` };
    }
    const targetItems = layered ? 4 : 3;
    if (!hasDress && items.length > targetItems) return { ok: false, reason: `too many items (${items.length}, need ${targetItems}) \u2014 duplicate slot` };
  }
  if (preferences && Array.isArray(preferences.avoidColors) && preferences.avoidColors.length > 0) {
    const avoid = preferences.avoidColors.map((c) => c.toLowerCase().trim()).filter(Boolean);
    for (const it of items) {
      const src = itemMap.get(it.id) || {};
      const itemColor = (it.color || src.color || "").toLowerCase();
      const itemName = (it.name || src.name || "").toLowerCase();
      const itemDesc = (it.description || src.description || "").toLowerCase();
      for (const color of avoid) {
        if (itemColor.includes(color) || itemName.includes(` ${color} `) || itemDesc.includes(` ${color} `)) {
          return { ok: false, reason: `item "${it.name || it.id}" matches user's avoided color "${color}"` };
        }
      }
    }
  }
  if (preferences && Array.isArray(preferences.dislikedPatterns) && preferences.dislikedPatterns.length > 0) {
    const disliked = preferences.dislikedPatterns.map((p) => p.toLowerCase().trim()).filter(Boolean);
    for (const it of items) {
      const src = itemMap.get(it.id) || {};
      const itemPattern = (it.pattern || src.pattern || "").toLowerCase();
      const itemName = (it.name || src.name || "").toLowerCase();
      const itemDesc = (it.description || src.description || "").toLowerCase();
      for (const pattern of disliked) {
        if (itemPattern.includes(pattern) || itemName.includes(pattern) || itemDesc.includes(pattern)) {
          return { ok: false, reason: `item "${it.name || it.id}" matches user's disliked pattern "${pattern}"` };
        }
      }
    }
  }
  const rejectKeywords = extractRejectKeywords(ctx.reject);
  for (const it of items) {
    if (preferences?.lockedAnchorId && String(it.id) === preferences.lockedAnchorId) continue;
    const src = itemMap.get(it.id) || {};
    const blob = `${it.name || ""} ${src.name || ""} ${src.description || ""} ${src.type || ""} ${it.type || ""} ${it.pattern || ""} ${src.pattern || ""} ${it.material || ""} ${src.material || ""}`.toLowerCase();
    for (const kw2 of rejectKeywords) {
      if (blob.includes(kw2)) {
        return { ok: false, reason: `item "${it.name || it.id}" contains rejected keyword "${kw2}" for style ${style}` };
      }
    }
  }
  return { ok: true };
}
function extractRejectKeywords(rejectRules) {
  const out = [];
  for (const rule of rejectRules) {
    const m = rule.match(/(?:NEVER include|Avoid):\s*(.+?)\.?$/i);
    if (!m) continue;
    const parts = m[1].split(/,|;/).map((s) => s.trim().toLowerCase()).filter(Boolean);
    for (const p of parts) {
      out.push(p);
    }
  }
  return Array.from(new Set(out));
}
function validateOutfitCompatibility(outfit, itemMap, style) {
  const items = Array.isArray(outfit.items) ? outfit.items : [];
  let hasFormalLayer = false;
  let hasShorts = false;
  for (const it of items) {
    const src = itemMap.get(it.id) || {};
    const merged = { ...src, ...it };
    if (isFormalLayerItem(merged)) hasFormalLayer = true;
    if (isShortsItem(merged)) hasShorts = true;
  }
  if (hasFormalLayer && hasShorts) {
    return { ok: false, reason: "formal outerwear paired with shorts (disallowed)" };
  }
  const s = (style || "").toLowerCase().replace(/[\s-]+/g, "_");
  if ((s === "business_casual" || s === "old_money" || s === "classic" || s === "formal") && hasShorts) {
    return { ok: false, reason: `shorts disallowed in formal deck ${s}` };
  }
  return { ok: true };
}
function filterValidOutfits(outfits, style, itemMap, layered, preferences, occasion, weather, onReject) {
  const kept = [];
  const signatures = /* @__PURE__ */ new Set();
  const shopOnly = isShopOnlyDeck(style, occasion) && !preferences?.lockedAnchorId;
  const hasOwnedCandidates = !shopOnly && Array.from(itemMap.values()).some(
    (item) => !item.isShopItem && !String(item.id || "").startsWith("shop_")
  );
  for (const [outfitIndex, proposal] of outfits.entries()) {
    const reject = (reason2) => onReject?.(outfitIndex + 1, reason2.slice(0, 300));
    const resolved = (proposal.items || []).map((it) => itemMap.get(String(it.id))).filter(Boolean);
    const reason = styleJudgeReason(resolved, style, occasion, weather, preferences?.lockedAnchorId, preferences);
    if (reason || resolved.length !== (proposal.items || []).length) {
      reject(reason || "unknown_item_id");
      console.warn("[quality-reject]", { reason: reason || "unknown_item_id" });
      continue;
    }
    const structural = fourSlotOutfitReason(resolved, layered);
    if (structural) {
      reject(structural);
      console.warn("[quality-reject]", { reason: structural });
      continue;
    }
    const o = { ...proposal, items: resolved };
    const v = validateOutfitAgainstStyle(o, isWorkoutOccasion(occasion) ? "sport" : style, itemMap, layered, preferences);
    if (!v.ok) {
      reject(v.reason || "invalid_style_or_slots");
      console.log(`[validate] rejected outfit for ${style} (layered=${layered}): ${v.reason}`);
      continue;
    }
    const c = validateOutfitCompatibility(o, itemMap, isWorkoutOccasion(occasion) ? "sport" : style);
    if (!c.ok) {
      reject(c.reason || "incompatible_items");
      console.log(`[validate] rejected outfit for ${style} (layered=${layered}): ${c.reason}`);
      continue;
    }
    const budgetCap = preferences?.maxShopBudget;
    if (budgetCap && typeof budgetCap.amount === "number") {
      const spend = outfitShopSpend(o.items || [], itemMap, budgetCap.currency, budgetCap.fx);
      if (!fitsBudgetAmount(spend, budgetCap.amount)) {
        reject(`shop spend ${spend} exceeds ${budgetCap.amount} ${budgetCap.currency || ""}`);
        console.log(`[validate] rejected outfit over budget: shop spend ${spend} > ${budgetCap.amount}`);
        continue;
      }
    }
    if (isWorkishOccasion(occasion)) {
      let hasSwimBottom = false;
      for (const it of o.items || []) {
        const src = itemMap.get(String(it.id)) || {};
        if (isSwimBottomItem({ ...src, ...it })) {
          hasSwimBottom = true;
          break;
        }
      }
      if (hasSwimBottom) {
        reject("swim bottoms are not allowed for this occasion");
        console.log(`[validate] rejected swim bottom for workish occasion ${occasion}`);
        continue;
      }
      let hasDriverShoe = false;
      for (const it of o.items || []) {
        const src = itemMap.get(String(it.id)) || {};
        if (isDriverShoeItem({ ...src, ...it })) {
          hasDriverShoe = true;
          break;
        }
      }
      if (hasDriverShoe) {
        reject("driving moccasins are not allowed for this occasion");
        console.log(`[validate] rejected driving moccasin for workish occasion ${occasion}`);
        continue;
      }
    }
    const ids = (o.items || []).map((item) => String(item.id)).sort();
    const signature = ids.join("|");
    if (!signature || signatures.has(signature)) {
      reject("duplicate or empty outfit combination");
      console.log(`[validate] rejected duplicate outfit combination: ${signature}`);
      continue;
    }
    if (shopOnly) {
      const allShop = (o.items || []).length > 0 && (o.items || []).every((item) => {
        const source = itemMap.get(String(item.id)) || item;
        return Boolean(source.isShopItem) || String(source.id || item.id || "").startsWith("shop_");
      });
      if (!allShop) {
        reject("shop-only requests cannot include owned wardrobe items");
        console.log(`[validate] rejected mixed/wardrobe outfit for shop-only deck ${style}/${occasion}`);
        continue;
      }
    }
    if (hasOwnedCandidates) {
      const usesOwnedItem = (o.items || []).some((item) => {
        const source = itemMap.get(String(item.id)) || item;
        return !source.isShopItem && !String(source.id || item.id || "").startsWith("shop_");
      });
      if (!usesOwnedItem) {
        reject("use a compatible owned item when wardrobe candidates are available");
        console.log("[validate] rejected shop-only outfit while wardrobe items are available");
        continue;
      }
    }
    signatures.add(signature);
    kept.push(o);
  }
  return kept;
}
var PLACEHOLDER_SHOES = {
  id: "placeholder_shoes",
  type: "shoes",
  macroCategory: "shoes",
  color: "neutral",
  name: "Shoes",
  imageUrl: "",
  image: "",
  recommendation: "Add shoes to your wardrobe for better outfits",
  isShopItem: false
};
var SHOP_COLOR_WORDS = ["black", "white", "offwhite", "ecru", "cream", "beige", "stone", "sand", "taupe", "camel", "khaki", "olive", "green", "sage", "mint", "blue", "navy", "indigo", "brown", "chocolate", "grey", "gray", "silver", "charcoal", "red", "burgundy", "maroon", "pink", "purple", "lilac", "yellow", "mustard", "orange", "rust", "tan", "burgundy", "forest"];
var SHOP_BOILERPLATE_RE = /white-bg|white background|product shots?|front views?|back views?|side views?|flat lays?|ghost mannequins?|on models?|studio shots?|model shot|e-commerce/gi;
function inferShopColor(row) {
  for (const evidence of [row?.color, row?.primary_color, row?.name, row?.description]) {
    const blob = String(evidence || "").toLowerCase().replace(SHOP_BOILERPLATE_RE, " ");
    const words2 = SHOP_COLOR_WORDS.join("|");
    const match = blob.match(new RegExp(`\\b(?:(?:dark|light|pale|muted|dusty|pastel|bright|neon|electric|vivid|fluorescent)[ -]+)?(?:${words2})\\b`));
    if (match) return match[0];
  }
  return "neutral";
}
function filterShopByPrefs(items, preferences) {
  if (!preferences || !Array.isArray(items) || items.length === 0) return items;
  let out = [...items];
  if (Array.isArray(preferences.avoidColors) && preferences.avoidColors.length > 0) {
    const avoid = preferences.avoidColors.map((c) => c.toLowerCase().trim()).filter(Boolean);
    out = out.filter((item) => {
      const color = (item.color || "").toLowerCase();
      const name = (item.name || "").toLowerCase();
      const desc = (item.description || "").toLowerCase();
      return !avoid.some((c) => color.includes(c) || name.includes(` ${c} `) || desc.includes(` ${c} `));
    });
  }
  if (Array.isArray(preferences.dislikedPatterns) && preferences.dislikedPatterns.length > 0) {
    const disliked = preferences.dislikedPatterns.map((p) => p.toLowerCase().trim()).filter(Boolean);
    out = out.filter((item) => {
      const pattern = (item.pattern || "").toLowerCase();
      const name = (item.name || "").toLowerCase();
      const desc = (item.description || "").toLowerCase();
      return !disliked.some((p) => pattern.includes(p) || name.includes(p) || desc.includes(p));
    });
  }
  return out;
}
var CatalogLookupError = class extends Error {
  constructor() {
    super("catalog_lookup_unavailable");
  }
};
var discoveryRPCUnavailableUntil = /* @__PURE__ */ new Map();
async function fetchShopItemsForAllSlotsEdge(supabaseClient, requiredSlots, style, perSlot = 4, excludeSet, eligibility, budgetFx, nativeBudget, requestIntent = "", recycleTarget = 0, historyPriority, acceptsLayeredTop, generationSeed = crypto.randomUUID()) {
  const picks = [];
  const styleKey = style.toLowerCase().replace(/[\s-]+/g, "_");
  const ctx = getStyleContext(style);
  const rejectKeywords = extractRejectKeywords(ctx.reject);
  const controller = new AbortController();
  let timer;
  const deadline = new Promise((_, reject) => {
    timer = setTimeout(() => {
      controller.abort();
      reject(new CatalogLookupError());
    }, 15e3);
  });
  const capabilityScope = typeof supabaseClient.supabaseUrl === "string" ? supabaseClient.supabaseUrl : void 0;
  let discoveryAvailable = typeof supabaseClient.rpc === "function" && (!capabilityScope || Date.now() >= (discoveryRPCUnavailableUntil.get(capabilityScope) || 0));
  try {
    for (const slot of requiredSlots) {
      let cursor;
      let cursorCost = null;
      let nativePage = 0;
      let discoveryPage = 0;
      let useDiscovery = !budgetFx && !nativeBudget && discoveryAvailable;
      let scored = [];
      const compare = (a, b) => {
        const value = (entry) => budgetFx ? entry.row._budget_cost === null ? Infinity : Number(entry.row._budget_cost) : Number(entry.row.price);
        return (budgetFx || nativeBudget ? value(a) - value(b) : 0) || b.score - a.score;
      };
      const rowIdentity = (row) => ({ id: `shop_${row.id}`, name: row.name, imageUrl: row.image_url, macroCategory: slot });
      const excluded = (row) => Boolean(excludeSet && wasRecentlyShown(rowIdentity(row), excludeSet));
      const matchedTops = () => {
        if (slot !== "top" || !acceptsLayeredTop || !recycleTarget) return [];
        const layers = picks.filter((item) => item.macroCategory === "outerwear");
        const matched = /* @__PURE__ */ new Map(), seenProducts2 = /* @__PURE__ */ new Set();
        const assign = (entry, visited) => {
          for (let i = 0; i < layers.length; i++) {
            if (visited.has(i) || !acceptsLayeredTop(layers[i], { ...entry.row, macroCategory: "top", type: entry.row.garment_type || "top", color: inferShopColor(entry.row) })) continue;
            visited.add(i);
            const previous = matched.get(i);
            if (!previous || assign(previous, visited)) {
              matched.set(i, entry);
              return true;
            }
          }
          return false;
        };
        for (const entry of scored) {
          if (!entry.row.image_url || excluded(entry.row)) continue;
          const keys = productIdentityKeys(rowIdentity(entry.row));
          if (keys.some((key) => seenProducts2.has(key))) continue;
          keys.forEach((key) => seenProducts2.add(key));
          assign(entry, /* @__PURE__ */ new Set());
          if (matched.size >= Math.min(recycleTarget, layers.length)) break;
        }
        return [...matched.values()];
      };
      for (; ; ) {
        let q;
        if (useDiscovery) {
          q = supabaseClient.rpc("catalog_discovery_candidates", {
            p_slot: slot,
            p_source: "all",
            p_search: "",
            p_seed: generationSeed,
            p_limit: 100,
            p_offset: discoveryPage * 100,
            p_currency: null,
            p_max_price: null
          }).abortSignal(controller.signal);
        } else if (budgetFx) {
          q = supabaseClient.rpc("budget_catalog_candidates", {
            p_slot: slot,
            p_rates: budgetFx.snapshot.uzsPerUnit,
            p_after_cost: cursorCost,
            p_after_id: cursor || null,
            p_limit: 100,
            p_activity: styleKey === "sport"
          }).abortSignal(controller.signal);
        } else {
          q = supabaseClient.from("shop_catalog").select("id, brand, name, price, currency, image_url, garment_type, category, description, source, source_url").eq("is_active", true).limit(100).abortSignal(controller.signal);
          if (nativeBudget) q = q.eq("currency", nativeBudget.currency).lte("price", nativeBudget.amount).order("price", { ascending: true }).order("id", { ascending: true }).range(nativePage * 100, nativePage * 100 + 99);
          else q = q.order("id", { ascending: true });
          let slotFilter = "";
          if (slot === "shoes") slotFilter = "category.eq.shoes,garment_type.eq.shoes,name.ilike.%shoe%,name.ilike.%boot%,name.ilike.%loafer%,name.ilike.%trainer%,name.ilike.%sneaker%,name.ilike.%derby%,name.ilike.%espadrill%,name.ilike.%moccasin%";
          else if (slot === "outerwear") slotFilter = "category.eq.outerwear,garment_type.eq.outerwear,name.ilike.%jacket%,name.ilike.%coat%,name.ilike.%blazer%,name.ilike.%cardigan%,name.ilike.%sweater%,name.ilike.%hoodie%,name.ilike.%puffer%,name.ilike.%vest%";
          else if (slot === "top") slotFilter = "category.eq.tops,garment_type.eq.upper_body";
          else if (slot === "bottom") slotFilter = "category.eq.bottoms,garment_type.eq.lower_body";
          const evening = /black[ -]?tie|блэк[ -]?тай|смокинг/i.test(requestIntent);
          const eveningTerms = {
            top: ["tuxedo", "dress shirt", "formal shirt", "white shirt"],
            bottom: ["tuxedo", "dinner suit", "black trousers", "black pants", "formal trousers"],
            outerwear: ["tuxedo", "dinner jacket"],
            shoes: ["patent", "oxford", "opera pump"]
          };
          if (evening && eveningTerms[slot]) {
            const intentFilter = eveningTerms[slot].flatMap((term) => [`name.ilike.%${term}%`, `description.ilike.%${term}%`]).join(",");
            q = q.or(`and(or(${slotFilter}),or(${intentFilter}))`);
          } else if (styleKey === "sport") {
            const terms = ["athletic", "training", "running", "workout", "performance", "sportswear", "activewear", "gym", "\u0441\u043F\u043E\u0440\u0442\u0438\u0432\u043D", "\u0442\u0440\u0435\u043D\u0438\u0440\u043E\u0432"];
            const activityFilter = terms.flatMap((term) => [`name.ilike.%${term}%`, `description.ilike.%${term}%`]).join(",");
            q = q.or(`and(or(${slotFilter}),or(${activityFilter}))`);
          } else if (slotFilter) q = q.or(slotFilter);
          if (cursor && !nativeBudget) q = q.gt("id", cursor);
        }
        const pageKey = !useDiscovery && capabilityScope ? JSON.stringify([
          capabilityScope,
          slot,
          budgetFx?.snapshot.id,
          nativeBudget,
          cursor,
          cursorCost,
          nativePage,
          styleKey === "sport",
          /black[ -]?tie|блэк[ -]?тай|смокинг/i.test(requestIntent)
        ]) : void 0;
        const cachedPage = pageKey ? catalogPages.get(pageKey) : void 0;
        const cacheHit = !!cachedPage && Date.now() - cachedPage.at < CATALOG_PAGE_TTL;
        const pageResponse = cacheHit && cachedPage ? { data: cachedPage.rows, error: null } : await Promise.race([q, deadline]);
        const data = pageResponse.data;
        const error = pageResponse.error;
        if (pageKey && !cacheHit && !error && Array.isArray(data)) {
          if (catalogPages.size >= 64) catalogPages.delete(catalogPages.keys().next().value);
          catalogPages.set(pageKey, { at: Date.now(), rows: data });
        }
        if (useDiscovery && error && ["PGRST202", "42883"].includes(error.code)) {
          useDiscovery = false;
          discoveryAvailable = false;
          if (capabilityScope) {
            discoveryRPCUnavailableUntil.set(capabilityScope, Date.now() + 6e4);
            if (discoveryRPCUnavailableUntil.size > 8) discoveryRPCUnavailableUntil.delete(discoveryRPCUnavailableUntil.keys().next().value);
          }
          console.info("catalog_retrieval", { strategy: "bounded-keyset-cache", slot });
          continue;
        }
        if (controller.signal.aborted || error || !Array.isArray(data)) throw new CatalogLookupError();
        if (!data.length && !useDiscovery) break;
        if (budgetFx && data.some((row) => row._budget_cost !== null && (typeof row._budget_cost !== "string" || !/^[0-9]+([.][0-9]+)?$/.test(row._budget_cost)))) throw new CatalogLookupError();
        const pageScored = data.map((row) => ({
          row,
          score: scoreGarmentForContext({ ...row, macroCategory: slot }, stylingContext({ style, occasion: requestIntent })) * 0.6 + 0.4 * scoreItemForStyle(
            { name: row.name, description: row.description, brand: row.brand, color: inferShopColor(row), type: row.garment_type, category: row.category, macroCategory: slot },
            normalizeStyleId(styleKey)
          )
        })).filter(({ row }) => {
          if (isNonOutfitItem(row)) return false;
          const blob = `${(row.name || "").toLowerCase()} ${(row.description || "").toLowerCase()} ${row.garment_type || ""} ${row.category || ""}`.toLowerCase();
          for (const kw2 of rejectKeywords) {
            if (blob.includes(kw2)) return false;
          }
          if (eligibility && !eligibility({
            ...row,
            id: `shop_${row.id}`,
            isShopItem: true,
            isWardrobe: false,
            type: row.garment_type || row.category || slot,
            color: inferShopColor(row),
            macroCategory: slot
          }, picks)) return false;
          if (styleKey === "sport" && rejectForActivity({ ...row, macroCategory: slot }, "workout")) return false;
          const resolved = macroCategory(row.category || "", row.garment_type || "", row.name || "");
          if (resolved !== "other" && resolved !== slot) return false;
          const nameMismatch = detectSlotMismatch(slot, row.name || "", row.garment_type || "");
          if (nameMismatch && nameMismatch !== slot) return false;
          const rowType = String(row.garment_type || "").toLowerCase();
          if (rowType === "accessory") return false;
          const nameBlob2 = String(row.name || "").toLowerCase();
          if (/\b(wallets?|card(\s|-)holders?|purses?|handbags?|backpacks?|belts?|scarves?|scarf|hats?|beanies?|jewelry|jewellery|watches?|sunglasses|gloves?|keychains?|cufflinks?|pocket[\s-]?squares?)\b/.test(nameBlob2)) return false;
          const itemName = String(row.name || "").trim();
          if (!itemName || /^[0-9\s\-]+$/.test(itemName) || itemName === String(row.id)) return false;
          return true;
        }).sort(compare);
        scored.push(...pageScored);
        scored.sort(compare);
        if (useDiscovery) {
          discoveryPage++;
          const distinct = new Set(scored.filter((entry) => !excluded(entry.row)).map((entry) => canonicalProductKey(rowIdentity(entry.row)))).size;
          if (data.length < 100 || discoveryPage >= 3) {
            if (distinct >= perSlot && (!acceptsLayeredTop || slot !== "top" || matchedTops().length >= Math.min(recycleTarget, picks.filter((item) => item.macroCategory === "outerwear").length))) {
              console.info("catalog_retrieval", { strategy: "balanced-rpc", slot, pages: discoveryPage, candidates: scored.length });
              break;
            }
            useDiscovery = false;
            scored = [];
            console.info("catalog_retrieval", { strategy: "bounded-keyset-cache", slot, reason: "candidate_pressure" });
          }
          continue;
        }
        if (data.length < 100) break;
        const distinctFresh = countDistinctProducts(scored.filter((entry) => !excluded(entry.row)).map((entry) => rowIdentity(entry.row)));
        if (nativePage + 1 >= MAX_LEGACY_PAGES_PER_SLOT || nativePage >= 2 && distinctFresh >= perSlot && (!acceptsLayeredTop || slot !== "top" || matchedTops().length >= recycleTarget)) {
          console.info("catalog_retrieval", { strategy: "bounded-keyset-cache", slot, pages: nativePage + 1, distinctFresh, exhaustive: false });
          if (recycleTarget && distinctFresh < recycleTarget) throw new CatalogLookupError();
          break;
        }
        const next = data[data.length - 1]?.id;
        if (typeof next !== "string" || !next || next === cursor) throw new CatalogLookupError();
        cursor = next;
        nativePage++;
        if (budgetFx) cursorCost = data[data.length - 1]._budget_cost;
      }
      const seenProducts = /* @__PURE__ */ new Set();
      const seenIdentityKeys = /* @__PURE__ */ new Set();
      const slotPicks = [];
      let added = 0;
      const pushRow = (row) => {
        const pkey = canonicalProductKey(rowIdentity(row));
        const keys = productIdentityKeys(rowIdentity(row));
        if (seenProducts.has(pkey) || keys.some((key) => seenIdentityKeys.has(key))) return;
        seenProducts.add(pkey);
        keys.forEach((key) => seenIdentityKeys.add(key));
        slotPicks.push({
          id: `shop_${row.id}`,
          type: row.garment_type || row.category || slot,
          category: row.category || slot,
          macroCategory: slot,
          name: row.name || row.brand || "Shop pick",
          brand: row.brand || "",
          color: inferShopColor(row),
          imageUrl: row.image_url,
          image: row.image_url,
          style: style || "Casual",
          isShopItem: true,
          price: row.price || void 0,
          currency: row.currency,
          shopUrl: row.source_url,
          description: row.description || "",
          recommendation: `Suggested from shop to complete your ${slot}`
        });
        added++;
      };
      for (const { row } of [...matchedTops(), ...scored]) {
        if (added >= perSlot) break;
        if (!row || !row.image_url) continue;
        if (excluded(row)) continue;
        pushRow(row);
      }
      if (recycleTarget && countDistinctProducts(slotPicks) < recycleTarget) {
        const historical = [...scored].sort((a, b) => compareRotationOrder(historyPriority?.([rowIdentity(b.row)]) || [], historyPriority?.([rowIdentity(a.row)]) || []));
        for (const { row } of historical) {
          if (added >= perSlot) break;
          if (row?.image_url && excluded(row)) pushRow(row);
        }
      }
      picks.push(...slotPicks);
    }
    return picks;
  } catch (_) {
    throw new CatalogLookupError();
  } finally {
    if (timer !== void 0) clearTimeout(timer);
  }
}
function requiredShopSupplementSlots(items, requiredSlots, limit, selectedItemIds = [], eligibility) {
  const manualSelection = Array.isArray(selectedItemIds) && selectedItemIds.length > 0;
  const targetChoicesPerSlot = manualSelection ? 1 : limit >= 10 ? 16 : items.some((item) => !isShopRow(item)) ? 6 : 12;
  const countBySlot = /* @__PURE__ */ new Map();
  const seenProducts = /* @__PURE__ */ new Set();
  for (const item of items) {
    if (eligibility && !eligibility(item)) continue;
    const slot = String(item.macroCategory || "").toLowerCase();
    if (!slot) continue;
    const key = `${slot}:${isShopRow(item) ? canonicalProductKey(item) : item.id}`;
    if (seenProducts.has(key)) continue;
    seenProducts.add(key);
    countBySlot.set(slot, (countBySlot.get(slot) || 0) + 1);
  }
  return {
    targetChoicesPerSlot,
    slots: requiredSlots.filter((slot) => (countBySlot.get(slot) || 0) < targetChoicesPerSlot)
  };
}
async function localFallback(items, style, occasion, limit, layered, supabaseClient, preferences, wardrobeOnly = false, deckSalt = -1, weather, qualityOf = assessOutfitQuality) {
  if ((limit ?? 3) <= 0) return [];
  const shopOnly = isShopOnlyDeck(style, occasion) && !preferences?.lockedAnchorId;
  let filteredItems = [...items];
  if (preferences) {
    if (Array.isArray(preferences.avoidColors) && preferences.avoidColors.length > 0) {
      const avoid = preferences.avoidColors.map((c) => c.toLowerCase().trim()).filter(Boolean);
      filteredItems = filteredItems.filter((item) => {
        const color = (item.color || item.primary_color || "").toLowerCase();
        const name = (item.name || "").toLowerCase();
        const desc = (item.description || "").toLowerCase();
        return !avoid.some((c) => color.includes(c) || name.includes(` ${c} `) || desc.includes(` ${c} `));
      });
    }
  }
  if (shopOnly) {
    filteredItems = filteredItems.filter((item) => item.isShopItem || String(item.id || "").startsWith("shop_"));
  }
  {
    const styleCtx = getStyleContext(style);
    const rejKws = extractRejectKeywords(styleCtx.reject);
    if (rejKws.length > 0 && filteredItems.length > 0) {
      const blobOf2 = (it) => `${it?.name || ""} ${it?.description || ""} ${it?.type || ""} ${it?.pattern || ""} ${it?.material || ""} ${it?.brand || ""}`.toLowerCase();
      const clean = filteredItems.filter((it) => !rejKws.some((kw2) => blobOf2(it).includes(kw2)));
      const slotOf = (it) => String(it.macroCategory || "").toLowerCase();
      const needSlots = ["top", "bottom", "shoes"];
      if (layered) needSlots.push("outerwear");
      const guarded = [...clean];
      for (const s of needSlots) {
        if (!guarded.some((it) => slotOf(it) === s)) {
          for (const it of filteredItems) {
            if (slotOf(it) === s && !guarded.includes(it)) guarded.push(it);
          }
        }
      }
      if (guarded.length > 0 && guarded.length !== filteredItems.length) {
        console.log(`[style-guard] ${style}: ${filteredItems.length} -> ${guarded.length} items`);
      }
      if (guarded.length > 0) filteredItems = guarded;
    }
  }
  const sortByFavorite = (pool) => {
    if (!preferences || !Array.isArray(preferences.favoriteColors) || preferences.favoriteColors.length === 0) {
      return pool;
    }
    const favorites = preferences.favoriteColors.map((c) => c.toLowerCase().trim()).filter(Boolean);
    return [...pool].sort((a, b) => {
      const colorA = (a.color || a.primary_color || "").toLowerCase();
      const colorB = (b.color || b.primary_color || "").toLowerCase();
      const matchA = favorites.some((c) => colorA.includes(c)) ? 1 : 0;
      const matchB = favorites.some((c) => colorB.includes(c)) ? 1 : 0;
      return matchB - matchA;
    });
  };
  let baseTops = shuffleArray(filteredItems.filter((i) => i.macroCategory === "top"));
  let outerwear = shuffleArray(filteredItems.filter((i) => i.macroCategory === "outerwear"));
  let legacyTops = shuffleArray(filteredItems.filter((i) => ["top", "outerwear"].includes(i.macroCategory)));
  let bottoms = shuffleArray(filteredItems.filter((i) => i.macroCategory === "bottom"));
  const nonShortsBottoms = shuffleArray(bottoms.filter((b) => !isShortsItem(b)));
  const casualOuterwear = shuffleArray(outerwear.filter((o) => !isFormalLayerItem(o)));
  let shoes = shuffleArray(filteredItems.filter((i) => i.macroCategory === "shoes"));
  if (preferences) {
    baseTops = sortByFavorite(baseTops);
    outerwear = sortByFavorite(outerwear);
    legacyTops = sortByFavorite(legacyTops);
    bottoms = sortByFavorite(bottoms);
    shoes = sortByFavorite(shoes);
  }
  const requiredSlots = ["top", "bottom", "shoes"];
  if (layered) requiredSlots.push("outerwear");
  const shopSupplement = shopOnly ? { slots: requiredSlots, targetChoicesPerSlot: limit >= 10 ? 16 : 8 } : requiredShopSupplementSlots(filteredItems, requiredSlots, limit);
  const { slots: fillSlots, targetChoicesPerSlot } = shopSupplement;
  let shopFills = [];
  if (wardrobeOnly) {
    console.log(`[wardrobeOnly] localFallback skipping shop supplement \u2014 ${filteredItems.length} wardrobe item(s) only`);
  } else if (supabaseClient) {
    try {
      shopFills = await fetchShopItemsForAllSlotsEdge(supabaseClient, fillSlots, style, targetChoicesPerSlot);
      shopFills = filterShopByPrefs(shopFills, preferences);
      if (shopFills.length > 0) {
        console.log(`[shop-fill] Local fallback added ${shopFills.length} shop item(s) for sparse slots: ${fillSlots.join(", ")} (target=${targetChoicesPerSlot})`);
      }
    } catch (_) {
    }
  }
  const allItems = [...filteredItems, ...shopFills];
  const shopOnlyPool = shopOnly ? [...shopFills, ...filteredItems.filter((i) => i.isShopItem || String(i.id || "").startsWith("shop_"))] : null;
  const poolItems = shopOnly ? shopOnlyPool ?? [] : allItems;
  const deckShopOnly = shopOnly;
  const targetItems = layered ? 4 : 3;
  const styleLabel = (() => {
    const key = String(style || "casual").toLowerCase();
    const known = {
      old_money: "Old Money",
      semi_classic: "Semi-Classic",
      minimalist: "Minimalist",
      casual: "Casual",
      classic: "Classic",
      business_casual: "Modern Professional"
    };
    return known[key] || "Casual";
  })();
  const stylePhrase = `${/^[aeiou]/i.test(styleLabel) ? "An" : "A"} ${styleLabel}`;
  const allBaseTops = poolItems.filter((i) => i.macroCategory === "top");
  const allOuterwear = poolItems.filter((i) => i.macroCategory === "outerwear");
  const allLegacyTops = poolItems.filter((i) => ["top", "outerwear"].includes(i.macroCategory));
  const allBottoms = poolItems.filter((i) => i.macroCategory === "bottom");
  const allNonShortsBottoms = allBottoms.filter((b) => !isShortsItem(b));
  const allCasualOuterwear = allOuterwear.filter((o) => !isFormalLayerItem(o));
  const allShoes = poolItems.filter((i) => i.macroCategory === "shoes");
  const ownedItems = deckShopOnly ? [] : allItems.filter((i) => !i.isShopItem && !String(i.id || "").startsWith("shop_"));
  const shopItems = (shopOnly ? shopOnlyPool ?? [] : allItems).filter((i) => i.isShopItem || String(i.id || "").startsWith("shop_"));
  const isOwnedPart = (p) => p && !p.isShopItem && !String(p.id || "").startsWith("shop_") && p.id !== PLACEHOLDER_SHOES.id;
  const isShopPart = (p) => p && (p.isShopItem || String(p.id || "").startsWith("shop_"));
  const replacePart = (parts, replacement, protectId) => {
    const slot = String(replacement.macroCategory || "").toLowerCase();
    const sameSlotIndex = parts.findIndex((p) => String(p.macroCategory || "").toLowerCase() === slot && p.id !== replacement.id && p.id !== protectId);
    if (sameSlotIndex >= 0) {
      parts[sameSlotIndex] = replacement;
      return;
    }
    const shopIndex = parts.findIndex((p) => isShopPart(p) && p.id !== protectId);
    if (shopIndex >= 0) {
      parts[shopIndex] = replacement;
      return;
    }
    const replaceableIndex = parts.findIndex((p) => p.id !== protectId);
    if (replaceableIndex >= 0) parts[replaceableIndex] = replacement;
  };
  const isFormalStyle = ["old_money", "classic", "business_casual"].includes(style.toLowerCase());
  if (isFormalStyle && allNonShortsBottoms.length > 0) {
    allBottoms.length = 0;
    allBottoms.push(...allNonShortsBottoms);
  }
  const spreadPoolsEdge = () => {
    const spreadInPlace = (pool, slot) => {
      if (pool.length <= 2) return;
      const spread = spreadBySimEdge(pool, (p) => simKeyEdge(p.name || p.type || "", slot));
      pool.length = 0;
      pool.push(...spread);
    };
    spreadInPlace(allBaseTops, "top");
    spreadInPlace(allOuterwear, "outerwear");
    spreadInPlace(allBottoms, "bottom");
    spreadInPlace(allShoes, "shoes");
    spreadInPlace(allLegacyTops, "top");
  };
  spreadPoolsEdge();
  const outfits = [];
  const slotSize = (pool) => Math.max(pool.length, 1);
  const seenSigs = /* @__PURE__ */ new Set();
  const DIVERSE_STEPS = [7, 11, 13, 17];
  const salt = deckSalt < 0 ? Math.floor(Math.random() * 32) : Number(deckSalt) || 0;
  const deckSeed = (deckSeedEdge(style, occasion) + salt) % 32;
  const deckReverse = salt % 2 === 1 ? !deckReverseEdge(style, occasion) : deckReverseEdge(style, occasion);
  const diverseIdx = (outfitIdx, slotOffset, attempt, step, len) => {
    if (len <= 0) return 0;
    const base = (outfitIdx + deckSeed + slotOffset + attempt * step) % len;
    return deckReverse ? len - 1 - base : base;
  };
  const buildParts = (i, attemptForPick) => {
    const parts = [];
    const topIndex = diverseIdx(i, 0, attemptForPick, DIVERSE_STEPS[0], slotSize(allOuterwear));
    const baseIndex = diverseIdx(i, 3, attemptForPick, DIVERSE_STEPS[1], slotSize(allBaseTops));
    const bottomIndex = diverseIdx(i, 6, attemptForPick, DIVERSE_STEPS[2], slotSize(allBottoms));
    const shoeIndex = diverseIdx(i, 9, attemptForPick, DIVERSE_STEPS[3], slotSize(allShoes));
    const legacyTopIndex = diverseIdx(i, 0, attemptForPick, DIVERSE_STEPS[0], slotSize(allLegacyTops));
    const legacyBottomIndex = diverseIdx(i, 3, attemptForPick, DIVERSE_STEPS[1], slotSize(allBottoms));
    const bottom = allBottoms[(layered ? bottomIndex : legacyBottomIndex) % slotSize(allBottoms)] || allBottoms[0];
    const bottomIsShorts = !!bottom && isShortsItem(bottom);
    if (layered) {
      let outer = allOuterwear[topIndex % slotSize(allOuterwear)];
      if (bottomIsShorts && outer && isFormalLayerItem(outer)) {
        outer = allCasualOuterwear[topIndex % slotSize(allCasualOuterwear)] || void 0;
      }
      const mainTop = outer || allLegacyTops[legacyTopIndex % slotSize(allLegacyTops)];
      if (mainTop) parts.push({ ...mainTop, recommendation: deckShopOnly ? "Suggested from shop" : "Main top / outerwear layer" });
      const base = allBaseTops[baseIndex % slotSize(allBaseTops)];
      const baseTopItem = base || outer || allLegacyTops[(baseIndex + 1) % slotSize(allLegacyTops)];
      if (baseTopItem && baseTopItem.id !== mainTop?.id) {
        parts.push({ ...baseTopItem, recommendation: deckShopOnly ? "Suggested from shop" : "Base top worn underneath" });
      }
    } else {
      const top = allLegacyTops[legacyTopIndex % slotSize(allLegacyTops)];
      if (top) parts.push({ ...top, recommendation: deckShopOnly ? "Suggested from shop" : "Key piece" });
    }
    let finalBottom = bottom;
    if (layered && bottomIsShorts) {
      const mainTopItem = parts[0];
      if (mainTopItem && isFormalLayerItem(mainTopItem) && allNonShortsBottoms.length > 0) {
        finalBottom = allNonShortsBottoms[bottomIndex % allNonShortsBottoms.length];
      }
    }
    const nonLayeredShoeIndex = diverseIdx(i, 6, attemptForPick, DIVERSE_STEPS[2], slotSize(allShoes));
    const shoe = allShoes[(layered ? shoeIndex : nonLayeredShoeIndex) % slotSize(allShoes)] || allShoes[0];
    if (finalBottom) parts.push({ ...finalBottom, recommendation: deckShopOnly ? "Suggested from shop" : "Pairs well" });
    if (shoe) parts.push({ ...shoe, recommendation: deckShopOnly ? "Suggested from shop" : "Completes the look" });
    else if (!deckShopOnly) parts.push({ ...PLACEHOLDER_SHOES });
    if (layered && parts.length < targetItems && poolItems.length > 0) {
      const presentMacros = new Set(parts.map((p) => String(p.macroCategory || "").toLowerCase()));
      let fillIndex = 0;
      while (parts.length < targetItems && fillIndex < poolItems.length) {
        const fillItem = poolItems[(fillIndex + i) % poolItems.length];
        const alreadyAdded = parts.some((p) => p.id === fillItem.id);
        const fillMacro = String(fillItem.macroCategory || "").toLowerCase();
        if (!alreadyAdded && !presentMacros.has(fillMacro)) {
          parts.push({ ...fillItem, recommendation: deckShopOnly ? "Suggested from shop" : "Complementary piece" });
          presentMacros.add(fillMacro);
        }
        fillIndex++;
      }
    }
    if (!layered && parts.length > targetItems) parts.length = targetItems;
    {
      const seenIds = /* @__PURE__ */ new Set();
      for (let pi = 0; pi < parts.length; pi++) {
        const pid = String(parts[pi].id);
        if (!seenIds.has(pid)) {
          seenIds.add(pid);
          continue;
        }
        const slot = String(parts[pi].macroCategory || "").toLowerCase();
        const slotPool = slot === "top" ? allBaseTops : slot === "outerwear" ? allOuterwear : slot === "bottom" ? allBottoms : slot === "shoes" ? allShoes : poolItems;
        const repl = slotPool.find((c) => !seenIds.has(String(c.id))) || poolItems.find((c) => !seenIds.has(String(c.id)));
        if (repl) {
          parts[pi] = { ...repl, recommendation: parts[pi].recommendation };
          seenIds.add(String(repl.id));
        } else {
          parts.splice(pi, 1);
          pi--;
        }
      }
      if (parts.length > 0 && parts.length < targetItems && poolItems.length > 0) {
        const presentAfterDrop = new Set(parts.map((p) => String(p.macroCategory || "").toLowerCase()));
        for (const fill of poolItems) {
          if (parts.length >= targetItems) break;
          if (!fill || parts.some((p) => p.id === fill.id)) continue;
          if (deckShopOnly && !isShopPart(fill)) continue;
          const fillMacro = String(fill.macroCategory || "").toLowerCase();
          if (presentAfterDrop.has(fillMacro)) continue;
          parts.push({ ...fill, recommendation: deckShopOnly ? "Suggested from shop" : "Complementary piece" });
          presentAfterDrop.add(fillMacro);
        }
      }
    }
    if (!parts.length) return null;
    if (deckShopOnly) {
      const filtered = parts.filter((p) => isShopPart(p));
      if (filtered.length < targetItems) return null;
      parts.length = 0;
      parts.push(...filtered.slice(0, targetItems));
    }
    let protectedOwnedId;
    if (!deckShopOnly && ownedItems.length > 0 && !parts.some(isOwnedPart)) {
      const owned = { ...ownedItems[i % ownedItems.length], recommendation: "From your wardrobe" };
      protectedOwnedId = owned.id;
      replacePart(parts, owned);
    } else if (!deckShopOnly) {
      protectedOwnedId = parts.find(isOwnedPart)?.id;
    }
    if (shopItems.length > 0 && !parts.some(isShopPart)) {
      replacePart(parts, { ...shopItems[i % shopItems.length], recommendation: "Suggested from shop" }, protectedOwnedId);
    }
    if (deckShopOnly && !parts.every(isShopPart)) return null;
    {
      const slotCounts = /* @__PURE__ */ new Map();
      for (const p of parts) {
        const m = String(p?.macroCategory || "").toLowerCase();
        if (!m) continue;
        slotCounts.set(m, (slotCounts.get(m) || 0) + 1);
      }
      for (const slot of ["top", "outerwear", "bottom", "shoes"]) {
        if ((slotCounts.get(slot) || 0) > 1) return null;
      }
    }
    {
      const hasShorts = parts.some((p) => isShortsItem(p));
      const hasFormal = parts.some((p) => isFormalLayerItem(p));
      const sKey = (style || "").toLowerCase();
      if (hasShorts && (hasFormal || ["old_money", "classic", "business_casual"].includes(sKey))) return null;
    }
    if (isWorkishOccasion(occasion)) {
      const hasBeachShoe = parts.some((p) => {
        if (String(p.macroCategory || "").toLowerCase() !== "shoes") return false;
        const b = `${p.name || ""} ${p.type || ""}`.toLowerCase();
        return /\b(boat|espadrilles?|mules?|backless|slip[\s-]?ons?|moccasins?|drivers?|driving|gommino|sandals?|slides?)\b/.test(b);
      });
      if (hasBeachShoe) return null;
    }
    const budgetCap = preferences?.maxShopBudget;
    if (budgetCap && typeof budgetCap.amount === "number") {
      const fitted = fitOutfitToBudget(parts, poolItems, budgetCap);
      parts.length = 0;
      parts.push(...fitted);
      if (!fitsBudgetAmount(outfitShopSpend(parts, void 0, budgetCap.currency, budgetCap.fx), budgetCap.amount)) return null;
    }
    return parts;
  };
  const keepParts = (parts) => {
    if (fourSlotOutfitReason(parts) || styleJudgeReason(parts, style, occasion, weather, preferences?.lockedAnchorId, preferences)) return false;
    const sig = parts.map((p) => String(p.id)).sort().join("|");
    if (!sig || seenSigs.has(sig)) return false;
    seenSigs.add(sig);
    outfits.push({
      id: `local_${Date.now()}_${Math.floor(Math.random() * 1e5)}`,
      style: style || "Casual",
      occasion: occasion || "Everyday",
      description: deckShopOnly ? `${stylePhrase} look curated from the shop.` : `${stylePhrase} look built around your wardrobe.`,
      confidence: 0.75 + Math.random() * 0.1,
      // Slight confidence variation
      items: parts,
      stylingTips: pickStylingTips(layered)
    });
    return true;
  };
  let edgeAttempt = 0;
  for (let i = 0; outfits.length < limit && i < limit * 12; i++) {
    const candidate = buildParts(i, edgeAttempt);
    if (!candidate) {
      edgeAttempt++;
      continue;
    }
    if (keepParts(candidate)) edgeAttempt = 0;
    else edgeAttempt++;
  }
  if (outfits.length === 0 && allItems.length > 0) {
    const parts = [];
    if (layered) {
      const candidateBottom = allBottoms[0] || allItems[0];
      const bottomIsShorts = !!candidateBottom && isShortsItem(candidateBottom);
      let outer = allOuterwear[0] || allLegacyTops[0];
      if (bottomIsShorts && outer && isFormalLayerItem(outer)) {
        outer = allCasualOuterwear[0] || allLegacyTops.find((t) => !isFormalLayerItem(t)) || outer;
      }
      const base = allBaseTops[0] || allLegacyTops[0] || outer;
      const finalBottom = bottomIsShorts && outer && isFormalLayerItem(outer) && allNonShortsBottoms[0] ? allNonShortsBottoms[0] : candidateBottom;
      const shoe = allShoes[0];
      if (outer) parts.push({ ...outer, recommendation: "Main top / outerwear layer" });
      if (base && base.id !== outer?.id) parts.push({ ...base, recommendation: "Base top worn underneath" });
      if (finalBottom) parts.push({ ...finalBottom, recommendation: "Pairs well" });
      if (shoe && shoe.id !== finalBottom?.id) parts.push({ ...shoe, recommendation: deckShopOnly ? "Suggested from shop" : "Completes the look" });
      else if (!deckShopOnly) parts.push({ ...PLACEHOLDER_SHOES });
      if (parts.length > 0 && parts.length < 4) {
        for (const fill of poolItems) {
          if (parts.length >= 4) break;
          if (!fill || parts.some((p) => p.id === fill.id)) continue;
          if (deckShopOnly && !isShopPart(fill)) continue;
          parts.push({ ...fill, recommendation: "Complementary piece" });
        }
      }
    } else {
      const top = allLegacyTops[0] || allItems[0];
      const bottom = allBottoms[0] || allItems[1] || allItems[0];
      const shoe = allShoes[0];
      if (top) parts.push({ ...top, recommendation: "Key piece" });
      if (bottom && bottom.id !== top?.id) parts.push({ ...bottom, recommendation: "Pairs well" });
      if (shoe && shoe.id !== bottom?.id) parts.push({ ...shoe, recommendation: deckShopOnly ? "Suggested from shop" : "Completes the look" });
      else if (!deckShopOnly) parts.push({ ...PLACEHOLDER_SHOES });
      if (parts.length > 3) parts.length = 3;
    }
    if (parts.length > 0) {
      outfits.push({
        id: `local_0_${Date.now()}`,
        style: style || "Casual",
        occasion: occasion || "Everyday",
        description: `${stylePhrase} look built around your wardrobe.`,
        confidence: 0.75,
        items: parts,
        stylingTips: pickStylingTips(layered)
      });
    }
  }
  for (const o of outfits) {
    for (const it of o.items || []) {
      if (typeof it.imageUrl !== "string") it.imageUrl = it.image_url || it.image || "";
      if (typeof it.image !== "string") it.image = it.imageUrl;
      if (!it.macroCategory) it.macroCategory = "other";
    }
  }
  return outfits.filter((outfit) => !fourSlotOutfitReason(outfit.items) && !styleJudgeReason(outfit.items, style, occasion, weather, preferences?.lockedAnchorId, preferences)).map((outfit) => ({ ...outfit, confidence: qualityOf(outfit.items, style, occasion, weather, preferences?.lockedAnchorId, preferences).score }));
}
function safeParseOutfits(raw) {
  try {
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed?.outfits) && parsed.outfits.length > 0) return normalizeModelOutfits(parsed.outfits);
    if (Array.isArray(parsed) && parsed.length > 0) return normalizeModelOutfits(parsed);
  } catch (_) {
  }
  const match = raw.match(/\{[\s\S]*"outfits"\s*:\s*\[[\s\S]*\][\s\S]*\}/);
  if (match) {
    try {
      const parsed = JSON.parse(match[0]);
      if (Array.isArray(parsed.outfits)) return normalizeModelOutfits(parsed.outfits);
    } catch (_) {
    }
  }
  const arrMatch = raw.match(/\[[\s\S]*\]/);
  if (arrMatch) {
    try {
      const arr = JSON.parse(arrMatch[0]);
      if (Array.isArray(arr) && arr.length > 0 && arr[0].items) return normalizeModelOutfits(arr);
    } catch (_) {
    }
  }
  return null;
}
var DEFAULT_NVIDIA_MODEL = "meta/llama-3.2-11b-vision-instruct";
function shuffleArray(array) {
  const arr = [...array];
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}
function resolveNvidiaModel(input) {
  const model = (input || "").trim();
  return model.length > 0 ? model : DEFAULT_NVIDIA_MODEL;
}
async function callNvidia(key, model, sys, usr, retries = 0, trace, layered = false, outputContract) {
  for (let attempt = 0; attempt <= retries; attempt++) {
    const controller = new AbortController();
    let timedOut = false;
    let timer;
    const deadline = new Promise((_, reject) => {
      timer = setTimeout(() => {
        timedOut = true;
        reject(new Error("nvidia_request_timeout"));
        controller.abort();
      }, 45e3);
    });
    const withinDeadline = (operation) => Promise.race([operation, deadline]);
    try {
      console.log(`[NVIDIA] attempt ${attempt + 1}, model: ${model}`);
      const r = await withinDeadline(fetch("https://integrate.api.nvidia.com/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json", "Authorization": `Bearer ${key}` },
        body: JSON.stringify({ model, ...model.includes("nemotron") ? { chat_template_kwargs: { enable_thinking: false } } : {}, messages: [{ role: "system", content: sys + (outputContract || "\nOUTPUT CONTRACT: Return " + JSON.stringify({ outfits: [{ description: "brief explanation", items: (layered ? ["BASE TOP", "OUTERWEAR", "BOTTOM", "SHOES"] : ["TOP", "BOTTOM", "SHOES"]).map((slot) => ({ id: "EXACT " + slot + " CANDIDATE ID" })) }] }) + ". Every item must contain the id field as a string copied exactly from the candidate list. The example lists required garment slots; replace every placeholder with a real candidate ID for that slot. Never replace a required slot with another top. A dress replaces top and bottom: choose dress plus shoes, and add outerwear when layering is required; never put a separate bottom with a dress. Include additional selected pieces only when requested and compatible; never layer two pullover knits. Do not omit required slots, return names instead of IDs, or invent IDs. Item metadata is attached by the server; only IDs are needed in items.") }, { role: "user", content: usr }], temperature: 0.25, max_tokens: outputContract ? 700 : 2e3, frequency_penalty: 0.6, presence_penalty: 0.4, response_format: { type: "json_object" } }),
        signal: controller.signal
      }));
      if (r.status === 429 || r.status >= 500) {
        console.warn(`[NVIDIA] ${r.status}, retrying in 2s...`);
        trace?.push({ provider: "nvidia", model, status: String(r.status) });
        if (attempt < retries) {
          await new Promise((r2) => setTimeout(r2, 2e3));
          continue;
        }
        return null;
      }
      if (!r.ok) {
        console.error("[NVIDIA] error:", r.status);
        trace?.push({ provider: "nvidia", model, status: String(r.status) });
        return null;
      }
      const d = await withinDeadline(r.json());
      const content = d.choices?.[0]?.message?.content || "{}";
      console.log(`[NVIDIA] response length: ${content.length} chars`);
      const outfits = (outputContract ? parseRankedOutfitChoices(content) : null) || safeParseOutfits(content);
      if (outfits && outfits.length > 0) {
        console.log(`[NVIDIA] parsed ${outfits.length} outfits`);
        trace?.push({ provider: "nvidia", model, status: "ok" });
        return outfits;
      }
      console.warn("[NVIDIA] returned unparseable content");
      trace?.push({ provider: "nvidia", model, status: "unparseable" });
      return null;
    } catch (e) {
      console.error("[NVIDIA] call error:", e);
      trace?.push({ provider: "nvidia", model, status: timedOut ? "timeout" : e instanceof SyntaxError ? "invalid_response" : "network_error" });
      if (attempt < retries) {
        await new Promise((r) => setTimeout(r, 1500));
        continue;
      }
      return null;
    } finally {
      clearTimeout(timer);
    }
  }
  return null;
}
function enforceAnchorInOutfits(outfits, anchorDbItem) {
  const anchorMacro = detectSlotMismatch(anchorDbItem.macroCategory || macroCategory(anchorDbItem.category || "", anchorDbItem.type || ""), anchorDbItem.name || anchorDbItem.type || "", anchorDbItem.type || "") || macroCategory(anchorDbItem.category || "", anchorDbItem.type || "");
  const anchorOutfitItem = {
    id: anchorDbItem.id,
    type: anchorDbItem.type || anchorDbItem.category || "clothing",
    macroCategory: anchorMacro,
    color: Array.isArray(anchorDbItem.color) ? anchorDbItem.color.join(", ") : anchorDbItem.primary_color || anchorDbItem.color || "neutral",
    name: anchorDbItem.name || anchorDbItem.type || anchorDbItem.category || "Item",
    imageUrl: anchorDbItem.image_url || anchorDbItem.imageUrl || "",
    image: anchorDbItem.image_url || anchorDbItem.imageUrl || "",
    brand: anchorDbItem.brand || "",
    recommendation: "Anchor item \u2014 user selected",
    isShopItem: false
  };
  return outfits.map((outfit) => {
    const items = Array.isArray(outfit.items) ? [...outfit.items] : [];
    const slotIdx = items.findIndex(
      (i) => macroCategory(i.category || i.macroCategory || "", i.type || "") === anchorMacro || (i.macroCategory || "").toLowerCase() === anchorMacro
    );
    if (slotIdx >= 0) {
      items[slotIdx] = anchorOutfitItem;
    } else {
      items.unshift(anchorOutfitItem);
    }
    return { ...outfit, items };
  });
}
serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }
  if (req.method !== "POST") return new Response(JSON.stringify({ success: false, outfits: [], error: "method_not_allowed" }), { status: 405, headers: { ...corsHeaders, "Content-Type": "application/json" } });
  const rateLimitResponse = await enforceRateLimit(req, { feature: "generate-outfits", limit: 10, headers: corsHeaders });
  if (rateLimitResponse) return rateLimitResponse;
  const SUPABASE_URL = Deno.env.get("SUPABASE_URL");
  const SUPABASE_ANON_KEY = Deno.env.get("SUPABASE_ANON_KEY");
  const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");
  let body;
  try {
    body = await readOutfitGenerationRequest(req);
  } catch (error) {
    return new Response(JSON.stringify({ success: false, outfits: [], error: "invalid_generation_request" }), {
      status: error instanceof OutfitRequestError ? error.status : 400,
      headers: { ...corsHeaders, "Content-Type": "application/json" }
    });
  }
  let {
    prompt = "",
    stylePreferences = "Casual",
    occasion = "Everyday",
    weather,
    limit = 3,
    // Deck size the caller will actually DISPLAY. The Personal Stylist
    // requests a few extra candidates (limit) but shows three cards
    // (deckTarget). Capacity must be judged against the DISPLAYED deck
    // size, otherwise a 5-candidate request would mark a 3-shoe slot
    // "thin" and license repeating shoes on the three visible cards.
    deckTarget,
    selectedItemIds = [],
    wardrobeItems: legacyWardrobe = [],
    useProvidedWardrobeOnly = false,
    catalogOnly = false,
    anchorItemId,
    // Wardrobe-only (Supabase clothing) requests: build ONLY from
    // clothing_items + provided wardrobe items — no shop_catalog
    // supplement, no static fill.
    wardrobeOnly = false,
    // Per-tap salt from the client (generation nonce). The local-fallback
    // deck rotation is otherwise a pure function of style+occasion, so
    // repeat taps rebuild byte-identical decks. Absent (other callers) →
    // random salt: variety by default, determinism only when requested.
    deckSalt = -1,
    // Item ids the user ALREADY SAW in recent decks (rolling window,
    // maintained client-side). Excluded from the candidate pool so
    // regenerate taps bring NEW pieces instead of recombining the same
    // ~40 items forever. Manual selection (selectedItemIds) ignores this.
    excludeItemIds = [],
    excludeProductKeys = [],
    previousDeckProductKeys = [],
    olderRecentProductKeys = [],
    rotationDecks = [],
    previousShoeProductKeys,
    recentShoeProductKeys = [],
    shoeRotationDecks
  } = body;
  if (catalogOnly && anchorItemId && !String(anchorItemId).startsWith("shop_") || wardrobeOnly && String(anchorItemId || "").startsWith("shop_")) return new Response(JSON.stringify({ success: false, outfits: [], code: "no_matching_clothing", error: "The locked item is outside the selected source." }), { status: 422, headers: { ...corsHeaders, "Content-Type": "application/json" } });
  if (catalogOnly) {
    legacyWardrobe = [];
    useProvidedWardrobeOnly = true;
    wardrobeOnly = false;
  }
  const retrievalSeed = body.generationRequestId || (deckSalt >= 0 ? `replay:${deckSalt}` : crypto.randomUUID());
  weather = resolveStylistWeather(prompt, weather);
  const selectionStyle = isWorkoutOccasion(`${occasion} ${prompt}`) ? "sport" : stylePreferences;
  const excludeSet = new Set(
    (Array.isArray(excludeItemIds) ? excludeItemIds : []).slice(0, 200).map((v) => String(v ?? "").trim()).filter(Boolean)
  );
  for (const key of [...excludeProductKeys, ...previousDeckProductKeys, ...olderRecentProductKeys]) excludeSet.add(key);
  for (const key of [...recentShoeProductKeys, ...previousShoeProductKeys || []]) excludeSet.add(key);
  const previousSet = new Set(previousDeckProductKeys);
  const previousShoeSet = new Set(previousShoeProductKeys ?? previousDeckProductKeys);
  const deckPriority = createShoeRotationPriority(
    excludeSet,
    previousSet,
    rotationDecks,
    anchorItemId,
    shoeRotationDecks ? { previous: previousShoeSet, decks: shoeRotationDecks } : void 0
  );
  const rotationTarget = deckTarget === 3 ? 3 : 0;
  let parsedBudget;
  let requestFx;
  const loadRequestFx = async () => {
    if (!requestFx) {
      try {
        const admin = createClient2(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, { auth: { persistSession: false, autoRefreshToken: false } });
        const id = body.preferences?.budgetRatesId;
        const snapshot = id === void 0 ? await currentBudgetRates(admin) : await budgetRatesById(admin, id);
        requestFx = budgetFxContext(snapshot);
      } catch {
        throw new BudgetFxError();
      }
    }
    return requestFx;
  };
  try {
    const selected = strictestOutfitBudget(body.preferences?.maxShopBudget, null, budgetInCurrencyEdge);
    const candidates = budgetCandidatesFromText(`${prompt} ${occasion}`);
    const needsFx = (/* @__PURE__ */ new Set([...candidates.map((b) => b.currency), ...selected ? [selected.currency] : []])).size > 1;
    const hasPinnedBudgetRatesId = body.preferences?.budgetRatesId !== void 0;
    if (hasPinnedBudgetRatesId || needsFx) {
      try {
        await loadRequestFx();
      } catch (fxError) {
        if (!needsFx && !hasPinnedBudgetRatesId) requestFx = void 0;
        else throw fxError;
      }
    }
    parsedBudget = strictestOutfitBudget(
      selected,
      parseBudgetFromText(`${prompt} ${occasion}`, requestFx),
      (budget, currency2) => budgetInCurrencyEdge({ ...budget, fx: requestFx }, currency2)
    );
    if (parsedBudget && requestFx) parsedBudget.fx = requestFx;
  } catch (error) {
    const unavailable = error instanceof BudgetFxError;
    return new Response(JSON.stringify({ success: false, error: unavailable ? "budget_rates_unavailable" : "invalid_outfit_budget", outfits: [] }), {
      status: unavailable ? 503 : 400,
      headers: { ...corsHeaders, "Content-Type": "application/json" }
    });
  }
  const preferences = parsedBudget ? { ...body.preferences || {}, maxShopBudget: parsedBudget, lockedAnchorId: anchorItemId } : { ...body.preferences || {}, lockedAnchorId: anchorItemId };
  try {
    const authHeader = req.headers.get("Authorization") || "";
    let dbClothingItems = [];
    let requestUserId = null;
    const authReply = (status) => new Response(JSON.stringify({
      success: false,
      outfits: [],
      error: status === 401 ? "unauthorized" : "auth_unavailable"
    }), { status, headers: { ...corsHeaders, "Content-Type": "application/json" } });
    if (!/^Bearer\s+\S+$/i.test(authHeader.trim())) return authReply(401);
    const userSupabase = createClient2(SUPABASE_URL, SUPABASE_ANON_KEY, {
      global: { headers: { Authorization: authHeader } }
    });
    let user = null;
    try {
      const result = await userSupabase.auth.getUser();
      if (result.error) {
        const status = Number(result.error.status);
        return authReply(!status || status >= 500 ? 503 : 401);
      }
      user = result.data?.user;
    } catch {
      return authReply(503);
    }
    if (!user?.id) return authReply(401);
    requestUserId = user.id;
    const { data: recentWearLogs, error: wearHistoryError } = await userSupabase.from("wear_logs").select("item_ids").eq("user_id", requestUserId).order("created_at", { ascending: false }).order("id", { ascending: false }).limit(5);
    if (wearHistoryError || !Array.isArray(recentWearLogs)) throw new CatalogLookupError();
    const wornItemIds = new Set(recentWearLogs.flatMap((log) => Array.isArray(log.item_ids) ? log.item_ids.filter((id) => typeof id === "string") : []));
    if (!useProvidedWardrobeOnly) {
      if (user) {
        const rows = [];
        const pageSize = 500;
        for (let offset = 0; ; offset += pageSize) {
          let q = userSupabase.from("clothing_items").select("id, type, category, color, style, brand, season, seasons, occasions, pattern, material, image_url, description, primary_color, sub_category, wear_count, name").eq("user_id", requestUserId).order("id", { ascending: true }).range(offset, offset + pageSize - 1);
          if (selectedItemIds && selectedItemIds.length > 0) q = q.in("id", selectedItemIds);
          const { data, error } = await q;
          if (error || !Array.isArray(data)) throw new CatalogLookupError();
          rows.push(...data);
          if (data.length < pageSize) break;
        }
        if (rows && rows.length > 0) {
          dbClothingItems = rows.map((row) => {
            const occasionsRaw = row.occasions ?? row.occasion;
            const seasonsRaw = row.seasons ?? row.season;
            const fmtList = (v, fallback) => {
              if (Array.isArray(v)) return v.join(", ");
              if (v && typeof v === "object") return Object.keys(v).join(", ");
              if (typeof v === "string" && v.length > 0) return v;
              return fallback;
            };
            return {
              id: row.id,
              type: row.type || row.sub_category || row.category || "clothing",
              category: row.category || "Other",
              macroCategory: macroCategory(row.category || "", row.type || row.sub_category || ""),
              color: Array.isArray(row.color) ? row.color.join(", ") : row.primary_color || row.color || "neutral",
              style: row.style || "Casual",
              brand: row.brand || "",
              name: row.name || row.type || row.sub_category || row.category || "Item",
              material: row.material || "",
              pattern: row.pattern || "",
              season: fmtList(seasonsRaw, "All Seasons"),
              occasion: fmtList(occasionsRaw, ""),
              description: row.description || "",
              imageUrl: row.image_url || "",
              wearCount: row.wear_count || 0
            };
          });
        }
      }
    }
    let wardrobeItems = dbClothingItems.length > 0 && !useProvidedWardrobeOnly ? dbClothingItems : (legacyWardrobe || []).map((item) => ({
      ...item,
      macroCategory: item.macroCategory || macroCategory(item.category || "", item.type || "")
    }));
    if (anchorItemId && useProvidedWardrobeOnly && !wardrobeItems.some((item) => String(item.id) === String(anchorItemId))) {
      const { data: anchor, error: anchorError } = await userSupabase.from("clothing_items").select("id,name,type,sub_category,category,primary_color,brand,description,image_url").eq("id", anchorItemId).eq("user_id", requestUserId).maybeSingle();
      if (anchorError) throw new CatalogLookupError();
      if (!anchor) return new Response(JSON.stringify({ success: false, outfits: [], error: "anchor_not_found" }), { status: 422, headers: { ...corsHeaders, "Content-Type": "application/json" } });
      wardrobeItems.push({
        ...anchor,
        type: anchor.type || anchor.sub_category || anchor.category,
        macroCategory: macroCategory(anchor.category || "", `${anchor.type || ""} ${anchor.sub_category || ""}`, anchor.name || ""),
        color: anchor.primary_color || "neutral",
        imageUrl: anchor.image_url,
        isWardrobe: true,
        isShopItem: false
      });
    }
    wardrobeItems = wardrobeItems.map((item) => {
      const fixedMacro = detectSlotMismatch(item.macroCategory, item.name || item.type || "", item.type || "");
      const withSlot = fixedMacro ? { ...item, macroCategory: fixedMacro } : item;
      const hasTags = Array.isArray(withSlot.styleTags) && withSlot.styleTags.length > 0;
      if (hasTags && withSlot.material !== void 0) return withSlot;
      const inferred = inferAttrsEdge(withSlot);
      return {
        ...withSlot,
        styleTags: hasTags ? withSlot.styleTags : inferred.styleTags,
        material: withSlot.material || inferred.materials,
        pattern: withSlot.pattern || inferred.patterns,
        formality: typeof withSlot.formality === "number" ? withSlot.formality : inferred.formality
      };
    });
    if (preferences && Array.isArray(preferences.avoidColors) && preferences.avoidColors.length > 0) {
      const avoid = preferences.avoidColors.map((c) => c.toLowerCase().trim()).filter(Boolean);
      const beforeAvoid = wardrobeItems.length;
      wardrobeItems = wardrobeItems.filter((item) => {
        const color = (item.color || item.primary_color || "").toLowerCase();
        const name = (item.name || "").toLowerCase();
        const desc = (item.description || "").toLowerCase();
        return !avoid.some((c) => color.includes(c) || name.includes(` ${c} `) || desc.includes(` ${c} `));
      });
      console.log(`[preferences] Filtered out avoided colors: ${beforeAvoid} -> ${wardrobeItems.length} items`);
    }
    wardrobeItems = wardrobeItems.filter((it) => rotationTarget || String(it.id) === String(anchorItemId || "") || (selectedItemIds || []).includes(String(it.id)) || !excludeSet.has(String(it.id)));
    const qualityContext = stylingContext({ style: stylePreferences, occasion, request: prompt, weather, anchorId: anchorItemId, preferences, sourceMode: wardrobeOnly ? "wardrobe" : catalogOnly ? "catalog" : "mixed" });
    const memoQuality = createQualityMemo();
    const lookSignals = /* @__PURE__ */ new WeakMap();
    const signalsFor = (look) => {
      let signals = lookSignals.get(look);
      if (!signals) {
        const recency = recentLookSimilarity(look.items, body.recentLookSignatures ?? []);
        const penalty = recency >= 0.8 ? (recency - 0.8) * 0.15 : 0;
        signals = { recency, quality: Math.max(0, memoQuality(look.items, styleKey, `${occasion} ${prompt}`, weather, anchorItemId, preferences).score - penalty) };
        lookSignals.set(look, signals);
      }
      return signals;
    };
    const eligibleCandidate = (item) => !garmentContextReason(item, qualityContext) && !wornItemIds.has(String(item.id)) && !wornItemIds.has(`shop_${item.id}`) && isFourSlotCandidate(item) && !isNonOutfitItem(item) && !rejectForStyleOccasion(item, String(item.id) === String(anchorItemId || "") ? "casual" : selectionStyle, `${occasion} ${prompt}`) && !rejectForClimate(item, weather) && !rejectForRequestIntent(item, `${occasion} ${prompt}`, weather) && !rejectForActivity(item, `${occasion} ${prompt}`);
    const requiredIds = /* @__PURE__ */ new Set([...(selectedItemIds || []).map(String), ...anchorItemId ? [String(anchorItemId)] : []]);
    const blockedSelectedItemIds = wardrobeItems.filter((item) => requiredIds.has(String(item.id)) && !eligibleCandidate(item)).map((item) => String(item.id));
    if (blockedSelectedItemIds.length) return new Response(JSON.stringify({
      success: false,
      outfits: [],
      code: "no_matching_clothing",
      error: "A selected or locked item is incompatible with the requested style, occasion or weather.",
      blockedSelectedItemIds
    }), {
      status: 422,
      headers: { ...corsHeaders, "Content-Type": "application/json" }
    });
    if (isWorkoutOccasion(`${occasion} ${prompt}`)) {
      const blockedAnchor = wardrobeItems.some((item) => String(item.id) === String(anchorItemId || "") && rejectForActivity(item, `${occasion} ${prompt}`));
      if (blockedAnchor) return new Response(JSON.stringify({ success: false, outfits: [], code: "no_matching_clothing", error: "The locked item is not suitable for this activity." }), {
        status: 422,
        headers: { ...corsHeaders, "Content-Type": "application/json" }
      });
      wardrobeItems = wardrobeItems.filter((item) => !rejectForActivity(item, `${occasion} ${prompt}`));
    }
    if (rotationTarget) wardrobeItems = rotationPool(wardrobeItems.filter(eligibleCandidate), excludeSet, rotationTarget, requiredIds).items;
    if (!wardrobeOnly) {
      let requiredSlots = ["top", "bottom", "shoes"];
      if (needsLayering2(stylePreferences, weather, prompt)) requiredSlots.push("outerwear");
      const lockedSlot = wardrobeItems.find((item) => String(item.id) === String(anchorItemId || ""))?.macroCategory;
      if (lockedSlot) requiredSlots = requiredSlots.filter((slot) => slot !== lockedSlot);
      const shopOnlyRequest = isShopOnlyDeck(stylePreferences, occasion);
      const { slots: missingSlots, targetChoicesPerSlot } = shopOnlyRequest ? { slots: requiredSlots, targetChoicesPerSlot: limit >= 10 ? 16 : 8 } : requiredShopSupplementSlots(
        wardrobeItems,
        requiredSlots,
        limit,
        selectedItemIds,
        (item) => eligibleCandidate(item) && (!rotationTarget || !wasRecentlyShown(item, excludeSet)) && (!parsedBudget || !isShopPricedItem(item))
      );
      try {
        const svcClient = createClient2(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);
        if (parsedBudget && missingSlots.length > 0) {
          try {
            parsedBudget.fx = await loadRequestFx();
          } catch (error) {
            if (body.preferences?.budgetRatesId || new Set(budgetCandidatesFromText(`${prompt} ${occasion}`).map((b) => b.currency)).size > 1) throw error;
            console.warn("[budget-rates] using_native_currency_only", { currency: parsedBudget.currency });
          }
          preferences.maxShopBudget = parsedBudget;
        }
        const lockedItem = wardrobeItems.find((item) => String(item.id) === String(anchorItemId || ""));
        const shopEligible = (item, discovered = []) => {
          if (!eligibleCandidate(item) || filterShopByPrefs([item], preferences).length !== 1 || lockedItem && rejectOutfitIncoherence([lockedItem, item], selectionStyle, `${occasion} ${prompt}`)) return false;
          const layers = [...wardrobeItems, ...discovered].filter((layer) => layer.macroCategory === "outerwear");
          return !lockedItem || item.macroCategory !== "top" || !layers.length || layers.some((layer) => !rejectOutfitIncoherence([...lockedItem.macroCategory === "outerwear" ? [] : [lockedItem], layer, item], selectionStyle, `${occasion} ${prompt}`));
        };
        const discoverySlots = lockedItem ? [...missingSlots].sort((a, b) => Number(b === "outerwear") - Number(a === "outerwear")) : missingSlots;
        const fetchedShopItems = await fetchShopItemsForAllSlotsEdge(
          svcClient,
          discoverySlots,
          selectionStyle,
          targetChoicesPerSlot,
          excludeSet,
          shopEligible,
          parsedBudget ? requestFx : void 0,
          parsedBudget && !requestFx ? parsedBudget : void 0,
          `${occasion} ${prompt}`,
          rotationTarget,
          deckPriority,
          lockedItem ? (layer, top) => !rejectOutfitIncoherence([...lockedItem.macroCategory === "outerwear" ? [] : [lockedItem], layer, top], selectionStyle, `${occasion} ${prompt}`) : void 0,
          retrievalSeed
        );
        const shopItems = filterShopByPrefs(fetchedShopItems, preferences);
        if (String(selectionStyle).toLowerCase().replace(/[ -]+/g, "_") === "old_money" && /\bdate\b/i.test(`${occasion} ${prompt}`)) {
          const freshLayers = [...wardrobeItems, ...shopItems].filter((item) => item.macroCategory === "outerwear" && !wasRecentlyShown(item, excludeSet));
          const families = new Set(freshLayers.map(garmentArchetypeToken));
          if (freshLayers.length >= 3 && families.size === 1) {
            const alternatives = await fetchShopItemsForAllSlotsEdge(
              svcClient,
              ["outerwear"],
              selectionStyle,
              3,
              excludeSet,
              (item) => shopEligible(item) && !families.has(garmentArchetypeToken(item)),
              parsedBudget ? requestFx : void 0,
              parsedBudget && !requestFx ? parsedBudget : void 0,
              `${occasion} ${prompt}`,
              rotationTarget,
              deckPriority,
              void 0,
              retrievalSeed
            );
            shopItems.push(...filterShopByPrefs(alternatives, preferences));
          }
          const layers = spreadQualityFamilies(shopItems.filter((item) => item.macroCategory === "outerwear"), garmentArchetypeToken);
          shopItems.splice(0, shopItems.length, ...layers, ...shopItems.filter((item) => item.macroCategory !== "outerwear"));
        }
        if (shopItems.length > 0) {
          console.log(`[shop-fill] Added ${shopItems.length} shop item(s) for sparse slots: ${missingSlots.join(", ")} (target=${targetChoicesPerSlot})`);
          wardrobeItems = [...wardrobeItems, ...shopItems];
        }
      } catch (error) {
        if (error instanceof CatalogLookupError || error instanceof BudgetFxError) throw error;
        console.warn("[shop-fill] Failed to fetch shop catalog items");
      }
    } else {
      console.log(`[wardrobeOnly] Skipping shop supplement \u2014 ${wardrobeItems.length} Supabase wardrobe item(s) only`);
    }
    const fetchedShoeIds = wardrobeItems.filter((item) => item.macroCategory === "shoes").map((item) => item.id);
    const rotation = rotationTarget ? rotationPool(wardrobeItems.filter(eligibleCandidate), excludeSet, rotationTarget, requiredIds) : null;
    if (rotation) wardrobeItems = rotation.items;
    if (isShopOnlyDeck(stylePreferences, occasion) && !anchorItemId) {
      const beforeShopOnly = wardrobeItems.length;
      wardrobeItems = wardrobeItems.filter((item) => isShopRow(item));
      console.log(`[shop-only] Pool reduced to catalog-only: ${beforeShopOnly} -> ${wardrobeItems.length} (owned rows dropped)`);
    }
    if (isWorkoutOccasion(`${occasion} ${prompt}`)) {
      wardrobeItems = wardrobeItems.filter((item) => !rejectForActivity(item, `${occasion} ${prompt}`));
    }
    if (parsedBudget && wardrobeItems.some((item) => isShopPricedItem(item) && Number.isFinite(Number(item.price)) && Number(item.price) > 0 && ["USD", "EUR", "GBP", "UZS"].includes(String(item.currency || "").trim().toUpperCase()) && String(item.currency || "").trim().toUpperCase() !== parsedBudget.currency)) {
      try {
        await loadRequestFx();
      } catch (error) {
        if (body.preferences?.budgetRatesId) throw error;
        wardrobeItems = wardrobeItems.filter((item) => !isShopPricedItem(item) || String(item.currency || "").toUpperCase() === parsedBudget.currency);
        console.warn("[budget-rates] using_native_currency_only", { currency: parsedBudget.currency });
      }
      parsedBudget.fx = requestFx;
      preferences.maxShopBudget = parsedBudget;
    }
    if (parsedBudget && !wardrobeItems.some(isDressGarment)) {
      const layeredBudget = needsLayering2(stylePreferences, weather, prompt);
      const requiredSlots = ["top", "bottom", "shoes", ...layeredBudget ? ["outerwear"] : []];
      const cost = (item) => {
        if (!isShopPricedItem(item)) return 0;
        const value = shopPriceOf(item, parsedBudget.currency, parsedBudget.fx);
        return Number.isFinite(value) ? value : null;
      };
      const minimum = minimumSlotSpend(wardrobeItems, requiredSlots, cost);
      const anyAffordable = wardrobeItems.some((item) => {
        const value = cost(item);
        return value !== null && fitsBudgetAmount(value, parsedBudget.amount);
      });
      if (anyAffordable && minimum !== null && !fitsBudgetAmount(minimum, parsedBudget.amount)) {
        return new Response(
          JSON.stringify({
            success: false,
            outfits: [],
            code: "no_outfit_within_budget",
            error: "No complete outfit in the selected clothing pool fits this budget.",
            budget: { amount: parsedBudget.amount, currency: parsedBudget.currency },
            minimumShopSpend: minimum
          }),
          { status: 422, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }
    }
    let budgetFilteredAll = false;
    const unverifiedBudgetItems = [];
    if (parsedBudget) {
      const beforeBudget = wardrobeItems.length;
      wardrobeItems = wardrobeItems.filter((item) => {
        if (!isShopPricedItem(item)) return true;
        const p = shopPriceOf(item, parsedBudget.currency, parsedBudget.fx);
        if (!Number.isFinite(p)) {
          unverifiedBudgetItems.push(item);
          return false;
        }
        if (p <= 0) return true;
        return fitsBudgetAmount(p, parsedBudget.amount);
      });
      budgetFilteredAll = beforeBudget > 0 && wardrobeItems.length === 0;
      console.log(`[budget] shop items over ${parsedBudget?.amount} ${parsedBudget?.currency} dropped: ${beforeBudget} -> ${wardrobeItems.length}`);
    }
    const styleKey = (selectionStyle || "casual").toLowerCase().replace(/[\s-]+/g, "_");
    const anchorPreRank = anchorItemId ? wardrobeItems.find((i) => String(i.id) === String(anchorItemId)) : void 0;
    if (!selectedItemIds || selectedItemIds.length === 0) {
      const before = wardrobeItems.length;
      const occKey = String(occasion || "").toLowerCase();
      const formalOcc = /\b(work|office|date|formal|wedding)\b/.test(occKey);
      const classicStyle = /old_money|classic|business/.test(styleKey);
      const tasteEligible = (it) => {
        if (!(formalOcc || classicStyle)) return true;
        const blob = `${it.name || ""} ${it.type || ""} ${it.description || ""}`.toLowerCase();
        if (isDriverShoeItem(it)) return false;
        if (/\b(washed[\s-]?effect|bell[\s-]?bottom|splatter|uneven[\s-]?hem|twisted[\s-]?seams?)\b/.test(blob)) return false;
        if (formalOcc && !/\bdate\b/.test(occKey) && /\bpiqu/.test(blob) && /\bpolo/.test(blob)) return false;
        return true;
      };
      if (rotationTarget) wardrobeItems.sort((a, b) => compareRotationOrder(deckPriority([b]), deckPriority([a])));
      const budgetSource = wardrobeItems;
      const rankEligible = (item) => eligibleCandidate(item) && (String(item.id) === String(anchorItemId || "") || tasteEligible(item));
      wardrobeItems = rankItemsForStyleEdge(wardrobeItems, styleKey, { minKeep: 16, dropThreshold: -2, perCategoryFloor: 6, eligibility: rankEligible });
      const compareRankedCandidates = (a, b) => {
        const score = (item) => 0.6 * scoreGarmentForContext(item, qualityContext) + 0.4 * scoreItemForStyle(item, normalizeStyleId(styleKey));
        return (rotationTarget ? compareRotationOrder(deckPriority([b]), deckPriority([a])) : 0) || score(b) - score(a) || (!parsedBudget ? compareUnbudgetedValue(a, b) : 0);
      };
      wardrobeItems.sort(compareRankedCandidates);
      wardrobeItems = collapseBySimEdge(wardrobeItems, (i) => canonicalProductKey(i));
      if (parsedBudget) {
        wardrobeItems = retainBudgetFloorCandidates(
          budgetSource,
          wardrobeItems,
          (item) => shopPriceOf(item, parsedBudget.currency, parsedBudget.fx),
          rankEligible,
          anchorPreRank,
          10
        );
      } else {
        wardrobeItems = retainBudgetFloorCandidates([...budgetSource].sort(compareRankedCandidates), wardrobeItems, () => 0, rankEligible, anchorPreRank, 10);
      }
      wardrobeItems = wardrobeItems.slice(0, 40);
      if (anchorPreRank && !wardrobeItems.some((i) => String(i.id) === String(anchorItemId))) {
        console.log(`[anchor] Re-adding locked item ${anchorItemId} after rank trim`);
        wardrobeItems = [anchorPreRank, ...wardrobeItems].slice(0, 40);
      }
      console.log(`[rank] ${before} -> ${wardrobeItems.length} items after style filter for ${styleKey}`);
    } else {
      wardrobeItems = wardrobeItems.filter(eligibleCandidate);
    }
    console.log("[generation-pool]", { count: wardrobeItems.length, style: selectionStyle, slots: Object.fromEntries(["top", "bottom", "shoes", "outerwear"].map((slot) => [slot, wardrobeItems.filter((item) => item.macroCategory === slot).length])) });
    if (parsedBudget && unverifiedBudgetItems.length > 0) {
      const layeredBudget = needsLayering2(stylePreferences, weather, prompt);
      const requiredBudgetSlots = ["top", "bottom", "shoes", ...layeredBudget ? ["outerwear"] : []];
      const hasBudgetSlot = (slot) => wardrobeItems.some((item) => String(item.macroCategory || "").toLowerCase() === slot || slot === "top" && !layeredBudget && String(item.macroCategory || "").toLowerCase() === "outerwear");
      const unverifiedRequiredSlot = !wardrobeItems.some(isDressGarment) && requiredBudgetSlots.some((slot) => !hasBudgetSlot(slot) && unverifiedBudgetItems.some((item) => String(item.macroCategory || "").toLowerCase() === slot));
      const unverifiedAnchor = anchorItemId && unverifiedBudgetItems.some((item) => String(item.id) === String(anchorItemId));
      if (wardrobeItems.length === 0 || unverifiedRequiredSlot || unverifiedAnchor) {
        return new Response(JSON.stringify({
          success: false,
          outfits: [],
          code: "unverified_shop_pricing",
          error: "A required shop price or currency is unavailable. Choose items with verified prices.",
          budget: parsedBudget ? { amount: parsedBudget.amount, currency: parsedBudget.currency } : null
        }), {
          status: 422,
          headers: { ...corsHeaders, "Content-Type": "application/json" }
        });
      }
    }
    if (wardrobeItems.length > 0) {
      const required = [...wardrobeItems.some(isDressGarment) ? [] : ["top", "bottom"], "shoes", ...needsLayering2(selectionStyle, weather, prompt) ? ["outerwear"] : []];
      const missingSlots = required.filter((slot) => !wardrobeItems.some((item) => item.macroCategory === slot));
      if (missingSlots.length) console.warn("[generation-reject]", { code: "no_matching_clothing", missingSlots, wardrobeOnly, useProvidedWardrobeOnly });
      if (missingSlots.length) return new Response(JSON.stringify({
        success: false,
        outfits: [],
        code: "no_matching_clothing",
        error: "The selected clothing pool has no complete outfit suitable for the requested style, occasion and weather.",
        missingSlots
      }), {
        status: 422,
        headers: { ...corsHeaders, "Content-Type": "application/json" }
      });
    }
    if (wardrobeItems.length === 0) {
      return new Response(JSON.stringify({
        success: false,
        budget: parsedBudget ? { amount: parsedBudget.amount, currency: parsedBudget.currency } : null,
        outfits: [],
        code: budgetFilteredAll ? "no_items_within_budget" : "no_matching_clothing",
        error: budgetFilteredAll ? "No clothing items fit this budget. Increase the budget or use owned clothing." : "No clothing items match your selections. Add suitable items or adjust your preferences.",
        debug: {
          dbClothingItems: dbClothingItems.length,
          legacyReceived: Array.isArray(legacyWardrobe) ? legacyWardrobe.length : 0,
          useProvidedWardrobeOnly: !!useProvidedWardrobeOnly,
          wardrobeOnly: !!wardrobeOnly
        }
      }), {
        status: 422,
        headers: { ...corsHeaders, "Content-Type": "application/json" }
      });
    }
    const budgetEligibleItems = wardrobeItems.filter((item) => !rejectForStyleOccasion(item, String(item.id) === String(anchorItemId || "") ? "casual" : selectionStyle, `${occasion} ${prompt}`) && !rejectForClimate(item, weather) && !rejectForActivity(item, `${occasion} ${prompt}`));
    if (parsedBudget && !budgetEligibleItems.some(isDressGarment)) {
      const required = ["top", "bottom", "shoes"];
      if (needsLayering2(stylePreferences, weather, prompt)) required.push("outerwear");
      const minimumShopSpend = minimumSlotSpend(budgetEligibleItems, required, (item) => {
        if (!isShopPricedItem(item)) return 0;
        const cost = shopPriceOf(item, parsedBudget.currency, parsedBudget.fx);
        return Number.isFinite(cost) && typeof item.price === "number" && item.price > 0 ? cost : null;
      });
      if (minimumShopSpend !== null && !fitsBudgetAmount(minimumShopSpend, parsedBudget.amount)) {
        return new Response(JSON.stringify({
          success: false,
          outfits: [],
          code: "no_outfit_within_budget",
          error: "No complete outfit in the selected clothing pool fits this budget.",
          budget: parsedBudget ? { amount: parsedBudget.amount, currency: parsedBudget.currency } : null,
          minimumShopSpend
        }), {
          status: 422,
          headers: { ...corsHeaders, "Content-Type": "application/json" }
        });
      }
    }
    let nvidiaKey = Deno.env.get("NVIDIA_API_KEY") || "";
    let nvidiaModel = resolveNvidiaModel(Deno.env.get("NVIDIA_TEXT_MODEL") || Deno.env.get("NVIDIA_MODEL"));
    if (SUPABASE_SERVICE_ROLE_KEY && !nvidiaKey) {
      const svc = createClient2(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);
      const { data: cfg } = await svc.from("app_config").select("key, value").in("key", ["nvidia_token", "gemma", "nvidia_model", "nvidia_text_model"]);
      if (cfg) {
        for (const row of cfg) {
          if ((row.key === "nvidia_token" || row.key === "gemma") && !nvidiaKey) nvidiaKey = row.value;
          if ((row.key === "nvidia_model" || row.key === "nvidia_text_model") && (!nvidiaModel || nvidiaModel === DEFAULT_NVIDIA_MODEL)) {
            nvidiaModel = resolveNvidiaModel(row.value);
          }
        }
      }
    }
    let paidTier = false;
    if (requestUserId && SUPABASE_SERVICE_ROLE_KEY) {
      try {
        const svc = createClient2(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);
        const { data: subs } = await svc.from("subscriptions").select("tier, status").eq("user_id", requestUserId).order("created_at", { ascending: false }).limit(5);
        if (subs) {
          paidTier = subs.some(
            (s) => String(s.tier || "").toLowerCase() !== "free" && ["active", "trial"].includes(String(s.status || "").toLowerCase())
          );
        }
        console.log(`[tier] user=${requestUserId.slice(0, 8)} paid=${paidTier}`);
      } catch (_) {
      }
    }
    nvidiaModel = Deno.env.get("NVIDIA_OUTFIT_MODEL") || nvidiaModel;
    const useCuratedChoices = Deno.env.get("NVIDIA_OUTFIT_SELECTION_MODE") === "curated";
    if (anchorItemId && (!isShopOnlyDeck(stylePreferences, occasion) || !!anchorItemId) && !wardrobeItems.some((i) => String(i.id) === String(anchorItemId))) {
      return new Response(JSON.stringify({ success: false, error: "anchor_not_found", outfits: [] }), { status: 422, headers: { ...corsHeaders, "Content-Type": "application/json" } });
    }
    const layered = needsLayering2(stylePreferences, weather, `${occasion} ${prompt}`);
    console.log(`[layer] style=${styleKey} weather=${weather?.temp ?? "?"}\xB0C layered=${layered}`);
    if (!nvidiaKey) throw new Error("nvidia_provider_not_configured");
    const sys = buildSystemPrompt(limit, selectionStyle, layered, preferences, occasion, parsedBudget);
    let usr = buildUserPrompt(wardrobeItems, prompt, selectionStyle, occasion, weather, limit, preferences);
    if (body.conversationContext) usr += `
Conversation context (the latest request above takes precedence):
${String(body.conversationContext).slice(-4e3)}`;
    if (anchorItemId && (!isShopOnlyDeck(stylePreferences, occasion) || !!anchorItemId)) {
      const anchorForPrompt = wardrobeItems.find((i) => String(i.id) === String(anchorItemId));
      if (anchorForPrompt) {
        const anchorDesc = [anchorForPrompt.color, anchorForPrompt.name || anchorForPrompt.type].filter(Boolean).join(" ");
        const anchorHint = `ANCHOR ITEM: Every outfit MUST be built around wardrobe item id="${anchorItemId}" (${anchorDesc || "the locked piece"}, slot=${anchorForPrompt.macroCategory || "unknown"}). Include this exact id in every outfit and pick complementary items that match its color, style, and material.`;
        usr = `${anchorHint}

${usr}`;
      }
    }
    const itemMap = new Map(wardrobeItems.map((i) => [String(i.id), i]));
    let verifiedChoiceFallback = [];
    const requiredChoiceIds = [.../* @__PURE__ */ new Set([...selectedItemIds || [], ...anchorItemId && (!isShopOnlyDeck(stylePreferences, occasion) || !!anchorItemId) ? [anchorItemId] : []])].map(String);
    const choiceRejections = /* @__PURE__ */ new Map();
    const rejectChoice = (reason) => {
      const key = reason.slice(0, 160);
      if (choiceRejections.has(key) || choiceRejections.size < 12) choiceRejections.set(key, (choiceRejections.get(key) || 0) + 1);
      return false;
    };
    const hasOwnedCandidates = wardrobeItems.some((item) => !isShopRow(item));
    const choiceVerdicts = /* @__PURE__ */ new Map();
    const rowTokens = /* @__PURE__ */ new WeakMap();
    let nextRowToken = 0;
    const acceptsCompleteChoice = (parts) => {
      const key = parts.map((item) => {
        if (!rowTokens.has(item)) rowTokens.set(item, ++nextRowToken);
        return rowTokens.get(item);
      }).join(":");
      if (choiceVerdicts.has(key)) return choiceVerdicts.get(key);
      const verdict = checkCompleteChoice(parts);
      choiceVerdicts.set(key, verdict);
      return verdict;
    };
    const checkCompleteChoice = (parts) => {
      if (parts.some((item) => String(item.id) !== String(anchorItemId || "") && wasRecentlyShown(item, excludeSet) && !(selectedItemIds || []).includes(String(item.id)) && !rotation?.recycledSlots.includes(diversitySlotOf(item)))) return rejectChoice("recently shown item");
      if (parts.some((item) => wornItemIds.has(String(item.id)))) return rejectChoice("recently worn item");
      if (preferences?.maxShopBudget && !fitsBudgetAmount(outfitShopSpend(parts, itemMap, preferences.maxShopBudget.currency, preferences.maxShopBudget.fx), preferences.maxShopBudget.amount)) return false;
      const shopOnly = isShopOnlyDeck(stylePreferences, occasion) && !anchorItemId;
      if (shopOnly && parts.some((item) => !isShopRow(item))) return false;
      if (!shopOnly && hasOwnedCandidates && parts.every((item) => isShopRow(item))) return false;
      const reason = fourSlotOutfitReason(parts, layered) || styleJudgeReason(parts, styleKey, `${occasion} ${prompt}`, weather, anchorItemId, preferences);
      if (reason) return rejectChoice(reason);
      const proposal = { items: parts };
      const styleVerdict = validateOutfitAgainstStyle(proposal, styleKey, itemMap, layered, preferences), compatibility = validateOutfitCompatibility(proposal, itemMap, styleKey);
      if (!styleVerdict.ok || !compatibility.ok) return rejectChoice(styleVerdict.reason || compatibility.reason || "incompatible");
      if (isWorkishOccasion(`${occasion} ${prompt}`) && parts.some((item) => isSwimBottomItem(item) || isDriverShoeItem(item))) return false;
      return true;
    };
    let completeChoiceCache;
    const completeChoices = () => completeChoiceCache ??= validatedQuartets(wardrobeItems, stylingContext({ style: styleKey, occasion, request: prompt, weather, anchorId: anchorItemId, preferences, sourceMode: catalogOnly ? "catalog" : wardrobeOnly ? "wardrobe" : "mixed", budget: preferences?.maxShopBudget }), acceptsCompleteChoice, { requiredIds: requiredChoiceIds, maximum: 24, searchLimit: 384, perSlot: 12, reserveMultiplier: 1 });
    async function generateCuratedChoices() {
      let choices = completeChoices();
      if (!choices.length) return null;
      const rankCompleteSet = Math.min(Number(deckTarget) || limit, limit) > 1;
      if (rankCompleteSet) {
        const anchor = anchorItemId && (!isShopOnlyDeck(stylePreferences, occasion) || !!anchorItemId) ? wardrobeItems.find((item) => String(item.id) === String(anchorItemId)) : void 0;
        choices = assembleDiverseDeck({
          candidates: choices,
          itemsOf: (choice) => choice.items,
          qualityOf: (choice) => signalsFor(choice).quality,
          recencyOf: (choice) => signalsFor(choice).recency,
          keysOf: (item) => productIdentityKeys(item),
          poolItems: wardrobeItems,
          usedKeys: /* @__PURE__ */ new Set(),
          anchorKeys: anchor ? new Set(productIdentityKeys(anchor)) : void 0,
          anchorSlot: anchor ? diversitySlotOf(anchor) : null,
          activeSlots: layered ? ["top", "outerwear", "bottom", "shoes"] : ["top", "bottom", "shoes"],
          target: Math.min(Number(deckTarget) || limit, limit),
          priorityOfDeck: (deck) => deckPriority(deck.flatMap((choice) => choice.items)),
          maxSearchSteps: 512
        }).deck;
      }
      if (!choices.length) return null;
      verifiedChoiceFallback = choices.map((choice, index) => ({ items: choice.items, curationRank: index, description: `${selectionStyle} outfit`, stylingTips: [] }));
      const usedIds = new Set(choices.flatMap((choice) => choice.items.map((item) => item.id)));
      const context = getStyleContext(selectionStyle);
      const selectionCount = rankCompleteSet ? choices.length : Math.min(limit, choices.length);
      const selectionTask = rankCompleteSet ? `Rank all ${selectionCount} supplied complete look IDs from best to least preferred. Return IDs only, without descriptions or explanations.` : `Choose exactly ${selectionCount} of the supplied complete look IDs. Provide a short description per selected look.`;
      const choiceSystem = `You are a fashion stylist curating complete outfits for ${selectionStyle}. ${context.vibe}
${selectionTask} Judge aesthetic quality, color harmony, occasion and the user's preferences. The options have already passed structural, style and budget validation. Never invent IDs or change the items inside a look. Select distinct look IDs. Return only JSON.`;
      console.log(`[curation] options=${choices.length} required=${selectionCount}`);
      const choiceUser = JSON.stringify({
        requiredCount: selectionCount,
        request: prompt,
        occasion,
        weather,
        preferences: { favoriteColors: preferences?.favoriteColors, fitPreference: preferences?.fitPreference, styleGoals: preferences?.styleGoals },
        products: wardrobeItems.filter((item) => usedIds.has(item.id)).map((item) => ({ id: item.id, name: item.name, slot: item.macroCategory, color: item.color, material: item.material, pattern: item.pattern })),
        looks: choices.map((choice) => ({ id: choice.id, items: choice.items.map((item) => item.id) }))
      });
      const choiceContract = rankCompleteSet ? '\nOUTPUT CONTRACT: {"rankedLookIds":["look001","look002"]}. Return every supplied look ID once, best first, in this flat string array. No outfit objects, descriptions, explanations, individual garment IDs or extra fields.' : '\nOUTPUT CONTRACT: {"outfits":[{"description":"brief styling explanation","items":[{"id":"look001"}]}]}. Each outfit selects exactly ONE supplied complete look ID. Do not return individual garment IDs.';
      providerCalls += 1;
      const selections = await callNvidia(nvidiaKey, nvidiaModel, choiceSystem, choiceUser, 1, providerTrace, false, choiceContract);
      const ranked = resolveOutfitChoices(selections, choices);
      if (rankCompleteSet && ranked?.length !== choices.length) return null;
      return rankCompleteSet && ranked ? ranked.map((outfit, index) => ({ ...outfit, curationRank: index })) : ranked;
    }
    const providerTrace = [];
    const aiProviderUsed = "nvidia:" + nvidiaModel;
    let generationSource = "ai";
    let providerCalls = 0;
    const generate = (system, user2) => {
      providerCalls += 1;
      return callNvidia(nvidiaKey, nvidiaModel, system, user2, 1, providerTrace, layered);
    };
    let aiOutfits = useCuratedChoices ? await generateCuratedChoices() : await generate(sys, usr);
    const lastAttempt = providerTrace[providerTrace.length - 1];
    if (!useCuratedChoices && !aiOutfits && nvidiaKey && lastAttempt?.provider === "nvidia" && lastAttempt.status === "unparseable" && providerCalls < 2) {
      const formatUsr = `${usr}

FORMAT CORRECTION: Return only a JSON object with an outfits array. Each outfit must have an items array of objects with an id string copied exactly from the candidate list. Omit descriptions and explanations. No markdown, prose, alternate keys or invented IDs. Preserve every requested slot, style, occasion, selected item and budget constraint.`;
      providerCalls += 1;
      aiOutfits = await callNvidia(nvidiaKey, nvidiaModel, sys, formatUsr, 0, providerTrace, layered);
    }
    let validatedOutfits = null;
    if (aiOutfits && aiOutfits.length > 0) {
      const validationFailures = [];
      const valid = filterValidOutfits(aiOutfits, styleKey, itemMap, layered, preferences, `${occasion} ${prompt}`, weather, (outfit, reason) => {
        if (validationFailures.length < 20) validationFailures.push({ outfit, reason });
      });
      const passRate = valid.length / aiOutfits.length;
      console.log(`[validate] ${valid.length}/${aiOutfits.length} outfits passed style check for ${styleKey} (layered=${layered})`);
      if (valid.length >= Math.max(1, Math.min(limit, aiOutfits.length) - 1)) {
        validatedOutfits = valid;
      } else if (!useCuratedChoices && passRate < 0.5 && nvidiaKey && providerCalls < 2) {
        console.log(`[validate] pass rate ${passRate.toFixed(2)} too low \u2014 retrying with correction`);
        const layeredReqs = layered ? `  1. EVERY outfit MUST be a FOUR-slot layered look with ALL of these macroCategories present: "top" (base layer \u2014 t-shirt, shirt, polo, blouse), "outerwear" (main top \u2014 sweater, blazer, cardigan, hoodie, jacket), "bottom" (pants/trousers/jeans/shorts/skirt), "shoes". A 3-item outfit is INVALID.
  2. The base top must sit UNDER the outerwear and color-coordinate with it. Think: white oxford shirt under a navy blazer, or cream tee under a camel cardigan.
  3. Do NOT return two base tops or two outerwear pieces \u2014 exactly one of each.` : `  1. EVERY outfit MUST include an item with macroCategory=shoes. No outfit without shoes is acceptable.
  2. EVERY outfit MUST include a top (or outerwear) AND a bottom.`;
        let correctionUsr = isWorkoutOccasion(`${occasion} ${prompt}`) ? `${usr}
Correction: choose verified athletic clothing for every required slot. No dress shirts, blazers, loafers or suede shoes. Return only complete outfits meeting the activity and budget requirements.` : `${usr}

IMPORTANT CORRECTION: Your previous attempt failed because (a) some items violated the ${styleKey} aesthetic, OR (b) outfits were missing required slots. REQUIREMENTS:
${layeredReqs}
  ${layered ? "4" : "3"}. Reject any item whose name/description contains: hoodie (unless casual), graphic tee, logo tee, cargo, ripped, sweatpants, neon, sequin, rhinestone, chunky sneaker, basketball, skate sneaker, casual t-shirt (for old_money / business_casual / minimalist / semi_classic).
  ${layered ? "5" : "4"}. For ${styleKey} shoes, prefer: loafers, penny loafers, bit loafers, dress shoes, boat shoes, derbies, leather shoes. Avoid ALL sneakers if style is old_money.
Try again and return ${limit} outfits that each have ${layered ? "outerwear+base-top+bottom+shoes" : "top+bottom+shoes"}.`;
        correctionUsr += `
AUTHORITATIVE VALIDATION FAILURES (JSON data): ${JSON.stringify(validationFailures)}. Fix these specific failures while preserving every category, occasion, style and budget requirement. Do not repeat the rejected combination.`;
        correctionUsr += outfitBudgetFeedback(
          aiOutfits,
          itemMap,
          preferences?.maxShopBudget,
          (amount2, from, to) => convertShopAmountEdge(amount2, from, to, preferences?.maxShopBudget?.fx),
          (item) => !rejectForStyleOccasion(item, String(item.id) === String(anchorItemId || "") ? "casual" : selectionStyle, `${occasion} ${prompt}`) && !rejectForClimate(item, weather) && !rejectForActivity(item, `${occasion} ${prompt}`)
        );
        const retry = await generate(sys, correctionUsr);
        if (retry && retry.length > 0) {
          const retryValid = filterValidOutfits(retry, styleKey, itemMap, layered, preferences, `${occasion} ${prompt}`, weather);
          validatedOutfits = retryValid.length > valid.length ? retryValid : valid.length > 0 ? valid : null;
        } else if (valid.length > 0) {
          validatedOutfits = valid;
        }
      } else if (valid.length > 0) {
        validatedOutfits = valid;
      }
      if (validatedOutfits && validatedOutfits.length > 0) {
        aiOutfits = validatedOutfits;
      } else {
        console.log("[validate] no validated outfits survived");
        aiOutfits = null;
      }
    }
    if (!aiOutfits?.length && verifiedChoiceFallback.length) {
      aiOutfits = filterValidOutfits(verifiedChoiceFallback, styleKey, itemMap, layered, preferences, `${occasion} ${prompt}`, weather);
      if (aiOutfits.length) generationSource = "deterministic_fallback";
    }
    if (!aiOutfits?.length) {
      console.warn("[provider] PROVIDER_GENERATION_FAILED", {
        attempts: providerTrace,
        provider: aiProviderUsed,
        style: styleKey,
        occasion
      });
      let fallbackCandidates = [];
      try {
        fallbackCandidates = await localFallback(
          wardrobeItems,
          styleKey,
          occasion,
          limit,
          layered,
          void 0,
          preferences,
          wardrobeOnly,
          deckSalt,
          weather,
          memoQuality
        );
      } catch (fallbackError) {
        console.error("[provider] deterministic fallback failed:", fallbackError);
      }
      for (const item of fallbackCandidates.flatMap((outfit) => outfit?.items || [])) {
        if (item?.id && !itemMap.has(String(item.id))) itemMap.set(String(item.id), item);
      }
      const fallbackValid = filterValidOutfits(
        fallbackCandidates,
        styleKey,
        itemMap,
        layered,
        preferences,
        `${occasion} ${prompt}`,
        weather
      );
      if (fallbackValid.length > 0) {
        aiOutfits = fallbackValid;
        generationSource = "deterministic_fallback";
        console.log("[provider] DETERMINISTIC_FALLBACK_USED", {
          candidates: fallbackCandidates.length,
          valid: fallbackValid.length,
          style: styleKey,
          occasion
        });
      }
    }
    if (!aiOutfits?.length && deckTarget === 3) {
      aiOutfits = completeChoices().map((choice) => ({ items: choice.items, description: `${selectionStyle} outfit`, stylingTips: [], confidence: 0.8 }));
      if (aiOutfits.length) generationSource = "deterministic_fallback";
    }
    if (!aiOutfits?.length) return new Response(JSON.stringify({ success: false, error: "ai_generation_unavailable", outfits: [], providerAttempts: providerTrace }), { status: 503, headers: { ...corsHeaders, "Content-Type": "application/json" } });
    const enriched = aiOutfits.map((outfit) => ({
      ...outfit,
      items: (outfit.items || []).map((item) => {
        const src = itemMap.get(String(item.id));
        if (!src) return null;
        return {
          ...src,
          id: String(src.id),
          imageUrl: src.imageUrl || src.image_url,
          // The source slot already passed every validator; enrichment must not reclassify it.
          macroCategory: src.macroCategory,
          isShopItem: isShopRow(src)
        };
      }).filter(Boolean)
    }));
    let finalOutfits = enriched;
    if (isShopOnlyDeck(stylePreferences, occasion) && !anchorItemId) {
      const beforeStrip = finalOutfits.length;
      finalOutfits = finalOutfits.map((o) => ({ ...o, items: (o.items || []).filter((it) => isShopRow(it)) })).filter((o) => (o.items || []).length > 0);
      console.log(`[shop-only] Post-enrich strip: ${beforeStrip} -> ${finalOutfits.length} outfit(s)`);
    }
    if (anchorItemId && (!isShopOnlyDeck(stylePreferences, occasion) || !!anchorItemId)) {
      const anchorDbItem = wardrobeItems.find((i) => String(i.id) === String(anchorItemId));
      if (anchorDbItem) {
        console.log(`[anchor] Enforcing anchor item ${anchorItemId} (${anchorDbItem.type || anchorDbItem.category}) in ${finalOutfits.length} AI outfits`);
        finalOutfits = enforceAnchorInOutfits(finalOutfits, anchorDbItem);
      } else {
        console.warn(`[anchor] anchorItemId ${anchorItemId} not found in wardrobeItems pool \u2014 skipping enforcement`);
      }
    }
    finalOutfits = finalOutfits.filter((outfit) => {
      const hasAnchor = !anchorItemId || (outfit.items || []).some((it) => String(it.id) === String(anchorItemId));
      const reason = hasAnchor ? (outfit.items || []).map((item) => garmentContextReason(item, qualityContext)).find(Boolean) || styleJudgeReason(outfit.items || [], styleKey, `${occasion || ""} ${prompt || ""}`, weather, anchorItemId, preferences) : "anchor_missing";
      const complete = !fourSlotOutfitReason(outfit.items, layered);
      if (reason || !complete) console.warn("[quality-reject]", { reason: reason || "missing_slots", style: styleKey, occasion });
      const repeats = outfit.items.some((it) => String(it.id) !== String(anchorItemId || "") && wasRecentlyShown(it, excludeSet) && !(selectedItemIds || []).includes(String(it.id)) && !rotation?.recycledSlots.includes(diversitySlotOf(it)));
      const budgetOk = !preferences?.maxShopBudget || fitsBudgetAmount(outfitShopSpend(outfit.items, itemMap, preferences.maxShopBudget.currency, preferences.maxShopBudget.fx), preferences.maxShopBudget.amount);
      return !reason && complete && !repeats && budgetOk;
    });
    const personalStylist = deckTarget === 3;
    let diversityCapacity;
    if (finalOutfits.length > 0) {
      const anchorForDiversity = anchorItemId && (!isShopOnlyDeck(stylePreferences, occasion) || !!anchorItemId) ? wardrobeItems.find((i) => String(i.id) === String(anchorItemId)) : void 0;
      const anchorIdentityKeys = new Set(
        anchorForDiversity ? productIdentityKeys(anchorForDiversity) : []
      );
      const diversityActiveSlots = layered ? ["top", "outerwear", "bottom", "shoes"] : ["top", "bottom", "shoes"];
      const requestedDeckTarget = Math.max(
        1,
        Math.min(Number(deckTarget) || limit, limit)
      );
      const selectDeck = (candidates) => assembleDiverseDeck({
        candidates,
        itemsOf: (o) => o.items || [],
        qualityOf: (o) => signalsFor(o).quality,
        recencyOf: (o) => signalsFor(o).recency,
        keysOf: (item) => productIdentityKeys(item),
        anchorKeys: anchorIdentityKeys.size > 0 ? anchorIdentityKeys : void 0,
        anchorSlot: anchorForDiversity ? diversitySlotOf(anchorForDiversity) : null,
        activeSlots: diversityActiveSlots,
        usedKeys: /* @__PURE__ */ new Set(),
        poolItems: wardrobeItems,
        // The displayed deck size: guarantee a diverse deck HERE and
        // keep the remaining validated looks behind it so the client
        // still has alternatives to choose from.
        target: requestedDeckTarget,
        priorityOfDeck: (deck) => deckPriority(deck.flatMap((outfit) => outfit.items)),
        maxSearchSteps: 512
      });
      let diverseDeck = selectDeck(finalOutfits);
      const needsReserve = diverseDeck.deck.length < requestedDeckTarget || diverseDeck.deck.some((look) => look.items.some((item) => String(item.id) !== String(anchorItemId || "") && (wasRecentlyShown(item, excludeSet) || wasRecentlyShown(item, previousSet))));
      if (personalStylist && needsReserve) {
        const signatures = new Set(finalOutfits.map((look) => look.items.map((item) => item.id).sort().join("|")));
        const replacements = completeChoices().filter((choice) => !signatures.has(choice.items.map((item) => item.id).sort().join("|"))).map((choice) => ({ items: choice.items, description: `${selectionStyle} outfit`, stylingTips: [], confidence: 0.8, curationRank: Number.MAX_SAFE_INTEGER }));
        finalOutfits.push(...replacements);
        diverseDeck = selectDeck(finalOutfits);
      }
      if (diverseDeck.deck.length > 0) {
        const signatureOf = (o) => (o.items || []).map((it) => String(it.id)).sort().join("|");
        const selectedSignatures = new Set(diverseDeck.deck.map(signatureOf));
        const extras = finalOutfits.filter((o) => !selectedSignatures.has(signatureOf(o)));
        if (personalStylist && diverseDeck.deck.some((outfit) => !enriched.includes(outfit))) generationSource = "deterministic_fallback";
        finalOutfits = personalStylist ? diverseDeck.deck : [...diverseDeck.deck, ...extras];
        const productDiversity = analyzeDeckProductDiversity(
          diverseDeck.deck.map((o) => o.items || []),
          {
            anchorKeys: anchorIdentityKeys,
            forcedSlots: diverseDeck.inventory.forcedRepeatSlots
          }
        );
        diversityCapacity = {
          activeSlots: diverseDeck.inventory.activeSlots,
          strictSlots: diverseDeck.inventory.strictSlots,
          forcedRepeatSlots: diverseDeck.inventory.forcedRepeatSlots
        };
        console.info("[stylist-selection]", {
          candidates: finalOutfits.length,
          selected: diverseDeck.deck.length,
          reserveUsed: personalStylist && needsReserve,
          reason: diverseDeck.relaxReason || "strict",
          illegalReuse: productDiversity.illegalProductReuseCount
        });
      }
    }
    let shoeRotationAudit;
    if (personalStylist && finalOutfits.length === 3) {
      const shoes = wardrobeItems.filter((item) => item.macroCategory === "shoes" && eligibleCandidate(item));
      const candidateShoes = new Set(finalOutfits.flatMap((look) => look.items.filter((item) => item.macroCategory === "shoes").map((item) => item.id)));
      const assigned = assignRotatingShoes(
        finalOutfits,
        shoes,
        (parts) => requiredChoiceIds.every((id) => parts.some((item) => String(item.id) === id)) && acceptsCompleteChoice(parts),
        deckPriority,
        (parts) => memoQuality(parts, styleKey, `${occasion} ${prompt}`, weather, anchorItemId, preferences).score,
        anchorItemId
      );
      if (assigned.deck.length === 3) finalOutfits = assigned.deck;
      const selected = finalOutfits.flatMap((look) => look.items.filter((item) => item.macroCategory === "shoes"));
      const validPhysicalShoesAvailable = countDistinctProducts(shoes);
      const previousShoeOverlap = selected.filter((shoe) => String(shoe.id) !== String(anchorItemId || "") && wasRecentlyShown(shoe, previousShoeSet)).length;
      const minimumPossibleOverlap = shoes.some((shoe) => String(shoe.id) === String(anchorItemId || "")) ? 0 : Math.max(0, finalOutfits.length + countDistinctProducts(shoes.filter((shoe) => wasRecentlyShown(shoe, previousShoeSet))) - validPhysicalShoesAvailable);
      shoeRotationAudit = {
        generationRequestId: body.generationRequestId,
        validPhysicalShoesAvailable,
        minimumPossibleOverlap,
        minimumFeasibleOverlap: assigned.feasibleMinimumOverlap,
        previousShoeOverlap,
        previousDeckShoeIds: shoes.filter((shoe) => wasRecentlyShown(shoe, previousShoeSet)).map((shoe) => shoe.id),
        freshShoesAvailable: countDistinctProducts(shoes.filter((shoe) => !wasRecentlyShown(shoe, excludeSet))),
        admissibleShoes: shoes.map((shoe) => ({ id: shoe.id, keys: productIdentityKeys(shoe) })),
        fetchedShoeIds,
        rankedShoeIds: shoes.map((shoe) => shoe.id),
        completeCandidateShoeIds: [...new Set((completeChoiceCache || []).flatMap((choice) => choice.items.filter((item) => item.macroCategory === "shoes").map((item) => item.id)))],
        preAssignmentShoeIds: [...candidateShoes],
        selectedShoeIds: selected.map((shoe) => shoe.id)
      };
      if (assigned.feasibleMinimumOverlap !== null && previousShoeOverlap > assigned.feasibleMinimumOverlap) {
        return new Response(JSON.stringify({ success: false, outfits: [], code: "shoe_rotation_unavailable", shoeRotationAudit }), { status: 422, headers: { ...corsHeaders, "Content-Type": "application/json" } });
      }
    }
    finalOutfits = finalOutfits.slice(0, limit);
    if (personalStylist && finalOutfits.length !== 3) return new Response(JSON.stringify({ success: false, outfits: [], code: "incomplete_outfit_deck", error: "Three complete, suitable outfits are required. Add more suitable clothing or adjust the request.", debug: { completeChoices: completeChoices().length, candidateSlots: Object.fromEntries(["top", "outerwear", "bottom", "shoes"].map((slot) => [slot, wardrobeItems.filter((item) => item.macroCategory === slot).length])), choiceRejections: Object.fromEntries(choiceRejections) } }), { status: 422, headers: { ...corsHeaders, "Content-Type": "application/json" } });
    if (!finalOutfits.length) return new Response(JSON.stringify({ success: false, error: "ai_generation_unavailable", outfits: [], providerAttempts: providerTrace }), { status: 503, headers: { ...corsHeaders, "Content-Type": "application/json" } });
    finalOutfits = finalOutfits.map((outfit) => ({ ...outfit, qualityAssessment: memoQuality(outfit.items, styleKey, `${occasion} ${prompt}`, weather, anchorItemId, preferences) }));
    const previousDeckOverlapBySlot = Object.fromEntries(["outerwear", "top", "bottom", "shoes"].map((slot) => [slot, finalOutfits.flatMap((outfit) => outfit.items).filter((item) => item.macroCategory === slot && String(item.id) !== String(anchorItemId || "") && wasRecentlyShown(item, slot === "shoes" ? previousShoeSet : previousSet)).length]));
    return new Response(JSON.stringify({ success: true, generationVersion: GENERATION_VERSION, shoeRotationAudit, generationRequestId: body.generationRequestId, rotationScope: body.rotationScope, previousDeckOverlapBySlot, outfits: finalOutfits, source: generationSource, layered, provider: aiProviderUsed, providerAttempts: providerTrace, diversity: diversityCapacity, rotation: rotation ? { recycledSlots: rotation.recycledSlots, freshProducts: freshProductCount(finalOutfits.flatMap((outfit) => outfit.items), excludeSet, anchorItemId) } : void 0, budgetReference: requestFx ? { id: requestFx.snapshot.id, asOf: requestFx.snapshot.asOf, source: "CBU" } : void 0 }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" }
    });
  } catch (error) {
    if (error instanceof CatalogLookupError) return new Response(JSON.stringify({ success: false, outfits: [], error: "catalog_lookup_unavailable" }), { status: 503, headers: { ...corsHeaders, "Content-Type": "application/json" } });
    if (error instanceof BudgetFxError) return new Response(JSON.stringify({ success: false, outfits: [], error: "budget_rates_unavailable" }), { status: 503, headers: { ...corsHeaders, "Content-Type": "application/json" } });
    console.error("generate-outfits error:", error);
    return new Response(JSON.stringify({ success: false, error: error.message, outfits: [] }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
      status: 500
    });
  }
});
