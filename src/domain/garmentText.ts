// Shared sleeve evidence; adjective and concatenated feed spellings included.
export const SHORT_SLEEVE_RE = /\bshort[\s-]?sleeve(?:s|d)?\b/;

/** Normalize multilingual catalog evidence before applying the same rules. */
export function normalizeGarmentText(value: string): string {
  const aliases: Array<[RegExp, string]> = [
    [/свитшот/gi, 'sweatshirt'],
    [/худи/gi, 'hoodie'],
    [/логотип|monologo/gi, 'logo'],
    [/лофер[а-я]*/gi, 'loafers'],
    [/кроссов[а-я]*|кеды/gi, 'sneakers'],
    [/рубаш[а-я]*/gi, 'shirt'],
    [/брюк[а-я]*/gi, 'trousers'],
    [/пиджак[а-я]*/gi, 'blazer'],
    [/смокинг[а-я]*/gi, 'tuxedo'],
    [/замш[а-я]*/gi, 'suede'],
    [/коричнев[а-я]*/gi, 'brown'],
    [/черн[а-я]*|чёрн[а-я]*/gi, 'black'],
    [/бел[а-я]*/gi, 'white'],
    [/футбол[а-я]*/gi, 't-shirt'],
    [/спортивн[а-я]*/gi, 'athletic'],
    [/трениров[а-я]*/gi, 'training'],
    [/шорт[а-я]*/gi, 'shorts'],
  ];
  return aliases.reduce(
    (text, [pattern, replacement]) => text.replace(pattern, ` ${replacement} `),
    value.toLowerCase()
  );
}
