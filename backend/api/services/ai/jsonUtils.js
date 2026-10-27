/**
 * JSON extraction helpers for LLM output — shared by the failover facade.
 * Kept dependency-free so it is directly unit-testable.
 */

/**
 * Extract and parse the first JSON object/array embedded in model text.
 * Handles markdown fences, prose around the JSON, and arrays vs objects.
 * @param {string} text
 * @returns {any} Parsed JSON
 * @throws {Error} when nothing parseable is found
 */
export function extractJSON(text) {
    if (typeof text !== "string") throw new Error("not_a_string");

    // Prefer ```json fenced blocks, then any fenced block, then bare JSON.
    const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/);
    const haystack = fenced ? fenced[1] : text;

    const objMatch = haystack.match(/\{[\s\S]*\}/);
    const arrMatch = haystack.match(/\[[\s\S]*\]/);
    const raw =
        arrMatch && (!objMatch || arrMatch.index < objMatch.index)
            ? arrMatch[0]
            : objMatch?.[0];

    if (!raw) throw new Error("no_json_found");
    return JSON.parse(raw);
}

/**
 * Validate that model text contains a JSON array of numbers of a given
 * length — used by the outfit-scoring route as its provider contract.
 */
export function parseNumberArray(text, expectedLength) {
    const parsed = extractJSON(text);
    const arr = Array.isArray(parsed) ? parsed : null;
    if (!arr || arr.length !== expectedLength || arr.some((n) => typeof n !== "number")) {
        throw new Error(
            `expected JSON array of ${expectedLength} numbers, got ${Array.isArray(parsed) ? `length ${parsed.length}` : typeof parsed}`
        );
    }
    return arr;
}

export default { extractJSON, parseNumberArray };
