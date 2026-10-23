/**
 * The Personal Stylist simulator as an automated regression gate.
 *
 * `scripts/stylist-diversity-report.ts` runs the REAL production chain
 * (valid pool → candidate looks → assembleDiverseDeck → createSlotNormalizer
 * replacements) for 20 style × occasion scenarios.
 *
 * The invariant is PRODUCT uniqueness: whenever the inventory has at least
 * three distinct products per active slot, no non-anchor product may appear
 * in more than one card. Archetype variety is reported but never required.
 */
import {
    formatReport,
    runAllScenarios,
    SCENARIOS,
} from '../../scripts/stylist-diversity-report';

describe('personal stylist simulator — 20 scenarios', () => {
    const reports = runAllScenarios();

    it('covers 20 style × occasion scenarios', () => {
        expect(SCENARIOS).toHaveLength(20);
        expect(reports).toHaveLength(20);
    });

    it('every scenario produces three complete looks', () => {
        for (const report of reports) {
            expect(report.archetypes.length).toBe(3);
        }
    });

    it('reports pairwise similarity as a soft signal', () => {
        for (const report of reports) {
            expect(report.similarities).toHaveLength(3);
            for (const score of report.similarities) {
                expect(Number.isFinite(score)).toBe(true);
                expect(score).toBeLessThanOrEqual(1);
            }
        }
    });

    it('no deck reuses a non-anchor product when the inventory can avoid it', () => {
        for (const report of reports) {
            if (report.supportsUniqueProducts) {
                expect(report.nonAnchorProductReuseCount).toBe(0);
                expect(report.repeatedProductKeys).toEqual([]);
            }
        }
    });

    it('no deck breaks the requested dress code', () => {
        for (const report of reports) {
            expect(report.notes.filter((note) => note.includes('style judge'))).toHaveLength(0);
        }
    });

    it('never relaxes uniqueness while the inventory has enough products', () => {
        for (const report of reports) {
            expect(report.supportsUniqueProducts).toBe(true);
            expect(report.mode).toBe('STRICT');
            expect(report.relaxReason).toBeNull();
        }
    });

    it('every scenario passes', () => {
        const failed = reports.filter((report) => !report.pass);
        expect(failed.map((report) => `scenario ${report.scenario.id}: ${report.notes.join('; ')}`)).toEqual([]);
    });

    it('prints the measured report', () => {
        // eslint-disable-next-line no-console
        console.log(formatReport(reports));
        expect(true).toBe(true);
    });
});
