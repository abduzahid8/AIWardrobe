/**
 * The Personal Stylist simulator as an automated regression gate.
 *
 * `scripts/stylist-diversity-report.ts` runs the REAL production chain
 * (valid pool → candidate looks → assembleDiverseDeck → createSlotNormalizer
 * replacements) for 20 style × occasion scenarios and measures the quality of
 * every deck: the outfit archetype of each look, the three pairwise
 * similarity scores and the diversity mode.
 *
 * This test turns that report into an invariant: whenever the inventory can
 * support three different styling directions, the three cards must BE three
 * different styling directions.
 */
import {
    formatReport,
    runAllScenarios,
    SCENARIOS,
} from '../../scripts/stylist-diversity-report';
import {
    MAX_VISUAL_SIMILARITY,
} from '../../features/outfit-generator/utils/outfitDiversity';

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

    it('every deck keeps the three looks pairwise below the strict threshold', () => {
        for (const report of reports) {
            expect(report.similarities).toHaveLength(3);
            for (const score of report.similarities) {
                expect(score).toBeLessThan(MAX_VISUAL_SIMILARITY);
            }
        }
    });

    it('no two looks in a deck share the same outfit formula', () => {
        for (const report of reports) {
            expect(report.notes.filter((note) => note.includes('outfit formula'))).toHaveLength(0);
        }
    });

    it('no deck breaks the requested dress code', () => {
        for (const report of reports) {
            expect(report.notes.filter((note) => note.includes('style judge'))).toHaveLength(0);
        }
    });

    it('never relaxes while the inventory can support three concepts', () => {
        for (const report of reports) {
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
