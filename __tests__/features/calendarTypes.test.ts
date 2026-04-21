import { matchesCategory } from '../../features/calendar/types';

describe('calendar category matching', () => {
    it('matches singular bottom aliases for pants slots', () => {
        expect(matchesCategory('bottom', 'pants')).toBe(true);
        expect(matchesCategory('trouser', 'pants')).toBe(true);
    });

    it('matches common singular shoe aliases', () => {
        expect(matchesCategory('shoe', 'shoes')).toBe(true);
        expect(matchesCategory('loafer', 'shoes')).toBe(true);
    });

    it('still matches descriptive top names', () => {
        expect(matchesCategory('ribbed knit top', 'top')).toBe(true);
        expect(matchesCategory('basic tee', 'top')).toBe(true);
    });
});
