/**
 * Tests for the quiz state management hook
 */

import { renderHook, act } from '@testing-library/react-native';
import { useQuizState } from '../../features/style-quiz/hooks/useQuizState';

describe('useQuizState', () => {
    const TOTAL_QUESTIONS = 5;

    it('starts at question 0 with empty answers', () => {
        const { result } = renderHook(() => useQuizState(TOTAL_QUESTIONS));
        expect(result.current.currentQuestion).toBe(0);
        expect(result.current.answers).toEqual({});
        expect(result.current.isComplete).toBe(false);
        expect(result.current.progress).toBeCloseTo(0.2); // 1/5
    });

    it('navigates to next question', () => {
        const { result } = renderHook(() => useQuizState(TOTAL_QUESTIONS));
        act(() => result.current.nextQuestion());
        expect(result.current.currentQuestion).toBe(1);
        expect(result.current.progress).toBeCloseTo(0.4); // 2/5
    });

    it('navigates to previous question', () => {
        const { result } = renderHook(() => useQuizState(TOTAL_QUESTIONS));
        act(() => result.current.nextQuestion());
        act(() => result.current.prevQuestion());
        expect(result.current.currentQuestion).toBe(0);
    });

    it('does not go below 0', () => {
        const { result } = renderHook(() => useQuizState(TOTAL_QUESTIONS));
        act(() => result.current.prevQuestion());
        expect(result.current.currentQuestion).toBe(0);
    });

    it('marks quiz complete on last question', () => {
        const { result } = renderHook(() => useQuizState(2));
        act(() => result.current.nextQuestion()); // Q1
        act(() => result.current.nextQuestion()); // Complete
        expect(result.current.isComplete).toBe(true);
    });

    it('selects single option when multiSelect is false', () => {
        const { result } = renderHook(() => useQuizState(TOTAL_QUESTIONS));
        act(() => result.current.selectOption('q1', 'option-a'));
        expect(result.current.getAnswer('q1')).toEqual(['option-a']);

        act(() => result.current.selectOption('q1', 'option-b'));
        expect(result.current.getAnswer('q1')).toEqual(['option-b']);
    });

    it('toggles options when multiSelect is true', () => {
        const { result } = renderHook(() => useQuizState(TOTAL_QUESTIONS));
        act(() => result.current.selectOption('q1', 'option-a', true));
        act(() => result.current.selectOption('q1', 'option-b', true));
        expect(result.current.getAnswer('q1')).toEqual(['option-a', 'option-b']);

        act(() => result.current.selectOption('q1', 'option-a', true));
        expect(result.current.getAnswer('q1')).toEqual(['option-b']);
    });

    it('resets quiz state', () => {
        const { result } = renderHook(() => useQuizState(TOTAL_QUESTIONS));
        act(() => result.current.selectOption('q1', 'option-a'));
        act(() => result.current.nextQuestion());
        act(() => result.current.resetQuiz());

        expect(result.current.currentQuestion).toBe(0);
        expect(result.current.answers).toEqual({});
        expect(result.current.isComplete).toBe(false);
    });
});
