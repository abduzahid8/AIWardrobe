/**
 * useQuizState — manages style quiz state machine
 */

import { useState, useCallback } from 'react';

export interface QuizAnswer {
    questionId: string;
    selectedOptions: string[];
}

export function useQuizState(totalQuestions: number) {
    const [currentQuestion, setCurrentQuestion] = useState(0);
    const [answers, setAnswers] = useState<Record<string, QuizAnswer>>({});
    const [isComplete, setIsComplete] = useState(false);

    const progress = (currentQuestion + 1) / totalQuestions;

    const selectOption = useCallback(
        (questionId: string, option: string, multiSelect = false) => {
            setAnswers((prev) => {
                const existing = prev[questionId]?.selectedOptions || [];
                let updated: string[];

                if (multiSelect) {
                    updated = existing.includes(option)
                        ? existing.filter((o) => o !== option)
                        : [...existing, option];
                } else {
                    updated = [option];
                }

                return {
                    ...prev,
                    [questionId]: { questionId, selectedOptions: updated },
                };
            });
        },
        []
    );

    const nextQuestion = useCallback(() => {
        if (currentQuestion < totalQuestions - 1) {
            setCurrentQuestion((q) => q + 1);
        } else {
            setIsComplete(true);
        }
    }, [currentQuestion, totalQuestions]);

    const prevQuestion = useCallback(() => {
        if (currentQuestion > 0) {
            setCurrentQuestion((q) => q - 1);
        }
    }, [currentQuestion]);

    const resetQuiz = useCallback(() => {
        setCurrentQuestion(0);
        setAnswers({});
        setIsComplete(false);
    }, []);

    return {
        currentQuestion,
        answers,
        isComplete,
        progress,
        selectOption,
        nextQuestion,
        prevQuestion,
        resetQuiz,
        getAnswer: (questionId: string) => answers[questionId]?.selectedOptions || [],
    };
}
