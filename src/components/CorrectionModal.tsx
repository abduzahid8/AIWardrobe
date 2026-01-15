/**
 * 🔧 Correction Modal Component
 * 
 * Allows users to correct AI detection mistakes.
 * Part of Phase 3: Human Feedback Loop
 */

import React, { useState } from 'react';
import {
    View,
    Text,
    Modal,
    TouchableOpacity,
    ScrollView,
    StyleSheet,
    Dimensions,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { submitCorrection, getTypeOptionsForCategory } from '../services/feedbackService';

interface CorrectionModalProps {
    visible: boolean;
    onClose: () => void;
    originalType: string;
    category: string;
    confidence: number;
    onCorrected?: (newType: string) => void;
}

const { width } = Dimensions.get('window');

const CorrectionModal: React.FC<CorrectionModalProps> = ({
    visible,
    onClose,
    originalType,
    category,
    confidence,
    onCorrected,
}) => {
    const [selectedType, setSelectedType] = useState<string | null>(null);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [submitted, setSubmitted] = useState(false);

    const typeOptions = getTypeOptionsForCategory(category);

    const handleSubmit = async () => {
        if (!selectedType) return;

        setIsSubmitting(true);
        const success = await submitCorrection(
            originalType,
            selectedType,
            category,
            confidence
        );

        setIsSubmitting(false);
        if (success) {
            setSubmitted(true);
            onCorrected?.(selectedType);
            setTimeout(() => {
                setSubmitted(false);
                setSelectedType(null);
                onClose();
            }, 1500);
        }
    };

    return (
        <Modal
            visible={visible}
            transparent
            animationType="slide"
            onRequestClose={onClose}
        >
            <View style={styles.overlay}>
                <View style={styles.container}>
                    {/* Header */}
                    <View style={styles.header}>
                        <Text style={styles.title}>Correct Detection</Text>
                        <TouchableOpacity onPress={onClose} style={styles.closeBtn}>
                            <Ionicons name="close" size={24} color="#666" />
                        </TouchableOpacity>
                    </View>

                    {/* Current Detection */}
                    <View style={styles.currentBox}>
                        <Text style={styles.label}>AI detected:</Text>
                        <Text style={styles.currentType}>{originalType}</Text>
                        <Text style={styles.confidence}>
                            Confidence: {(confidence * 100).toFixed(0)}%
                        </Text>
                    </View>

                    {/* Success Message */}
                    {submitted ? (
                        <View style={styles.successBox}>
                            <Ionicons name="checkmark-circle" size={48} color="#4CAF50" />
                            <Text style={styles.successText}>Thanks! Correction saved.</Text>
                            <Text style={styles.successSubtext}>
                                This helps improve AI accuracy.
                            </Text>
                        </View>
                    ) : (
                        <>
                            {/* Type Options */}
                            <Text style={styles.label}>What is the correct type?</Text>
                            <ScrollView
                                style={styles.optionsList}
                                showsVerticalScrollIndicator={false}
                            >
                                {typeOptions.map((type) => (
                                    <TouchableOpacity
                                        key={type}
                                        style={[
                                            styles.optionBtn,
                                            selectedType === type && styles.optionBtnSelected,
                                        ]}
                                        onPress={() => setSelectedType(type)}
                                    >
                                        <Text
                                            style={[
                                                styles.optionText,
                                                selectedType === type && styles.optionTextSelected,
                                            ]}
                                        >
                                            {type}
                                        </Text>
                                        {selectedType === type && (
                                            <Ionicons name="checkmark" size={20} color="#fff" />
                                        )}
                                    </TouchableOpacity>
                                ))}
                            </ScrollView>

                            {/* Submit Button */}
                            <TouchableOpacity
                                style={[
                                    styles.submitBtn,
                                    (!selectedType || isSubmitting) && styles.submitBtnDisabled,
                                ]}
                                onPress={handleSubmit}
                                disabled={!selectedType || isSubmitting}
                            >
                                <Text style={styles.submitText}>
                                    {isSubmitting ? 'Submitting...' : 'Submit Correction'}
                                </Text>
                            </TouchableOpacity>
                        </>
                    )}
                </View>
            </View>
        </Modal>
    );
};

const styles = StyleSheet.create({
    overlay: {
        flex: 1,
        backgroundColor: 'rgba(0,0,0,0.5)',
        justifyContent: 'flex-end',
    },
    container: {
        backgroundColor: '#fff',
        borderTopLeftRadius: 20,
        borderTopRightRadius: 20,
        padding: 20,
        maxHeight: '80%',
    },
    header: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 16,
    },
    title: {
        fontSize: 20,
        fontWeight: '700',
        color: '#1a1a1a',
    },
    closeBtn: {
        padding: 4,
    },
    currentBox: {
        backgroundColor: '#f5f5f5',
        padding: 16,
        borderRadius: 12,
        marginBottom: 16,
    },
    label: {
        fontSize: 14,
        color: '#666',
        marginBottom: 8,
    },
    currentType: {
        fontSize: 18,
        fontWeight: '600',
        color: '#333',
    },
    confidence: {
        fontSize: 12,
        color: '#999',
        marginTop: 4,
    },
    optionsList: {
        maxHeight: 300,
        marginBottom: 16,
    },
    optionBtn: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: 14,
        backgroundColor: '#f5f5f5',
        borderRadius: 10,
        marginBottom: 8,
    },
    optionBtnSelected: {
        backgroundColor: '#007AFF',
    },
    optionText: {
        fontSize: 16,
        color: '#333',
    },
    optionTextSelected: {
        color: '#fff',
        fontWeight: '600',
    },
    submitBtn: {
        backgroundColor: '#007AFF',
        padding: 16,
        borderRadius: 12,
        alignItems: 'center',
    },
    submitBtnDisabled: {
        backgroundColor: '#ccc',
    },
    submitText: {
        color: '#fff',
        fontSize: 16,
        fontWeight: '600',
    },
    successBox: {
        alignItems: 'center',
        padding: 32,
    },
    successText: {
        fontSize: 18,
        fontWeight: '600',
        color: '#4CAF50',
        marginTop: 12,
    },
    successSubtext: {
        fontSize: 14,
        color: '#666',
        marginTop: 4,
    },
});

export default CorrectionModal;
