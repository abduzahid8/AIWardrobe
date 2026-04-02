import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Modal, TouchableWithoutFeedback } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { BlurView } from 'expo-blur';
import Animated, { FadeIn, FadeOut, SlideInDown } from 'react-native-reanimated';

interface AddOutfitPopoverProps {
    visible: boolean;
    onClose: () => void;
    onOptionPress: (option: string) => void;
    dateLabel: string;
}

export const AddOutfitPopover = ({ visible, onClose, onOptionPress, dateLabel }: AddOutfitPopoverProps) => {
    return (
        <Modal visible={visible} transparent animationType="none">
            <TouchableWithoutFeedback onPress={onClose}>
                <View style={styles.overlay}>
                    <Animated.View entering={FadeIn} exiting={FadeOut} style={StyleSheet.absoluteFill}>
                        <BlurView intensity={30} tint="dark" style={StyleSheet.absoluteFill} />
                    </Animated.View>

                    <Animated.View entering={SlideInDown} exiting={FadeOut} style={styles.popoverContainer}>
                        <View style={styles.popover}>
                            <Text style={styles.dateLabel}>{dateLabel}</Text>

                            <TouchableOpacity
                                style={styles.option}
                                onPress={() => onOptionPress('closet')}
                            >
                                <Text style={styles.optionText}>Select from closet</Text>
                                <View style={styles.iconCircle}>
                                    <Ionicons name="shirt-outline" size={20} color="#FFF" />
                                </View>
                            </TouchableOpacity>

                            <View style={styles.separator} />

                            <TouchableOpacity
                                style={styles.option}
                                onPress={() => onOptionPress('saved')}
                            >
                                <Text style={styles.optionText}>From saved looks</Text>
                                <View style={styles.iconCircle}>
                                    <Ionicons name="heart-outline" size={20} color="#FFF" />
                                </View>
                            </TouchableOpacity>

                            <View style={styles.separator} />

                            <TouchableOpacity
                                style={styles.option}
                                onPress={() => onOptionPress('photo')}
                            >
                                <Text style={styles.optionText}>With a photo</Text>
                                <View style={styles.iconCircle}>
                                    <Ionicons name="camera-outline" size={20} color="#FFF" />
                                </View>
                            </TouchableOpacity>
                        </View>
                    </Animated.View>
                </View>
            </TouchableWithoutFeedback>
        </Modal>
    );
};

const styles = StyleSheet.create({
    overlay: {
        flex: 1,
        justifyContent: 'flex-end',
        alignItems: 'center',
        paddingBottom: 40,
    },
    popoverContainer: {
        width: '85%',
    },
    popover: {
        backgroundColor: '#2D2D2D',
        borderRadius: 32,
        padding: 24,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 10 },
        shadowOpacity: 0.3,
        shadowRadius: 20,
        elevation: 10,
    },
    dateLabel: {
        fontSize: 14,
        fontWeight: '600',
        color: '#98A2B3',
        textAlign: 'center',
        marginBottom: 20,
    },
    option: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        paddingVertical: 12,
    },
    optionText: {
        fontSize: 16,
        fontWeight: '600',
        color: '#FFF',
    },
    iconCircle: {
        width: 44,
        height: 44,
        borderRadius: 22,
        backgroundColor: 'rgba(255,255,255,0.1)',
        justifyContent: 'center',
        alignItems: 'center',
    },
    separator: {
        height: 1,
        backgroundColor: 'rgba(255,255,255,0.05)',
        marginVertical: 4,
    },
});
