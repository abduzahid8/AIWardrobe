// CelebrityClothingCard - Massimo Dutti style product card
// Clean white background, centered cutout, professional styling

import React from 'react';
import {
    View,
    Image,
    Text,
    StyleSheet,
    Pressable,
    ViewStyle,
    Dimensions,
} from 'react-native';
import Animated, {
    useSharedValue,
    useAnimatedStyle,
    withSpring,
    FadeInUp,
} from 'react-native-reanimated';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import { LinearGradient } from 'expo-linear-gradient';
import { useTheme } from '../../src/hooks/useTheme';

const { width } = Dimensions.get('window');

interface ClothingAttribute {
    primaryColor?: string;
    colorHex?: string;
    pattern?: { type: string; confidence: number };
    material?: { type: string; confidence: number };
}

interface CelebrityClothingCardProps {
    /** Clothing cutout image (base64 or URI) */
    imageUri: string;
    /** Clothing type (e.g., "Polo Sweater", "Denim Jacket") */
    clothingType: string;
    /** Specific type from AI detection */
    specificType?: string;
    /** Primary color name */
    color?: string;
    /** Hex color code */
    colorHex?: string;
    /** Material type */
    material?: string;
    /** Pattern type */
    pattern?: string;
    /** Confidence score 0-1 */
    confidence?: number;
    /** Full attributes object */
    attributes?: ClothingAttribute;
    /** Callback when card is pressed */
    onPress?: () => void;
    /** Callback when save button is pressed */
    onSave?: () => void;
    /** Whether item is already saved */
    isSaved?: boolean;
    /** Card index for staggered animation */
    index?: number;
    /** Custom style */
    style?: ViewStyle;
}

export const CelebrityClothingCard: React.FC<CelebrityClothingCardProps> = ({
    imageUri,
    clothingType,
    specificType,
    color,
    colorHex,
    material,
    pattern,
    confidence,
    attributes,
    onPress,
    onSave,
    isSaved = false,
    index = 0,
    style,
}) => {
    const { colors, spacing, borderRadius, shadows } = useTheme();
    const pressed = useSharedValue(false);

    const handlePressIn = () => {
        pressed.value = true;
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    };

    const handlePressOut = () => {
        pressed.value = false;
    };

    const handleSave = () => {
        if (onSave) {
            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
            onSave();
        }
    };

    const animatedCardStyle = useAnimatedStyle(() => {
        const scale = pressed.value ? 0.97 : 1;
        return {
            transform: [{ scale: withSpring(scale, { damping: 15, stiffness: 400 }) }],
        };
    });

    // Display name: prefer specificType over generic type
    const displayType = specificType || clothingType || 'Clothing';
    const displayColor = color || attributes?.primaryColor || '';
    const displayMaterial = material || attributes?.material?.type || '';
    const displayPattern = pattern || attributes?.pattern?.type || '';

    // Card width for 2-column grid
    const cardWidth = (width - 48) / 2;

    return (
        <Animated.View
            entering={FadeInUp.delay(index * 100).springify()}
            style={[{ width: cardWidth, padding: spacing.xs }, style]}
        >
            <Pressable
                onPress={onPress}
                onPressIn={handlePressIn}
                onPressOut={handlePressOut}
                onLongPress={onSave}
            >
                <Animated.View
                    style={[
                        styles.card,
                        {
                            backgroundColor: '#F8F9FA', // Light gray like reference
                            borderRadius: borderRadius.l,
                            ...shadows.card,
                        },
                        animatedCardStyle,
                    ]}
                >
                    {/* Image Container - Clean White Background */}
                    <View style={styles.imageContainer}>
                        <Image
                            source={{ uri: imageUri }}
                            style={styles.image}
                            resizeMode="contain"
                        />

                        {/* Save Button */}
                        {onSave && (
                            <Pressable
                                onPress={handleSave}
                                style={[
                                    styles.saveButton,
                                    {
                                        backgroundColor: isSaved
                                            ? colors.success
                                            : 'rgba(255,255,255,0.95)',
                                    },
                                ]}
                                hitSlop={{ top: 10, right: 10, bottom: 10, left: 10 }}
                            >
                                <Ionicons
                                    name={isSaved ? 'checkmark' : 'add'}
                                    size={18}
                                    color={isSaved ? '#fff' : colors.text.primary}
                                />
                            </Pressable>
                        )}

                        {/* Confidence Badge */}
                        {confidence && confidence > 0.7 && (
                            <View style={styles.confidenceBadge}>
                                <Ionicons name="sparkles" size={10} color="#FFD700" />
                            </View>
                        )}
                    </View>

                    {/* Info Section */}
                    <View style={[styles.info, { padding: spacing.s }]}>
                        {/* Color Dot + Type */}
                        <View style={styles.typeRow}>
                            {colorHex && (
                                <View
                                    style={[
                                        styles.colorDot,
                                        { backgroundColor: colorHex },
                                    ]}
                                />
                            )}
                            <Text
                                style={[
                                    styles.typeName,
                                    { color: colors.text.primary },
                                ]}
                                numberOfLines={1}
                            >
                                {displayType}
                            </Text>
                        </View>

                        {/* Color Name */}
                        {displayColor && (
                            <Text
                                style={[
                                    styles.colorName,
                                    { color: colors.text.secondary },
                                ]}
                                numberOfLines={1}
                            >
                                {displayColor}
                            </Text>
                        )}

                        {/* Material/Pattern Tags */}
                        {(displayMaterial || displayPattern) && (
                            <View style={styles.tagRow}>
                                {displayMaterial && displayMaterial !== 'unknown' && (
                                    <View
                                        style={[
                                            styles.tag,
                                            { backgroundColor: colors.surfaceHighlight },
                                        ]}
                                    >
                                        <Text
                                            style={[
                                                styles.tagText,
                                                { color: colors.text.muted },
                                            ]}
                                        >
                                            {displayMaterial}
                                        </Text>
                                    </View>
                                )}
                                {displayPattern && displayPattern !== 'solid' && displayPattern !== 'unknown' && (
                                    <View
                                        style={[
                                            styles.tag,
                                            { backgroundColor: colors.surfaceHighlight },
                                        ]}
                                    >
                                        <Text
                                            style={[
                                                styles.tagText,
                                                { color: colors.text.muted },
                                            ]}
                                        >
                                            {displayPattern}
                                        </Text>
                                    </View>
                                )}
                            </View>
                        )}
                    </View>
                </Animated.View>
            </Pressable>
        </Animated.View>
    );
};

const styles = StyleSheet.create({
    card: {
        overflow: 'hidden',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.08,
        shadowRadius: 12,
        elevation: 4,
    },
    imageContainer: {
        width: '100%',
        aspectRatio: 0.85, // Slightly taller for clothing
        backgroundColor: '#FFFFFF', // Pure white background like reference
        justifyContent: 'center',
        alignItems: 'center',
        position: 'relative',
    },
    image: {
        width: '85%',
        height: '85%',
    },
    saveButton: {
        position: 'absolute',
        top: 10,
        right: 10,
        width: 32,
        height: 32,
        borderRadius: 16,
        justifyContent: 'center',
        alignItems: 'center',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 4,
        elevation: 2,
    },
    confidenceBadge: {
        position: 'absolute',
        top: 10,
        left: 10,
        width: 22,
        height: 22,
        borderRadius: 11,
        backgroundColor: 'rgba(255,255,255,0.95)',
        justifyContent: 'center',
        alignItems: 'center',
    },
    info: {
        gap: 4,
    },
    typeRow: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 8,
    },
    colorDot: {
        width: 12,
        height: 12,
        borderRadius: 6,
        borderWidth: 1,
        borderColor: 'rgba(0,0,0,0.1)',
    },
    typeName: {
        fontSize: 15,
        fontWeight: '600',
        flex: 1,
    },
    colorName: {
        fontSize: 13,
        fontWeight: '400',
        marginLeft: 20, // Align with type name (after dot)
    },
    tagRow: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        gap: 6,
        marginTop: 4,
    },
    tag: {
        paddingHorizontal: 8,
        paddingVertical: 3,
        borderRadius: 6,
    },
    tagText: {
        fontSize: 10,
        fontWeight: '500',
        textTransform: 'capitalize',
    },
});

export default CelebrityClothingCard;
