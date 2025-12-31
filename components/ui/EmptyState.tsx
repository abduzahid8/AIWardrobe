import React from 'react';
import {
    View,
    Text,
    StyleSheet,
    ViewStyle,
    Image,
    ImageSourcePropType,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing } from '../../src/theme';
import { AnimatedButton } from '../AnimatedButton';

type EmptyStateVariant = 'default' | 'search' | 'error' | 'wardrobe' | 'outfit';

interface EmptyStateProps {
    title: string;
    description?: string;
    icon?: keyof typeof Ionicons.glyphMap;
    image?: ImageSourcePropType;
    variant?: EmptyStateVariant;
    actionLabel?: string;
    onActionPress?: () => void;
    style?: ViewStyle;
}

/**
 * Empty state component for when there's no data to display
 * 
 * @example
 * <EmptyState
 *   title="No items yet"
 *   description="Add your first clothing item to get started"
 *   icon="shirt-outline"
 *   actionLabel="Add Item"
 *   onActionPress={() => navigation.navigate('AddItem')}
 * />
 */
export const EmptyState: React.FC<EmptyStateProps> = ({
    title,
    description,
    icon,
    image,
    variant = 'default',
    actionLabel,
    onActionPress,
    style,
}) => {
    const variantConfig = getVariantConfig(variant);
    const iconName = icon || variantConfig.icon;

    return (
        <View style={[styles.container, style]}>
            {image ? (
                <Image source={image} style={styles.image} resizeMode="contain" />
            ) : (
                <View style={[styles.iconContainer, { backgroundColor: variantConfig.iconBg }]}>
                    <Ionicons name={iconName} size={48} color={variantConfig.iconColor} />
                </View>
            )}

            <Text style={styles.title}>{title}</Text>

            {description && (
                <Text style={styles.description}>{description}</Text>
            )}

            {actionLabel && onActionPress && (
                <AnimatedButton
                    title={actionLabel}
                    onPress={onActionPress}
                    style={styles.actionButton}
                />
            )}
        </View>
    );
};

interface VariantConfig {
    icon: keyof typeof Ionicons.glyphMap;
    iconColor: string;
    iconBg: string;
}

const getVariantConfig = (variant: EmptyStateVariant): VariantConfig => {
    switch (variant) {
        case 'search':
            return {
                icon: 'search',
                iconColor: colors.text.secondary,
                iconBg: colors.surfaceHighlight,
            };
        case 'error':
            return {
                icon: 'alert-circle',
                iconColor: colors.error,
                iconBg: `${colors.error}15`,
            };
        case 'wardrobe':
            return {
                icon: 'shirt-outline',
                iconColor: colors.text.accent,
                iconBg: `${colors.text.accent}20`,
            };
        case 'outfit':
            return {
                icon: 'layers-outline',
                iconColor: colors.text.accent,
                iconBg: `${colors.text.accent}20`,
            };
        default:
            return {
                icon: 'folder-open-outline',
                iconColor: colors.text.secondary,
                iconBg: colors.surfaceHighlight,
            };
    }
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        padding: spacing.xl,
    },
    iconContainer: {
        width: 100,
        height: 100,
        borderRadius: 50,
        justifyContent: 'center',
        alignItems: 'center',
        marginBottom: spacing.l,
    },
    image: {
        width: 150,
        height: 150,
        marginBottom: spacing.l,
    },
    title: {
        fontSize: 20,
        fontWeight: '700',
        color: colors.text.primary,
        textAlign: 'center',
        marginBottom: spacing.s,
    },
    description: {
        fontSize: 15,
        color: colors.text.secondary,
        textAlign: 'center',
        lineHeight: 22,
        maxWidth: 280,
    },
    actionButton: {
        marginTop: spacing.l,
        minWidth: 160,
    },
});

export default EmptyState;
