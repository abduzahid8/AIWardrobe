import React from 'react';
import {
    View,
    Text,
    TouchableOpacity,
    StyleSheet,
    ViewStyle,
    Platform,
    StatusBar,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { BlurView } from 'expo-blur';
import * as Haptics from 'expo-haptics';
import { colors, spacing } from '../../src/theme';

interface HeaderProps {
    title?: string;
    subtitle?: string;
    showBackButton?: boolean;
    onBackPress?: () => void;
    rightIcon?: keyof typeof Ionicons.glyphMap;
    onRightPress?: () => void;
    rightComponent?: React.ReactNode;
    transparent?: boolean;
    blur?: boolean;
    style?: ViewStyle;
    centerTitle?: boolean;
}

/**
 * Reusable screen header component
 * 
 * @example
 * <Header
 *   title="My Wardrobe"
 *   showBackButton
 *   onBackPress={() => navigation.goBack()}
 *   rightIcon="settings"
 *   onRightPress={() => openSettings()}
 * />
 */
export const Header: React.FC<HeaderProps> = ({
    title,
    subtitle,
    showBackButton = false,
    onBackPress,
    rightIcon,
    onRightPress,
    rightComponent,
    transparent = false,
    blur = false,
    style,
    centerTitle = false,
}) => {
    const insets = useSafeAreaInsets();

    const handleBackPress = () => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        onBackPress?.();
    };

    const handleRightPress = () => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        onRightPress?.();
    };

    const headerContent = (
        <>
            <View style={styles.leftContainer}>
                {showBackButton && (
                    <TouchableOpacity
                        onPress={handleBackPress}
                        style={styles.iconButton}
                        hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
                    >
                        <Ionicons name="chevron-back" size={28} color={colors.text.primary} />
                    </TouchableOpacity>
                )}
            </View>

            <View style={[styles.centerContainer, centerTitle && styles.centerTitleContainer]}>
                {title && (
                    <Text style={[styles.title, centerTitle && styles.centeredTitle]} numberOfLines={1}>
                        {title}
                    </Text>
                )}
                {subtitle && <Text style={styles.subtitle} numberOfLines={1}>{subtitle}</Text>}
            </View>

            <View style={styles.rightContainer}>
                {rightComponent || (rightIcon && (
                    <TouchableOpacity
                        onPress={handleRightPress}
                        style={styles.iconButton}
                        hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
                    >
                        <Ionicons name={rightIcon} size={24} color={colors.text.primary} />
                    </TouchableOpacity>
                ))}
            </View>
        </>
    );

    const containerStyle = [
        styles.container,
        { paddingTop: insets.top + spacing.s },
        transparent && styles.transparent,
        style,
    ];

    if (blur && Platform.OS === 'ios') {
        return (
            <BlurView intensity={80} tint="light" style={containerStyle}>
                <View style={styles.content}>{headerContent}</View>
            </BlurView>
        );
    }

    return (
        <View style={[containerStyle, !transparent && styles.defaultBackground]}>
            <View style={styles.content}>{headerContent}</View>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        width: '100%',
    },
    defaultBackground: {
        backgroundColor: colors.background,
        borderBottomWidth: StyleSheet.hairlineWidth,
        borderBottomColor: colors.border,
    },
    transparent: {
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        zIndex: 100,
    },
    content: {
        flexDirection: 'row',
        alignItems: 'center',
        paddingHorizontal: spacing.m,
        paddingBottom: spacing.m,
        minHeight: 44,
    },
    leftContainer: {
        width: 44,
        alignItems: 'flex-start',
    },
    centerContainer: {
        flex: 1,
        paddingHorizontal: spacing.s,
    },
    centerTitleContainer: {
        alignItems: 'center',
    },
    rightContainer: {
        width: 44,
        alignItems: 'flex-end',
    },
    iconButton: {
        width: 44,
        height: 44,
        justifyContent: 'center',
        alignItems: 'center',
    },
    title: {
        fontSize: 20,
        fontWeight: '700',
        color: colors.text.primary,
    },
    centeredTitle: {
        textAlign: 'center',
    },
    subtitle: {
        fontSize: 13,
        color: colors.text.secondary,
        marginTop: 2,
    },
});

export default Header;
