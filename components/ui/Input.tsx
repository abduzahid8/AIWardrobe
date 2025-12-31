import React, { useState } from 'react';
import {
    View,
    TextInput,
    Text,
    TextInputProps,
    StyleSheet,
    ViewStyle,
    TextStyle,
    TouchableOpacity,
} from 'react-native';
import Animated, {
    useSharedValue,
    useAnimatedStyle,
    withTiming,
    interpolateColor,
} from 'react-native-reanimated';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, animations } from '../../src/theme';

type InputVariant = 'default' | 'filled' | 'outline';
type InputState = 'default' | 'focused' | 'error' | 'success';

interface InputProps extends Omit<TextInputProps, 'style'> {
    label?: string;
    error?: string;
    hint?: string;
    variant?: InputVariant;
    leftIcon?: keyof typeof Ionicons.glyphMap;
    rightIcon?: keyof typeof Ionicons.glyphMap;
    onRightIconPress?: () => void;
    containerStyle?: ViewStyle;
    inputStyle?: TextStyle;
    isPassword?: boolean;
}

const AnimatedView = Animated.createAnimatedComponent(View);

/**
 * Styled Input component with label, icons, and validation states
 * 
 * @example
 * <Input
 *   label="Email"
 *   placeholder="Enter your email"
 *   leftIcon="mail"
 *   error={errors.email}
 * />
 * 
 * @example
 * <Input
 *   label="Password"
 *   isPassword
 *   leftIcon="lock-closed"
 * />
 */
export const Input: React.FC<InputProps> = ({
    label,
    error,
    hint,
    variant = 'default',
    leftIcon,
    rightIcon,
    onRightIconPress,
    containerStyle,
    inputStyle,
    isPassword = false,
    ...textInputProps
}) => {
    const [isFocused, setIsFocused] = useState(false);
    const [isPasswordVisible, setIsPasswordVisible] = useState(!isPassword);
    const focusProgress = useSharedValue(0);

    const state: InputState = error ? 'error' : isFocused ? 'focused' : 'default';

    const handleFocus = () => {
        setIsFocused(true);
        focusProgress.value = withTiming(1, { duration: animations.timing.fast });
        textInputProps.onFocus?.({} as any);
    };

    const handleBlur = () => {
        setIsFocused(false);
        focusProgress.value = withTiming(0, { duration: animations.timing.fast });
        textInputProps.onBlur?.({} as any);
    };

    const animatedBorderStyle = useAnimatedStyle(() => {
        const borderColor = interpolateColor(
            focusProgress.value,
            [0, 1],
            [error ? colors.error : colors.border, error ? colors.error : colors.text.accent]
        );

        return {
            borderColor,
        };
    });

    const togglePasswordVisibility = () => {
        setIsPasswordVisible(!isPasswordVisible);
    };

    const getVariantStyles = (): ViewStyle => {
        switch (variant) {
            case 'filled':
                return {
                    backgroundColor: colors.surfaceHighlight,
                    borderWidth: 0,
                };
            case 'outline':
                return {
                    backgroundColor: 'transparent',
                    borderWidth: 1.5,
                };
            default:
                return {
                    backgroundColor: colors.surface,
                    borderWidth: 1,
                };
        }
    };

    return (
        <View style={[styles.container, containerStyle]}>
            {label && <Text style={styles.label}>{label}</Text>}

            <AnimatedView style={[styles.inputContainer, getVariantStyles(), animatedBorderStyle]}>
                {leftIcon && (
                    <Ionicons
                        name={leftIcon}
                        size={20}
                        color={state === 'error' ? colors.error : colors.text.secondary}
                        style={styles.leftIcon}
                    />
                )}

                <TextInput
                    {...textInputProps}
                    style={[
                        styles.input,
                        leftIcon && styles.inputWithLeftIcon,
                        (rightIcon || isPassword) && styles.inputWithRightIcon,
                        inputStyle,
                    ]}
                    placeholderTextColor={colors.text.secondary}
                    onFocus={handleFocus}
                    onBlur={handleBlur}
                    secureTextEntry={isPassword && !isPasswordVisible}
                />

                {isPassword ? (
                    <TouchableOpacity onPress={togglePasswordVisibility} style={styles.rightIcon}>
                        <Ionicons
                            name={isPasswordVisible ? 'eye-off' : 'eye'}
                            size={20}
                            color={colors.text.secondary}
                        />
                    </TouchableOpacity>
                ) : rightIcon ? (
                    <TouchableOpacity
                        onPress={onRightIconPress}
                        style={styles.rightIcon}
                        disabled={!onRightIconPress}
                    >
                        <Ionicons
                            name={rightIcon}
                            size={20}
                            color={colors.text.secondary}
                        />
                    </TouchableOpacity>
                ) : null}
            </AnimatedView>

            {error && <Text style={styles.error}>{error}</Text>}
            {hint && !error && <Text style={styles.hint}>{hint}</Text>}
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        marginBottom: spacing.m,
    },
    label: {
        fontSize: 14,
        fontWeight: '600',
        color: colors.text.primary,
        marginBottom: spacing.xs,
    },
    inputContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        borderRadius: 12,
        minHeight: 52,
        borderColor: colors.border,
    },
    input: {
        flex: 1,
        fontSize: 16,
        color: colors.text.primary,
        paddingVertical: spacing.m,
        paddingHorizontal: spacing.m,
    },
    inputWithLeftIcon: {
        paddingLeft: spacing.xs,
    },
    inputWithRightIcon: {
        paddingRight: spacing.xs,
    },
    leftIcon: {
        marginLeft: spacing.m,
    },
    rightIcon: {
        marginRight: spacing.m,
        padding: spacing.xs,
    },
    error: {
        fontSize: 12,
        color: colors.error,
        marginTop: spacing.xs,
    },
    hint: {
        fontSize: 12,
        color: colors.text.secondary,
        marginTop: spacing.xs,
    },
});

export default Input;
