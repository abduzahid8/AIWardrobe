import React, { useState } from 'react';
import {
    View,
    StyleSheet,
    ImageStyle,
    StyleProp,
} from 'react-native';
import { Image, ImageProps } from 'expo-image';
import Animated from 'react-native-reanimated';
import { colors } from '../../src/theme';

interface CachedImageProps extends Omit<ImageProps, 'source'> {
    uri: string;
    fallbackUri?: string;
    showLoader?: boolean;
    fadeIn?: boolean;
    style?: StyleProp<ImageStyle>;
}

export const CachedImage: React.FC<CachedImageProps> = React.memo(({
    uri,
    fallbackUri = 'https://via.placeholder.com/150',
    showLoader = true,
    fadeIn = true,
    style,
    contentFit = 'cover',
    ...props
}) => {
    const [error, setError] = useState(false);
    const sourceUri = error ? fallbackUri : uri;

    const handleError = () => {
        setError(true);
    };

    return (
        <View style={[styles.container, style as any]}>
            <Image
                {...props}
                source={sourceUri}
                style={[styles.image, style]}
                contentFit={contentFit}
                cachePolicy="memory-disk"
                transition={fadeIn ? 200 : undefined}
                onError={handleError}
            />
        </View>
    );
}, (prev, next) => prev.uri === next.uri && prev.style === next.style && prev.contentFit === next.contentFit);

// Optimized image for lists with placeholder
interface OptimizedImageProps extends CachedImageProps {
    aspectRatio?: number;
    placeholderColor?: string;
}

export const OptimizedImage: React.FC<OptimizedImageProps> = React.memo(({
    uri,
    aspectRatio = 1,
    placeholderColor = colors.surfaceHighlight,
    style,
    ...props
}) => {
    return (
        <View style={[styles.optimizedContainer, { aspectRatio }, style as any]}>
            {/* Placeholder */}
            <View style={[styles.placeholder, { backgroundColor: placeholderColor }]} />

            {/* Actual image */}
            <Image
                {...props}
                source={uri}
                style={[styles.optimizedImage]}
                contentFit="cover"
                cachePolicy="memory-disk"
                transition={200}
            />
        </View>
    );
});

// Avatar component with caching
interface CachedAvatarProps {
    uri?: string;
    size?: number;
    fallbackInitials?: string;
    style?: StyleProp<ImageStyle>;
}

export const CachedAvatar: React.FC<CachedAvatarProps> = React.memo(({
    uri,
    size = 40,
    fallbackInitials = '?',
    style,
}) => {
    const [error, setError] = useState(false);
    const showFallback = !uri || error;

    return (
        <View style={[styles.avatar, { width: size, height: size, borderRadius: size / 2 }, style as any]}>
            {showFallback ? (
                <View style={[styles.avatarFallback, { width: size, height: size, borderRadius: size / 2 }]}>
                    <Animated.Text style={[styles.avatarText, { fontSize: size * 0.4 }]}>
                        {fallbackInitials.substring(0, 2).toUpperCase()}
                    </Animated.Text>
                </View>
            ) : (
                <Image
                    source={uri}
                    style={[styles.avatarImage, { width: size, height: size, borderRadius: size / 2 }]}
                    contentFit="cover"
                    cachePolicy="memory-disk"
                    onError={() => setError(true)}
                />
            )}
        </View>
    );
});

const styles = StyleSheet.create({
    container: {
        overflow: 'hidden',
    },
    loaderContainer: {
        ...StyleSheet.absoluteFillObject,
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: colors.surfaceHighlight,
    },
    image: {
        width: '100%',
        height: '100%',
    },
    optimizedContainer: {
        overflow: 'hidden',
        backgroundColor: colors.surfaceHighlight,
    },
    placeholder: {
        ...StyleSheet.absoluteFillObject,
    },
    optimizedImage: {
        ...StyleSheet.absoluteFillObject,
        width: '100%',
        height: '100%',
    },
    avatar: {
        overflow: 'hidden',
    },
    avatarFallback: {
        backgroundColor: colors.text.accent,
        justifyContent: 'center',
        alignItems: 'center',
    },
    avatarText: {
        color: '#FFF',
        fontWeight: '600',
    },
    avatarImage: {
        width: '100%',
        height: '100%',
    },
});

export default CachedImage;
