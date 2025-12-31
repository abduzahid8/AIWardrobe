import React, { useState, useEffect } from 'react';
import {
    Image,
    ImageProps,
    View,
    StyleSheet,
    ActivityIndicator,
    ImageStyle,
    StyleProp,
} from 'react-native';
import Animated, {
    useSharedValue,
    useAnimatedStyle,
    withTiming,
    interpolate,
} from 'react-native-reanimated';
import { useCachedImage } from '../../src/utils/imageCache';
import { colors, borderRadius } from '../../src/theme';

interface CachedImageProps extends Omit<ImageProps, 'source'> {
    uri: string;
    fallbackUri?: string;
    showLoader?: boolean;
    fadeIn?: boolean;
    style?: StyleProp<ImageStyle>;
}

const AnimatedImage = Animated.createAnimatedComponent(Image);

export const CachedImage: React.FC<CachedImageProps> = ({
    uri,
    fallbackUri = 'https://via.placeholder.com/150',
    showLoader = true,
    fadeIn = true,
    style,
    ...props
}) => {
    const { cachedUri, loading: cacheLoading } = useCachedImage(uri);
    const [imageLoading, setImageLoading] = useState(true);
    const [error, setError] = useState(false);

    const opacity = useSharedValue(0);

    const animatedStyle = useAnimatedStyle(() => ({
        opacity: fadeIn ? opacity.value : 1,
    }));

    const handleLoad = () => {
        setImageLoading(false);
        if (fadeIn) {
            opacity.value = withTiming(1, { duration: 300 });
        } else {
            opacity.value = 1;
        }
    };

    const handleError = () => {
        setError(true);
        setImageLoading(false);
    };

    useEffect(() => {
        // Reset state when uri changes
        setError(false);
        setImageLoading(true);
        opacity.value = 0;
    }, [uri]);

    const isLoading = cacheLoading || imageLoading;
    const sourceUri = error ? fallbackUri : cachedUri;

    return (
        <View style={[styles.container, style as any]}>
            {/* Loading indicator */}
            {showLoader && isLoading && (
                <View style={styles.loaderContainer}>
                    <ActivityIndicator size="small" color={colors.text.muted} />
                </View>
            )}

            {/* Image */}
            <AnimatedImage
                {...props}
                source={{ uri: sourceUri }}
                style={[styles.image, style, animatedStyle]}
                onLoad={handleLoad}
                onError={handleError}
            />
        </View>
    );
};

// Optimized image for lists with blurhash placeholder
interface OptimizedImageProps extends CachedImageProps {
    aspectRatio?: number;
    placeholderColor?: string;
}

export const OptimizedImage: React.FC<OptimizedImageProps> = ({
    uri,
    aspectRatio = 1,
    placeholderColor = colors.surfaceHighlight,
    style,
    ...props
}) => {
    const { cachedUri, loading: cacheLoading } = useCachedImage(uri);
    const [loaded, setLoaded] = useState(false);

    const opacity = useSharedValue(0);

    const animatedStyle = useAnimatedStyle(() => ({
        opacity: opacity.value,
    }));

    const handleLoad = () => {
        setLoaded(true);
        opacity.value = withTiming(1, { duration: 200 });
    };

    return (
        <View style={[styles.optimizedContainer, { aspectRatio }, style as any]}>
            {/* Placeholder */}
            <View style={[styles.placeholder, { backgroundColor: placeholderColor }]} />

            {/* Actual image */}
            {!cacheLoading && (
                <AnimatedImage
                    {...props}
                    source={{ uri: cachedUri }}
                    style={[styles.optimizedImage, animatedStyle]}
                    onLoad={handleLoad}
                />
            )}
        </View>
    );
};

// Avatar component with caching
interface CachedAvatarProps {
    uri?: string;
    size?: number;
    fallbackInitials?: string;
    style?: StyleProp<ImageStyle>;
}

export const CachedAvatar: React.FC<CachedAvatarProps> = ({
    uri,
    size = 40,
    fallbackInitials = '?',
    style,
}) => {
    const { cachedUri, loading } = useCachedImage(uri || '');
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
                    source={{ uri: cachedUri }}
                    style={[styles.avatarImage, { width: size, height: size, borderRadius: size / 2 }]}
                    onError={() => setError(true)}
                />
            )}
        </View>
    );
};

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
