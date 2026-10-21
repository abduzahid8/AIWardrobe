import React, { useEffect, useState } from 'react';
import {
    View,
    StyleSheet,
    ImageStyle,
    StyleProp,
} from 'react-native';
import { ImageProps } from 'expo-image';
import Animated from 'react-native-reanimated';
import { Ionicons } from '@expo/vector-icons';
import { colors } from '../../src/theme';
import { ExpoImage as Image } from '../../src/components/MediaImage';


interface CachedImageProps extends Omit<ImageProps, 'source'> {
    uri: string;
    /** @deprecated No longer fetched — a broken/missing image now renders a
     * local icon placeholder instead of a remote fallback URL (that remote
     * service was unreliable and left cards blank with no indication of
     * failure). Kept only so existing call sites don't break at compile time. */
    fallbackUri?: string;
    showLoader?: boolean;
    fadeIn?: boolean;
    style?: StyleProp<ImageStyle>;
    /** Icon shown when `uri` is missing or fails to load. */
    fallbackIconSize?: number;
    fallbackIconColor?: string;
    requestHeaders?: Record<string, string>;
}

export const CachedImage: React.FC<CachedImageProps> = React.memo(({
    uri,
    showLoader = true,
    fadeIn = true,
    style,
    contentFit = 'cover',
    fallbackIconSize = 28,
    fallbackIconColor = 'rgba(10,25,49,0.2)',
    requestHeaders,
    cachePolicy = 'memory-disk',
    ...props
}) => {
    const [error, setError] = useState(false);
    // Phones on flaky mobile data often fail the first attempt with a
    // transient timeout while the same URL loads fine on retry (simulators
    // on stable Wi-Fi rarely hit this). Retry once after a short delay
    // before settling on the placeholder — without this a single blip
    // permanently blanks the card until the row is recycled.
    const [retried, setRetried] = useState(false);

    // A list reuses this component instance across items as `uri` changes
    // (e.g. FlatList row recycling); without this the failed-load flag from
    // a previous item's broken photo would stick and blank out the next
    // item's perfectly good one.
    useEffect(() => {
        setError(false);
        setRetried(false);
    }, [uri, requestHeaders]);

    useEffect(() => {
        if (!error || retried || !uri) return;
        // Clear the error when the timer fires so the retry <Image> below
        // actually remounts (key change) and re-requests the URL. Without
        // this, `error && retried` renders the fallback without ever
        // attempting the second fetch.
        const t = setTimeout(() => {
            setError(false);
            setRetried(true);
        }, 2500);
        return () => clearTimeout(t);
    }, [error, retried, uri]);

    if (!uri || (error && retried)) {
        return (
            <View style={[styles.container, styles.fallback, style as any]}>
                <Ionicons name="shirt-outline" size={fallbackIconSize} color={fallbackIconColor} />
            </View>
        );
    }

    return (
        <View style={[styles.container, style as any]}>
            <Image
                {...props}
                source={requestHeaders ? { uri, headers: requestHeaders } : uri}
                key={retried ? `${uri}#retry` : uri}
                style={[styles.image, style]}
                contentFit={contentFit}
                cachePolicy={cachePolicy}
                recyclingKey={uri}
                transition={fadeIn ? 200 : undefined}
                onError={() => setError(true)}
            />
        </View>
    );
}, (prev, next) => prev.uri === next.uri && prev.style === next.style && prev.contentFit === next.contentFit
    && prev.requestHeaders === next.requestHeaders && prev.cachePolicy === next.cachePolicy);

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
    fallback: {
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: 'rgba(10,25,49,0.04)',
    },
    loaderContainer: {
        ...StyleSheet.absoluteFill,
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
        ...StyleSheet.absoluteFill,
    },
    optimizedImage: {
        ...StyleSheet.absoluteFill,
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
