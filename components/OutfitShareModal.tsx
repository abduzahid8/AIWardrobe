import React, { useState, useRef } from 'react';
import {
    View,
    Text,
    StyleSheet,
    Modal,
    TouchableOpacity,
    Share,
    Image,
    Dimensions,
    TextInput,
    ScrollView,
    Alert,
    ActivityIndicator,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import * as MediaLibrary from 'expo-media-library';
import ViewShot from 'react-native-view-shot';
import { LinearGradient } from 'expo-linear-gradient';
import AppColors from '../constants/AppColors';

const { width } = Dimensions.get('window');

const COLORS = {
    background: AppColors.background,
    surface: AppColors.surface,
    primary: AppColors.primary,
    accent: AppColors.accent,
    text: AppColors.text,
    textSecondary: AppColors.textSecondary,
    border: AppColors.border,
};

// Social platforms for sharing
const SHARE_OPTIONS = [
    { id: 'instagram', name: 'Instagram', icon: 'logo-instagram', color: '#E4405F' },
    { id: 'whatsapp', name: 'WhatsApp', icon: 'logo-whatsapp', color: '#25D366' },
    { id: 'twitter', name: 'X / Twitter', icon: 'logo-twitter', color: '#000000' },
    { id: 'facebook', name: 'Facebook', icon: 'logo-facebook', color: '#1877F2' },
    { id: 'pinterest', name: 'Pinterest', icon: 'logo-pinterest', color: '#E60023' },
    { id: 'copy', name: 'Copy Link', icon: 'link-outline', color: COLORS.textSecondary },
    { id: 'more', name: 'More', icon: 'share-outline', color: COLORS.textSecondary },
];

// Emoji reactions for friend feedback
const REACTIONS = ['😍', '🔥', '👍', '👎', '🤔', '💯'];

interface OutfitShareModalProps {
    visible: boolean;
    onClose: () => void;
    outfit: {
        id: string;
        items: any[];
        occasion?: string;
        style?: string;
    };
}

export const OutfitShareModal = ({ visible, onClose, outfit }: OutfitShareModalProps) => {
    const [caption, setCaption] = useState('');
    const [isSharing, setIsSharing] = useState(false);
    const viewShotRef = useRef<any>(null);

    const handleShare = async (platform: string) => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
        setIsSharing(true);

        try {
            // Capture the outfit card as an image
            if (viewShotRef.current) {
                const uri = await viewShotRef.current.capture();

                if (platform === 'copy') {
                    // Copy link to clipboard (would be a real link in production)
                    const outfitLink = `https://aiwardrobe.app/outfit/${outfit.id}`;
                    // Clipboard.setString(outfitLink);
                    Alert.alert('Link Copied!', 'Outfit link copied to clipboard');
                } else if (platform === 'more' || platform === 'native') {
                    // Use native share sheet
                    const result = await Share.share({
                        message: caption || `Check out this outfit I created with AIWardrobe! 👗✨`,
                        url: uri,
                        title: 'My Outfit',
                    });

                    if (result.action === Share.sharedAction) {
                        onClose();
                    }
                } else {
                    // Platform-specific sharing would go here
                    // For now, use native share
                    await Share.share({
                        message: caption || `Check out this outfit I created! 👗✨ #AIWardrobe #OOTD`,
                        url: uri,
                    });
                }
            }
        } catch (error) {
            console.error('Share error:', error);
            Alert.alert('Error', 'Failed to share outfit');
        }

        setIsSharing(false);
    };

    const handleSaveToGallery = async () => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);

        try {
            const { status } = await MediaLibrary.requestPermissionsAsync();
            if (status !== 'granted') {
                Alert.alert('Permission needed', 'Please allow access to save images');
                return;
            }

            if (viewShotRef.current) {
                const uri = await viewShotRef.current.capture();
                await MediaLibrary.saveToLibraryAsync(uri);
                Alert.alert('Saved!', 'Outfit saved to your gallery');
            }
        } catch (error) {
            console.error('Save error:', error);
            Alert.alert('Error', 'Failed to save outfit');
        }
    };

    return (
        <Modal
            visible={visible}
            animationType="slide"
            presentationStyle="pageSheet"
            onRequestClose={onClose}
        >
            <View style={styles.modalContainer}>
                {/* Header */}
                <View style={styles.modalHeader}>
                    <TouchableOpacity onPress={onClose}>
                        <Text style={styles.cancelText}>Cancel</Text>
                    </TouchableOpacity>
                    <Text style={styles.modalTitle}>Share Outfit</Text>
                    <View style={{ width: 50 }} />
                </View>

                <ScrollView style={styles.modalContent}>
                    {/* Outfit Preview (Capturable) */}
                    <ViewShot
                        ref={viewShotRef}
                        options={{ format: 'png', quality: 1 }}
                        style={styles.outfitPreviewCapture}
                    >
                        <LinearGradient
                            colors={[COLORS.background, COLORS.surface]}
                            style={styles.previewGradient}
                        >
                            <View style={styles.previewItemsRow}>
                                {outfit.items.slice(0, 3).map((item, idx) => (
                                    <View key={idx} style={styles.previewItem}>
                                        {item.image || item.imageUrl ? (
                                            <Image
                                                source={{ uri: item.image || item.imageUrl }}
                                                style={styles.previewItemImage}
                                                resizeMode="cover"
                                            />
                                        ) : (
                                            <View style={styles.previewItemPlaceholder}>
                                                <Ionicons name="shirt" size={32} color={COLORS.textSecondary} />
                                            </View>
                                        )}
                                    </View>
                                ))}
                            </View>

                            {outfit.occasion && (
                                <View style={styles.previewBadge}>
                                    <Text style={styles.previewBadgeText}>
                                        {outfit.occasion.toUpperCase()}
                                    </Text>
                                </View>
                            )}

                            <View style={styles.watermark}>
                                <Text style={styles.watermarkText}>✨ AIWardrobe</Text>
                            </View>
                        </LinearGradient>
                    </ViewShot>

                    {/* Caption Input */}
                    <View style={styles.captionContainer}>
                        <TextInput
                            style={styles.captionInput}
                            placeholder="Add a caption..."
                            placeholderTextColor={COLORS.textSecondary}
                            value={caption}
                            onChangeText={setCaption}
                            multiline
                            maxLength={200}
                        />
                        <Text style={styles.charCount}>{caption.length}/200</Text>
                    </View>

                    {/* Save to Gallery */}
                    <TouchableOpacity
                        style={styles.saveButton}
                        onPress={handleSaveToGallery}
                    >
                        <Ionicons name="download-outline" size={20} color={COLORS.primary} />
                        <Text style={styles.saveButtonText}>Save to Gallery</Text>
                    </TouchableOpacity>

                    {/* Share Options */}
                    <Text style={styles.sectionTitle}>Share to</Text>
                    <View style={styles.shareGrid}>
                        {SHARE_OPTIONS.map((option) => (
                            <TouchableOpacity
                                key={option.id}
                                style={styles.shareOption}
                                onPress={() => handleShare(option.id)}
                                disabled={isSharing}
                            >
                                <View style={[styles.shareIconContainer, { backgroundColor: `${option.color}15` }]}>
                                    <Ionicons
                                        name={option.icon as any}
                                        size={28}
                                        color={option.color}
                                    />
                                </View>
                                <Text style={styles.shareName}>{option.name}</Text>
                            </TouchableOpacity>
                        ))}
                    </View>

                    {/* Send to Friends */}
                    <Text style={styles.sectionTitle}>Get Friend Feedback</Text>
                    <View style={styles.friendFeedbackInfo}>
                        <Ionicons name="people" size={24} color={COLORS.primary} />
                        <Text style={styles.friendFeedbackText}>
                            Share with friends and let them vote with reactions!
                        </Text>
                    </View>

                    <View style={styles.reactionsPreview}>
                        {REACTIONS.map((emoji, idx) => (
                            <Text key={idx} style={styles.reactionEmoji}>{emoji}</Text>
                        ))}
                    </View>
                </ScrollView>

                {/* Loading overlay */}
                {isSharing && (
                    <View style={styles.loadingOverlay}>
                        <ActivityIndicator size="large" color={COLORS.primary} />
                        <Text style={styles.loadingText}>Preparing to share...</Text>
                    </View>
                )}
            </View>
        </Modal>
    );
};

// ============================================
// FRIEND FEEDBACK COMPONENT
// ============================================

interface FriendFeedbackProps {
    outfitId: string;
    reactions: { emoji: string; count: number; users: string[] }[];
    onReact: (emoji: string) => void;
    myReaction?: string;
}

export const FriendFeedback = ({ outfitId, reactions, onReact, myReaction }: FriendFeedbackProps) => {
    return (
        <View style={styles.feedbackContainer}>
            <Text style={styles.feedbackTitle}>Friend Reactions</Text>

            <View style={styles.reactionsRow}>
                {REACTIONS.map((emoji) => {
                    const reactionData = reactions.find(r => r.emoji === emoji);
                    const count = reactionData?.count || 0;
                    const isSelected = myReaction === emoji;

                    return (
                        <TouchableOpacity
                            key={emoji}
                            style={[
                                styles.reactionButton,
                                isSelected && styles.reactionButtonSelected,
                                count > 0 && styles.reactionButtonActive
                            ]}
                            onPress={() => {
                                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                                onReact(emoji);
                            }}
                        >
                            <Text style={styles.reactionButtonEmoji}>{emoji}</Text>
                            {count > 0 && (
                                <Text style={styles.reactionCount}>{count}</Text>
                            )}
                        </TouchableOpacity>
                    );
                })}
            </View>
        </View>
    );
};

// ============================================
// STYLES
// ============================================

const styles = StyleSheet.create({
    modalContainer: {
        flex: 1,
        backgroundColor: COLORS.background,
    },
    modalHeader: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: 16,
        borderBottomWidth: 1,
        borderBottomColor: COLORS.border,
    },
    cancelText: {
        fontSize: 16,
        color: COLORS.textSecondary,
    },
    modalTitle: {
        fontSize: 18,
        fontWeight: '600',
        color: COLORS.text,
    },
    modalContent: {
        flex: 1,
        padding: 16,
    },

    // Outfit Preview
    outfitPreviewCapture: {
        borderRadius: 20,
        overflow: 'hidden',
        marginBottom: 16,
    },
    previewGradient: {
        padding: 20,
        alignItems: 'center',
    },
    previewItemsRow: {
        flexDirection: 'row',
        gap: 12,
        marginBottom: 16,
    },
    previewItem: {
        width: (width - 80) / 3,
        aspectRatio: 0.75,
        borderRadius: 12,
        overflow: 'hidden',
        backgroundColor: COLORS.surface,
    },
    previewItemImage: {
        width: '100%',
        height: '100%',
    },
    previewItemPlaceholder: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
    },
    previewBadge: {
        backgroundColor: COLORS.primary,
        paddingHorizontal: 16,
        paddingVertical: 6,
        borderRadius: 16,
        marginBottom: 8,
    },
    previewBadgeText: {
        fontSize: 12,
        fontWeight: '600',
        color: '#FFF',
        letterSpacing: 1,
    },
    watermark: {
        marginTop: 8,
    },
    watermarkText: {
        fontSize: 14,
        color: COLORS.textSecondary,
    },

    // Caption
    captionContainer: {
        backgroundColor: COLORS.surface,
        borderRadius: 16,
        padding: 16,
        marginBottom: 16,
    },
    captionInput: {
        fontSize: 16,
        color: COLORS.text,
        minHeight: 60,
    },
    charCount: {
        fontSize: 12,
        color: COLORS.textSecondary,
        textAlign: 'right',
        marginTop: 8,
    },

    // Save Button
    saveButton: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: COLORS.surface,
        padding: 14,
        borderRadius: 12,
        gap: 8,
        marginBottom: 24,
    },
    saveButtonText: {
        fontSize: 16,
        fontWeight: '500',
        color: COLORS.primary,
    },

    // Share Options
    sectionTitle: {
        fontSize: 16,
        fontWeight: '600',
        color: COLORS.text,
        marginBottom: 16,
    },
    shareGrid: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        gap: 16,
        marginBottom: 24,
    },
    shareOption: {
        width: (width - 80) / 4,
        alignItems: 'center',
    },
    shareIconContainer: {
        width: 56,
        height: 56,
        borderRadius: 28,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 8,
    },
    shareName: {
        fontSize: 12,
        color: COLORS.textSecondary,
        textAlign: 'center',
    },

    // Friend Feedback Info
    friendFeedbackInfo: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: COLORS.surface,
        padding: 16,
        borderRadius: 16,
        gap: 12,
        marginBottom: 16,
    },
    friendFeedbackText: {
        flex: 1,
        fontSize: 14,
        color: COLORS.textSecondary,
    },
    reactionsPreview: {
        flexDirection: 'row',
        justifyContent: 'center',
        gap: 8,
        marginBottom: 24,
    },
    reactionEmoji: {
        fontSize: 28,
    },

    // Loading
    loadingOverlay: {
        ...StyleSheet.absoluteFillObject,
        backgroundColor: 'rgba(0,0,0,0.5)',
        alignItems: 'center',
        justifyContent: 'center',
    },
    loadingText: {
        marginTop: 12,
        fontSize: 16,
        color: '#FFF',
    },

    // Friend Feedback Component
    feedbackContainer: {
        backgroundColor: COLORS.surface,
        borderRadius: 16,
        padding: 16,
    },
    feedbackTitle: {
        fontSize: 14,
        fontWeight: '600',
        color: COLORS.text,
        marginBottom: 12,
    },
    reactionsRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
    },
    reactionButton: {
        alignItems: 'center',
        padding: 8,
        borderRadius: 12,
        backgroundColor: COLORS.background,
        minWidth: 48,
    },
    reactionButtonSelected: {
        backgroundColor: `${COLORS.primary}20`,
        borderWidth: 2,
        borderColor: COLORS.primary,
    },
    reactionButtonActive: {
        backgroundColor: `${COLORS.accent}10`,
    },
    reactionButtonEmoji: {
        fontSize: 24,
    },
    reactionCount: {
        fontSize: 12,
        fontWeight: '600',
        color: COLORS.textSecondary,
        marginTop: 2,
    },
});

export default OutfitShareModal;
