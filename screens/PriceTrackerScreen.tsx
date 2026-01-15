import React, { useState, useEffect } from 'react';
import {
    View,
    Text,
    TextInput,
    TouchableOpacity,
    ScrollView,
    Image,
    StyleSheet,
    Dimensions,
    Alert,
    KeyboardAvoidingView,
    Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import * as Haptics from 'expo-haptics';
import Animated, {
    FadeIn,
    FadeInUp,
    FadeInDown,
    useAnimatedStyle,
    useSharedValue,
    withSpring,
} from 'react-native-reanimated';
import AppColors from '../constants/AppColors';
import usePriceTrackingStore, { TrackedItem, PriceAlert } from '../store/priceTrackingStore';

const { width } = Dimensions.get('window');

// New item input type for adding tracked items
interface NewItemInput {
    name: string;
    brand: string;
    currentPrice: number;
    currency: string;
    imageUrl: string;
    category: string;
    targetPrice?: number;
}

// Price trend indicator
const PriceTrend = ({ history }: { history: { price: number }[] }) => {
    if (history.length < 2) return null;

    const current = history[0].price;
    const previous = history[1].price;
    const isDown = current < previous;
    const change = Math.abs(Math.round((1 - current / previous) * 100));

    return (
        <View style={[styles.trendBadge, isDown ? styles.trendDown : styles.trendUp]}>
            <Ionicons
                name={isDown ? "trending-down" : "trending-up"}
                size={12}
                color={isDown ? "#34C759" : "#FF3B30"}
            />
            <Text style={[styles.trendText, isDown ? styles.trendTextDown : styles.trendTextUp]}>
                {change}%
            </Text>
        </View>
    );
};

// Tracked item card
const TrackedItemCard = ({
    item,
    onPress,
    onRemove
}: {
    item: TrackedItem;
    onPress: () => void;
    onRemove: () => void;
}) => {
    const scale = useSharedValue(1);

    const animatedStyle = useAnimatedStyle(() => ({
        transform: [{ scale: scale.value }],
    }));

    return (
        <Animated.View entering={FadeInUp.springify()} style={animatedStyle}>
            <TouchableOpacity
                style={styles.itemCard}
                onPress={onPress}
                onLongPress={() => {
                    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
                    Alert.alert(
                        "Remove Item",
                        `Stop tracking ${item.name}?`,
                        [
                            { text: "Cancel", style: "cancel" },
                            { text: "Remove", style: "destructive", onPress: onRemove }
                        ]
                    );
                }}
                activeOpacity={0.9}
            >
                {/* Image */}
                <View style={styles.itemImageContainer}>
                    <Image
                        source={{ uri: item.imageUrl }}
                        style={styles.itemImage}
                        resizeMode="cover"
                    />
                    {item.isOnSale && (
                        <View style={styles.saleBadge}>
                            <Text style={styles.saleBadgeText}>SALE</Text>
                        </View>
                    )}
                </View>

                {/* Info */}
                <View style={styles.itemInfo}>
                    <Text style={styles.itemBrand}>{item.brand}</Text>
                    <Text style={styles.itemName} numberOfLines={2}>{item.name}</Text>

                    <View style={styles.priceRow}>
                        <Text style={styles.currentPrice}>
                            {item.currency}{item.currentPrice.toFixed(2)}
                        </Text>
                        {item.originalPrice && item.originalPrice > item.currentPrice && (
                            <Text style={styles.originalPrice}>
                                {item.currency}{item.originalPrice.toFixed(2)}
                            </Text>
                        )}
                        <PriceTrend history={item.priceHistory} />
                    </View>

                    {item.targetPrice && (
                        <View style={styles.targetRow}>
                            <Ionicons name="flag-outline" size={12} color={AppColors.textMuted} />
                            <Text style={styles.targetText}>
                                Target: {item.currency}{item.targetPrice.toFixed(2)}
                            </Text>
                        </View>
                    )}
                </View>
            </TouchableOpacity>
        </Animated.View>
    );
};

// Alert card
const AlertCard = ({ alert, onPress }: { alert: PriceAlert; onPress: () => void }) => (
    <TouchableOpacity
        style={[styles.alertCard, !alert.seen && styles.alertCardUnseen]}
        onPress={onPress}
    >
        <View style={styles.alertIcon}>
            <Ionicons name="pricetag" size={20} color="#34C759" />
        </View>
        <View style={styles.alertContent}>
            <Text style={styles.alertTitle}>{alert.itemName}</Text>
            <Text style={styles.alertText}>
                Price dropped {alert.dropPercent}% from ${alert.previousPrice.toFixed(2)} to ${alert.newPrice.toFixed(2)}
            </Text>
        </View>
        {!alert.seen && <View style={styles.alertDot} />}
    </TouchableOpacity>
);

// Add item modal
const AddItemModal = ({
    visible,
    onClose,
    onAdd
}: {
    visible: boolean;
    onClose: () => void;
    onAdd: (item: NewItemInput) => void;
}) => {
    const [name, setName] = useState('');
    const [brand, setBrand] = useState('');
    const [price, setPrice] = useState('');
    const [imageUrl, setImageUrl] = useState('');
    const [targetPrice, setTargetPrice] = useState('');

    const handleAdd = () => {
        if (!name.trim() || !price.trim()) {
            Alert.alert("Error", "Please enter item name and price");
            return;
        }

        onAdd({
            name: name.trim(),
            brand: brand.trim() || 'Unknown',
            currentPrice: parseFloat(price),
            currency: '$',
            imageUrl: imageUrl.trim() || 'https://via.placeholder.com/150',
            category: 'clothing',
            targetPrice: targetPrice ? parseFloat(targetPrice) : undefined,
        });

        // Reset
        setName('');
        setBrand('');
        setPrice('');
        setImageUrl('');
        setTargetPrice('');
        onClose();
    };

    if (!visible) return null;

    return (
        <View style={styles.modalOverlay}>
            <KeyboardAvoidingView
                behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
                style={styles.modalContainer}
            >
                <Animated.View entering={FadeInUp.springify()} style={styles.modalContent}>
                    <View style={styles.modalHeader}>
                        <Text style={styles.modalTitle}>Track New Item</Text>
                        <TouchableOpacity onPress={onClose}>
                            <Ionicons name="close" size={24} color={AppColors.text} />
                        </TouchableOpacity>
                    </View>

                    <View style={styles.modalBody}>
                        <View style={styles.inputGroup}>
                            <Text style={styles.inputLabel}>Item Name *</Text>
                            <TextInput
                                style={styles.input}
                                placeholder="e.g., Nike Air Max 90"
                                placeholderTextColor={AppColors.textMuted}
                                value={name}
                                onChangeText={setName}
                            />
                        </View>

                        <View style={styles.inputGroup}>
                            <Text style={styles.inputLabel}>Brand</Text>
                            <TextInput
                                style={styles.input}
                                placeholder="e.g., Nike"
                                placeholderTextColor={AppColors.textMuted}
                                value={brand}
                                onChangeText={setBrand}
                            />
                        </View>

                        <View style={styles.inputRow}>
                            <View style={[styles.inputGroup, { flex: 1, marginRight: 8 }]}>
                                <Text style={styles.inputLabel}>Current Price *</Text>
                                <TextInput
                                    style={styles.input}
                                    placeholder="99.99"
                                    placeholderTextColor={AppColors.textMuted}
                                    keyboardType="decimal-pad"
                                    value={price}
                                    onChangeText={setPrice}
                                />
                            </View>
                            <View style={[styles.inputGroup, { flex: 1, marginLeft: 8 }]}>
                                <Text style={styles.inputLabel}>Target Price</Text>
                                <TextInput
                                    style={styles.input}
                                    placeholder="79.99"
                                    placeholderTextColor={AppColors.textMuted}
                                    keyboardType="decimal-pad"
                                    value={targetPrice}
                                    onChangeText={setTargetPrice}
                                />
                            </View>
                        </View>

                        <View style={styles.inputGroup}>
                            <Text style={styles.inputLabel}>Image URL (optional)</Text>
                            <TextInput
                                style={styles.input}
                                placeholder="https://..."
                                placeholderTextColor={AppColors.textMuted}
                                value={imageUrl}
                                onChangeText={setImageUrl}
                                autoCapitalize="none"
                            />
                        </View>
                    </View>

                    <TouchableOpacity style={styles.addButton} onPress={handleAdd}>
                        <Text style={styles.addButtonText}>Add to Tracker</Text>
                    </TouchableOpacity>
                </Animated.View>
            </KeyboardAvoidingView>
        </View>
    );
};

const PriceTrackerScreen = () => {
    const navigation = useNavigation();
    const [showAddModal, setShowAddModal] = useState(false);
    const [activeTab, setActiveTab] = useState<'items' | 'alerts'>('items');

    const {
        trackedItems,
        priceAlerts,
        addItem,
        removeItem,
        markAlertSeen,
        getUnseenAlerts,
        getTotalSavings,
        getItemsOnSale,
    } = usePriceTrackingStore();

    const unseenCount = getUnseenAlerts().length;
    const totalSavings = getTotalSavings();
    const itemsOnSale = getItemsOnSale();

    const handleAddItem = (item: NewItemInput) => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
        addItem(item);
    };

    return (
        <View style={styles.container}>
            <SafeAreaView style={{ flex: 1 }}>
                {/* Header */}
                <View style={styles.header}>
                    <TouchableOpacity
                        onPress={() => navigation.goBack()}
                        hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
                    >
                        <Ionicons name="arrow-back" size={24} color={AppColors.text} />
                    </TouchableOpacity>
                    <Text style={styles.headerTitle}>Price Tracker</Text>
                    <TouchableOpacity
                        onPress={() => setShowAddModal(true)}
                        hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
                    >
                        <Ionicons name="add" size={28} color={AppColors.text} />
                    </TouchableOpacity>
                </View>

                {/* Stats Bar */}
                <Animated.View entering={FadeIn.delay(100)} style={styles.statsBar}>
                    <View style={styles.statItem}>
                        <Text style={styles.statNumber}>{trackedItems.length}</Text>
                        <Text style={styles.statLabel}>Tracking</Text>
                    </View>
                    <View style={styles.statDivider} />
                    <View style={styles.statItem}>
                        <Text style={styles.statNumber}>{itemsOnSale.length}</Text>
                        <Text style={styles.statLabel}>On Sale</Text>
                    </View>
                    <View style={styles.statDivider} />
                    <View style={styles.statItem}>
                        <Text style={[styles.statNumber, { color: '#34C759' }]}>
                            ${totalSavings.toFixed(0)}
                        </Text>
                        <Text style={styles.statLabel}>Savings</Text>
                    </View>
                </Animated.View>

                {/* Tabs */}
                <View style={styles.tabsContainer}>
                    <TouchableOpacity
                        style={[styles.tab, activeTab === 'items' && styles.tabActive]}
                        onPress={() => setActiveTab('items')}
                    >
                        <Text style={[styles.tabText, activeTab === 'items' && styles.tabTextActive]}>
                            Items
                        </Text>
                    </TouchableOpacity>
                    <TouchableOpacity
                        style={[styles.tab, activeTab === 'alerts' && styles.tabActive]}
                        onPress={() => setActiveTab('alerts')}
                    >
                        <Text style={[styles.tabText, activeTab === 'alerts' && styles.tabTextActive]}>
                            Alerts
                        </Text>
                        {unseenCount > 0 && (
                            <View style={styles.alertBadge}>
                                <Text style={styles.alertBadgeText}>{unseenCount}</Text>
                            </View>
                        )}
                    </TouchableOpacity>
                </View>

                {/* Content */}
                <ScrollView
                    style={styles.content}
                    contentContainerStyle={styles.contentContainer}
                    showsVerticalScrollIndicator={false}
                >
                    {activeTab === 'items' ? (
                        trackedItems.length === 0 ? (
                            <Animated.View entering={FadeIn} style={styles.emptyState}>
                                <View style={styles.emptyIcon}>
                                    <Ionicons name="pricetags-outline" size={48} color={AppColors.textMuted} />
                                </View>
                                <Text style={styles.emptyTitle}>No items tracked</Text>
                                <Text style={styles.emptyText}>
                                    Add items you want to buy and we'll track price changes for you
                                </Text>
                                <TouchableOpacity
                                    style={styles.emptyButton}
                                    onPress={() => setShowAddModal(true)}
                                >
                                    <Ionicons name="add" size={20} color={AppColors.background} />
                                    <Text style={styles.emptyButtonText}>Add Item</Text>
                                </TouchableOpacity>
                            </Animated.View>
                        ) : (
                            trackedItems.map((item, index) => (
                                <TrackedItemCard
                                    key={item.id}
                                    item={item}
                                    onPress={() => {
                                        // Could navigate to detail
                                        Haptics.selectionAsync();
                                    }}
                                    onRemove={() => removeItem(item.id)}
                                />
                            ))
                        )
                    ) : (
                        priceAlerts.length === 0 ? (
                            <Animated.View entering={FadeIn} style={styles.emptyState}>
                                <View style={styles.emptyIcon}>
                                    <Ionicons name="notifications-outline" size={48} color={AppColors.textMuted} />
                                </View>
                                <Text style={styles.emptyTitle}>No alerts yet</Text>
                                <Text style={styles.emptyText}>
                                    We'll notify you when prices drop on your tracked items
                                </Text>
                            </Animated.View>
                        ) : (
                            priceAlerts.map((alert) => (
                                <AlertCard
                                    key={alert.id}
                                    alert={alert}
                                    onPress={() => {
                                        markAlertSeen(alert.id);
                                        Haptics.selectionAsync();
                                    }}
                                />
                            ))
                        )
                    )}
                </ScrollView>
            </SafeAreaView>

            {/* Add Modal */}
            <AddItemModal
                visible={showAddModal}
                onClose={() => setShowAddModal(false)}
                onAdd={handleAddItem}
            />
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: AppColors.background,
    },
    header: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: 20,
        paddingVertical: 16,
    },
    headerTitle: {
        fontSize: 18,
        fontWeight: '600',
        color: AppColors.text,
    },

    // Stats
    statsBar: {
        flexDirection: 'row',
        backgroundColor: AppColors.surface,
        marginHorizontal: 20,
        borderRadius: 16,
        padding: 16,
        marginBottom: 16,
    },
    statItem: {
        flex: 1,
        alignItems: 'center',
    },
    statNumber: {
        fontSize: 24,
        fontWeight: '700',
        color: AppColors.text,
    },
    statLabel: {
        fontSize: 12,
        color: AppColors.textSecondary,
        marginTop: 4,
    },
    statDivider: {
        width: 1,
        backgroundColor: AppColors.border,
    },

    // Tabs
    tabsContainer: {
        flexDirection: 'row',
        paddingHorizontal: 20,
        marginBottom: 16,
    },
    tab: {
        flex: 1,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        paddingVertical: 12,
        borderRadius: 12,
        backgroundColor: AppColors.surface,
        marginRight: 8,
    },
    tabActive: {
        backgroundColor: AppColors.primary,
    },
    tabText: {
        fontSize: 14,
        fontWeight: '600',
        color: AppColors.textSecondary,
    },
    tabTextActive: {
        color: AppColors.background,
    },
    alertBadge: {
        backgroundColor: '#FF3B30',
        minWidth: 18,
        height: 18,
        borderRadius: 9,
        alignItems: 'center',
        justifyContent: 'center',
        marginLeft: 6,
    },
    alertBadgeText: {
        color: '#FFF',
        fontSize: 11,
        fontWeight: '700',
    },

    // Content
    content: {
        flex: 1,
    },
    contentContainer: {
        paddingHorizontal: 20,
        paddingBottom: 40,
    },

    // Item Card
    itemCard: {
        flexDirection: 'row',
        backgroundColor: AppColors.surface,
        borderRadius: 16,
        overflow: 'hidden',
        marginBottom: 12,
        borderWidth: 1,
        borderColor: AppColors.border,
    },
    itemImageContainer: {
        width: 100,
        height: 120,
        position: 'relative',
    },
    itemImage: {
        width: '100%',
        height: '100%',
        backgroundColor: '#F5F5F5',
    },
    saleBadge: {
        position: 'absolute',
        top: 8,
        left: 8,
        backgroundColor: '#FF3B30',
        paddingHorizontal: 6,
        paddingVertical: 2,
        borderRadius: 4,
    },
    saleBadgeText: {
        color: '#FFF',
        fontSize: 9,
        fontWeight: '700',
    },
    itemInfo: {
        flex: 1,
        padding: 12,
    },
    itemBrand: {
        fontSize: 11,
        color: AppColors.textMuted,
        textTransform: 'uppercase',
        letterSpacing: 0.5,
    },
    itemName: {
        fontSize: 14,
        fontWeight: '600',
        color: AppColors.text,
        marginTop: 4,
        marginBottom: 8,
    },
    priceRow: {
        flexDirection: 'row',
        alignItems: 'center',
    },
    currentPrice: {
        fontSize: 16,
        fontWeight: '700',
        color: AppColors.text,
    },
    originalPrice: {
        fontSize: 14,
        color: AppColors.textMuted,
        textDecorationLine: 'line-through',
        marginLeft: 8,
    },
    trendBadge: {
        flexDirection: 'row',
        alignItems: 'center',
        paddingHorizontal: 6,
        paddingVertical: 2,
        borderRadius: 4,
        marginLeft: 8,
    },
    trendDown: {
        backgroundColor: 'rgba(52, 199, 89, 0.15)',
    },
    trendUp: {
        backgroundColor: 'rgba(255, 59, 48, 0.15)',
    },
    trendText: {
        fontSize: 11,
        fontWeight: '600',
        marginLeft: 2,
    },
    trendTextDown: {
        color: '#34C759',
    },
    trendTextUp: {
        color: '#FF3B30',
    },
    targetRow: {
        flexDirection: 'row',
        alignItems: 'center',
        marginTop: 8,
    },
    targetText: {
        fontSize: 12,
        color: AppColors.textMuted,
        marginLeft: 4,
    },

    // Alert Card
    alertCard: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: AppColors.surface,
        borderRadius: 12,
        padding: 14,
        marginBottom: 10,
        borderWidth: 1,
        borderColor: AppColors.border,
    },
    alertCardUnseen: {
        backgroundColor: 'rgba(52, 199, 89, 0.08)',
        borderColor: 'rgba(52, 199, 89, 0.2)',
    },
    alertIcon: {
        width: 40,
        height: 40,
        borderRadius: 10,
        backgroundColor: 'rgba(52, 199, 89, 0.15)',
        alignItems: 'center',
        justifyContent: 'center',
        marginRight: 12,
    },
    alertContent: {
        flex: 1,
    },
    alertTitle: {
        fontSize: 14,
        fontWeight: '600',
        color: AppColors.text,
        marginBottom: 2,
    },
    alertText: {
        fontSize: 12,
        color: AppColors.textSecondary,
    },
    alertDot: {
        width: 8,
        height: 8,
        borderRadius: 4,
        backgroundColor: '#34C759',
    },

    // Empty State
    emptyState: {
        alignItems: 'center',
        paddingVertical: 60,
    },
    emptyIcon: {
        width: 80,
        height: 80,
        borderRadius: 40,
        backgroundColor: AppColors.surface,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 20,
    },
    emptyTitle: {
        fontSize: 18,
        fontWeight: '600',
        color: AppColors.text,
        marginBottom: 8,
    },
    emptyText: {
        fontSize: 14,
        color: AppColors.textSecondary,
        textAlign: 'center',
        paddingHorizontal: 40,
        lineHeight: 20,
    },
    emptyButton: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: AppColors.primary,
        paddingHorizontal: 20,
        paddingVertical: 12,
        borderRadius: 12,
        marginTop: 24,
    },
    emptyButtonText: {
        fontSize: 15,
        fontWeight: '600',
        color: AppColors.background,
        marginLeft: 6,
    },

    // Modal
    modalOverlay: {
        ...StyleSheet.absoluteFillObject,
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        justifyContent: 'flex-end',
    },
    modalContainer: {
        flex: 1,
        justifyContent: 'flex-end',
    },
    modalContent: {
        backgroundColor: AppColors.background,
        borderTopLeftRadius: 24,
        borderTopRightRadius: 24,
        paddingBottom: 40,
    },
    modalHeader: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: 20,
        borderBottomWidth: 1,
        borderBottomColor: AppColors.border,
    },
    modalTitle: {
        fontSize: 18,
        fontWeight: '600',
        color: AppColors.text,
    },
    modalBody: {
        padding: 20,
    },
    inputGroup: {
        marginBottom: 16,
    },
    inputLabel: {
        fontSize: 13,
        fontWeight: '500',
        color: AppColors.textSecondary,
        marginBottom: 8,
    },
    input: {
        backgroundColor: AppColors.surface,
        borderRadius: 12,
        padding: 14,
        fontSize: 16,
        color: AppColors.text,
        borderWidth: 1,
        borderColor: AppColors.border,
    },
    inputRow: {
        flexDirection: 'row',
    },
    addButton: {
        backgroundColor: AppColors.primary,
        marginHorizontal: 20,
        paddingVertical: 16,
        borderRadius: 14,
        alignItems: 'center',
    },
    addButtonText: {
        fontSize: 16,
        fontWeight: '600',
        color: AppColors.background,
    },
});

export default PriceTrackerScreen;
