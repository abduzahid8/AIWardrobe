import React, { useState } from "react";
import {
    View,
    Text,
    Image,
    ScrollView,
    TouchableOpacity,
    StyleSheet,
    Dimensions,
    FlatList,
    Platform,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { Ionicons } from "@expo/vector-icons";
import { colors, shadows, spacing } from "../../theme";
import { useNavigation } from "@react-navigation/native";
import { useTranslation } from "react-i18next";
import { popularItems } from "../../../data"; // We can reuse the data for now

const { width } = Dimensions.get("window");

const CuratedClosetScreen = () => {
    const navigation = useNavigation();
    const { t } = useTranslation();
    const [activeFilter, setActiveFilter] = useState("All");

    const renderItem = React.useCallback(({ item, index }: { item: any; index: number }) => (
        <TouchableOpacity
            style={[styles.itemCard, index % 2 === 0 ? { marginRight: spacing.m } : {}, shadows.soft]}
            onPress={() => { }}
        >
            <Image source={{ uri: item.image }} style={styles.itemImage} resizeMode="cover" />
            <View style={styles.itemInfo}>
                <Text style={styles.itemName} numberOfLines={1}>{item.itemName}</Text>
                <Text style={styles.itemBrand}>Brand Name</Text>
            </View>
        </TouchableOpacity>
    ), []);

    return (
        <View style={styles.container}>
            <SafeAreaView style={styles.safeArea}>
                {/* Header */}
                <View style={styles.header}>
                    <Text style={styles.headerTitle}>{t('discover.curatedCloset')}</Text>
                    <View style={styles.headerActions}>
                        <TouchableOpacity style={styles.iconButton}>
                            <Ionicons name="search-outline" size={24} color={colors.text.primary} />
                        </TouchableOpacity>
                        <TouchableOpacity style={styles.iconButton}>
                            <Ionicons name="filter-outline" size={24} color={colors.text.primary} />
                        </TouchableOpacity>
                    </View>
                </View>

                {/* Filters */}
                <View style={styles.filterContainer}>
                    <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={{ paddingHorizontal: spacing.l, gap: spacing.m }}>
                        {[t('discover.filters.all'), t('discover.filters.tops'), t('discover.filters.bottoms'), t('discover.filters.shoes'), t('discover.filters.accessories')].map((filterLabel, idx) => {
                            const filterKey = ["All", "Tops", "Bottoms", "Shoes", "Accessories"][idx];
                            return (
                                <TouchableOpacity
                                    key={filterKey}
                                    onPress={() => setActiveFilter(filterKey)}
                                    style={[
                                        styles.filterChip,
                                        activeFilter === filterKey && styles.filterChipActive
                                    ]}
                                >
                                    <Text style={[
                                        styles.filterText,
                                        activeFilter === filterKey && styles.filterTextActive
                                    ]}>{filterLabel}</Text>
                                </TouchableOpacity>
                            );
                        })}
                    </ScrollView>
                </View>

                {/* Masonry Grid (Simulated with FlatList numColumns=2) */}
                <FlatList
                    data={popularItems}
                    renderItem={renderItem}
                    keyExtractor={(item, index) => index.toString()}
                    numColumns={2}
                    contentContainerStyle={styles.gridContent}
                    showsVerticalScrollIndicator={false}
                    columnWrapperStyle={{ justifyContent: "space-between" }}
                />

                {/* Floating "Add" Button */}
                <TouchableOpacity
                    style={[styles.fab, shadows.medium]}
                    onPress={() => (navigation as any).navigate("AddOutfit")}
                >
                    <Ionicons name="add" size={32} color="#FFF" />
                </TouchableOpacity>

            </SafeAreaView>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: colors.background,
    },
    safeArea: {
        flex: 1,
    },
    header: {
        flexDirection: "row",
        justifyContent: "space-between",
        alignItems: "center",
        paddingHorizontal: spacing.l,
        paddingVertical: spacing.m,
        borderBottomWidth: 1,
        borderBottomColor: colors.border,
    },
    headerTitle: {
        fontSize: 24,
        color: colors.text.primary,
        fontFamily: Platform.OS === 'ios' ? 'Didot' : 'serif',
        fontWeight: "600",
    },
    headerActions: {
        flexDirection: "row",
        gap: spacing.m,
    },
    iconButton: {
        padding: spacing.xs,
    },
    filterContainer: {
        paddingVertical: spacing.m,
    },
    filterChip: {
        paddingVertical: spacing.s,
        paddingHorizontal: spacing.m,
        borderRadius: 100,
        borderWidth: 1,
        borderColor: colors.border,
    },
    filterChipActive: {
        backgroundColor: colors.text.primary,
        borderColor: colors.text.primary,
    },
    filterText: {
        fontSize: 14,
        color: colors.text.primary,
    },
    filterTextActive: {
        color: colors.surface,
    },
    gridContent: {
        paddingHorizontal: spacing.l,
        paddingBottom: 100,
    },
    itemCard: {
        flex: 1,
        marginBottom: spacing.l,
        backgroundColor: colors.surface,
        borderRadius: 0, // Editorial style
    },
    itemImage: {
        width: "100%",
        aspectRatio: 0.75, // Portrait aspect ratio
        backgroundColor: colors.surfaceHighlight,
    },
    itemInfo: {
        padding: spacing.s,
    },
    itemName: {
        fontSize: 14,
        color: colors.text.primary,
        fontWeight: "500",
        marginBottom: 2,
    },
    itemBrand: {
        fontSize: 12,
        color: colors.text.secondary,
        textTransform: "uppercase",
    },
    fab: {
        position: "absolute",
        bottom: 32,
        right: 32,
        width: 64,
        height: 64,
        borderRadius: 32,
        backgroundColor: colors.text.accent,
        alignItems: "center",
        justifyContent: "center",
    },
});

export default CuratedClosetScreen;
