import {
  ActivityIndicator,
  Image,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  Platform,
  Alert,
  Modal,
  TextInput,
  Pressable,
  Dimensions,
} from "react-native";
import React, { useEffect, useState, useCallback } from "react";
import useAuthStore from "../store/auth";
import { Ionicons } from "@expo/vector-icons";
import axios from "axios";
import { mpants, mshirts, pants, shoes, skirts, tops } from "../images";
import { useTranslation } from "react-i18next";
import LanguageSelector from "../components/LanguageSelector";
import { ChipButton, IconButton } from "../components/AnimatedButton";
import { colors, shadows, spacing, animations } from "../src/theme";
import AsyncStorage from "@react-native-async-storage/async-storage";
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withSpring,
  withTiming,
  runOnJS,
} from "react-native-reanimated";
import * as Haptics from "expo-haptics";

import { useNavigation, useFocusEffect } from "@react-navigation/native";

const { width } = Dimensions.get("window");

// Animated card component with iOS 25-style interactions
const ClothingCard = ({
  item,
  index,
  onPress,
  onLongPress,
  onFavorite,
  isFavorite,
  uniqueKey,
}: {
  item: any;
  index: number;
  onPress: () => void;
  onLongPress: () => void;
  onFavorite: () => void;
  isFavorite: boolean;
  uniqueKey: string;
}) => {
  const scale = useSharedValue(1);
  const opacity = useSharedValue(1);

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: withSpring(scale.value, { damping: 15, stiffness: 400 }) }],
    opacity: opacity.value,
  }));

  const handlePressIn = () => {
    scale.value = 0.96;
    opacity.value = withTiming(0.9, { duration: 100 });
  };

  const handlePressOut = () => {
    scale.value = 1;
    opacity.value = withTiming(1, { duration: 150 });
  };

  return (
    <View style={styles.gridItem}>
      <Animated.View style={[animatedStyle]}>
        <Pressable
          onPressIn={handlePressIn}
          onPressOut={handlePressOut}
          onPress={() => {
            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
            onPress();
          }}
          onLongPress={() => {
            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
            onLongPress();
          }}
          delayLongPress={500}
        >
          <View style={[styles.itemCard, shadows.soft]}>
            {/* Favorite Button */}
            <TouchableOpacity
              style={styles.favoriteButton}
              onPress={() => {
                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                onFavorite();
              }}
              hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
            >
              <Ionicons
                name={isFavorite ? "heart" : "heart-outline"}
                size={20}
                color={isFavorite ? colors.favorite : colors.text.secondary}
              />
            </TouchableOpacity>

            {/* Saved Badge */}
            {item.isSaved && (
              <View style={styles.savedBadge}>
                <Text style={styles.savedBadgeText}>MY</Text>
              </View>
            )}

            <Image
              style={styles.itemImage}
              source={{ uri: item?.image }}
              resizeMode="contain"
            />
            <View style={styles.itemInfo}>
              <Text style={styles.itemType} numberOfLines={1}>
                {item?.type}
              </Text>
              {item?.color && (
                <Text style={styles.itemColor} numberOfLines={1}>
                  {item.color}
                </Text>
              )}
            </View>
          </View>
        </Pressable>
      </Animated.View>
    </View>
  );
};

const ProfileScreen = () => {
  const navigation = useNavigation();
  const { t } = useTranslation();
  const [activeTab, setActiveTab] = useState("Clothes");
  const [activeCategory, setActiveCategory] = useState("All");
  const { logout, user, token } = useAuthStore();
  const [outifts, setOutfits] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const username = user?.username || "sujanand";
  const email = user?.email || "";
  const followersCount = user?.followers?.length || 0;
  const followingCount = user?.following?.length || 0;
  const profileImage = user?.profileImage || "https://picsum.photos/100/100";

  // Edit Modal State
  const [showEditModal, setShowEditModal] = useState(false);
  const [editingItem, setEditingItem] = useState<any>(null);
  const [editType, setEditType] = useState("");
  const [editColor, setEditColor] = useState("");

  // Favorites State
  const [favorites, setFavorites] = useState<string[]>([]);

  // Settings menu options
  const handleLogout = async () => {
    Alert.alert(
      "Logout",
      "Are you sure you want to logout?",
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Logout",
          style: "destructive",
          onPress: async () => {
            try {
              await AsyncStorage.removeItem("userToken");
              await AsyncStorage.removeItem("userData");
              logout();
              setShowSettings(false);
              (navigation as any).reset({
                index: 0,
                routes: [{ name: "Login" }],
              });
            } catch (error) {
              console.error("Logout error:", error);
            }
          },
        },
      ]
    );
  };

  const handleEditProfile = () => {
    setShowSettings(false);
    Alert.alert("Edit Profile", "Profile editing coming soon!");
  };

  const handleNotifications = () => {
    setShowSettings(false);
    Alert.alert("Notifications", "Notification settings coming soon!");
  };

  const handlePrivacy = () => {
    setShowSettings(false);
    Alert.alert("Privacy", "Privacy settings coming soon!");
  };

  const handleHelp = () => {
    setShowSettings(false);
    Alert.alert(
      "Help & Support",
      "Contact us at: support@aiwardrobe.com\n\nVersion: 1.0.0"
    );
  };

  // Saved clothing items from local storage
  const [savedClothes, setSavedClothes] = useState<any[]>([]);
  const [loadingSaved, setLoadingSaved] = useState(false);

  // Load favorites from AsyncStorage
  const loadFavorites = useCallback(async () => {
    try {
      const data = await AsyncStorage.getItem('wardrobeFavorites');
      if (data) {
        setFavorites(JSON.parse(data));
      }
    } catch (error) {
      console.log("Error loading favorites:", error);
    }
  }, []);

  // Save favorites to AsyncStorage
  const saveFavorites = async (newFavorites: string[]) => {
    try {
      await AsyncStorage.setItem('wardrobeFavorites', JSON.stringify(newFavorites));
      setFavorites(newFavorites);
    } catch (error) {
      console.log("Error saving favorites:", error);
    }
  };

  // Toggle favorite
  const toggleFavorite = (itemId: string) => {
    const newFavorites = favorites.includes(itemId)
      ? favorites.filter(id => id !== itemId)
      : [...favorites, itemId];
    saveFavorites(newFavorites);
  };

  // Fetch saved clothes from local storage
  const fetchSavedClothes = useCallback(async () => {
    setLoadingSaved(true);

    try {
      // Load from local AsyncStorage
      const localData = await AsyncStorage.getItem('myWardrobeItems');
      const localItems = localData ? JSON.parse(localData) : [];
      console.log('📦 Loaded', localItems.length, 'items from local storage');
      setSavedClothes(localItems);
    } catch (error) {
      console.log("Error fetching saved clothes:", error);
    } finally {
      setLoadingSaved(false);
    }
  }, []);

  // Refresh data when screen comes into focus
  useFocusEffect(
    useCallback(() => {
      fetchSavedClothes();
      loadFavorites();
    }, [fetchSavedClothes, loadFavorites])
  );

  // Delete item function
  const deleteItem = async (item: any) => {
    Alert.alert(
      "Delete Item",
      `Are you sure you want to delete this ${item.type || 'item'}?`,
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Delete",
          style: "destructive",
          onPress: async () => {
            try {
              Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
              const updatedItems = savedClothes.filter(i => i.id !== item.id);
              await AsyncStorage.setItem('myWardrobeItems', JSON.stringify(updatedItems));
              setSavedClothes(updatedItems);

              // Also remove from favorites if present
              if (favorites.includes(item.id)) {
                saveFavorites(favorites.filter(id => id !== item.id));
              }
            } catch (error) {
              console.error("Delete error:", error);
              Alert.alert("Error", "Failed to delete item");
            }
          },
        },
      ]
    );
  };

  // Edit item function
  const openEditModal = (item: any) => {
    if (!item.isSaved) {
      Alert.alert("Info", "You can only edit items you've added to your wardrobe.");
      return;
    }
    setEditingItem(item);
    setEditType(item.type || "");
    setEditColor(item.color || "");
    setShowEditModal(true);
  };

  const saveEditedItem = async () => {
    if (!editingItem) return;

    try {
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      const updatedItems = savedClothes.map(item =>
        item.id === editingItem.id
          ? { ...item, type: editType, color: editColor }
          : item
      );
      await AsyncStorage.setItem('myWardrobeItems', JSON.stringify(updatedItems));
      setSavedClothes(updatedItems);
      setShowEditModal(false);
      setEditingItem(null);
    } catch (error) {
      console.error("Edit error:", error);
      Alert.alert("Error", "Failed to save changes");
    }
  };

  const popularClothes = React.useMemo(() => [
    ...pants,
    ...tops,
    ...skirts,
    ...mpants,
    ...mshirts,
    ...shoes,
  ].filter((item) => item.image), []);

  // Combine saved clothes with popular clothes for display - ensure unique IDs
  const allClothes = React.useMemo(() => {
    const timestamp = Date.now();
    const saved = savedClothes.map((item, idx) => ({
      ...item,
      uniqueId: `saved-${item.id || idx}-${timestamp}`,
      image: item.image || item.imageUrl || 'https://via.placeholder.com/150',
      isSaved: true
    }));
    const popular = popularClothes.map((item, idx) => ({
      ...item,
      uniqueId: `popular-${item.type || 'item'}-${idx}`,
      isSaved: false
    }));
    return [...saved, ...popular];
  }, [savedClothes, popularClothes]);

  const fetchOutfits = useCallback(async () => {
    if (!user?._id || !token) return;
    setLoading(true);

    try {
      const response = await axios.get(
        `https://aiwardrobe-ivh4.onrender.com/save-outfit/user/${user._id}`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );
      setOutfits(response.data);
    } catch (error) {
      console.log("Error", error);
    } finally {
      setLoading(false);
    }
  }, [user?._id, token]);

  useEffect(() => {
    fetchOutfits();
  }, [fetchOutfits]);

  const filteredClothes = React.useMemo(() => {
    let clothes = allClothes;

    // Filter by category
    if (activeCategory === "Favorites") {
      clothes = clothes.filter(item => favorites.includes(item.uniqueId));
    } else if (activeCategory !== "All") {
      clothes = clothes.filter((item) => {
        switch (activeCategory) {
          case "Tops":
            return item.type == "shirt" || item.type?.toLowerCase().includes('shirt') || item.type?.toLowerCase().includes('top');
          case "Bottoms":
            return item.type == "pants" || item.type == "skirts" || item.type?.toLowerCase().includes('pant') || item.type?.toLowerCase().includes('jean');
          case "Shoes":
            return item.type == "shoes" || item.type?.toLowerCase().includes('shoe') || item.type?.toLowerCase().includes('sneaker');
          default:
            return true;
        }
      });
    }

    return clothes;
  }, [activeCategory, allClothes, favorites]);

  const sortItems = useCallback((items: any[]) => {
    const order = ["shirt", "pants", "skirts", "shoes"];
    return items.sort(
      (a: any, b: any) => order.indexOf(a.type) - order.indexOf(b.type)
    );
  }, []);

  // Categories including Favorites
  const categories = [
    { key: "All", label: t('profile.categories.all') },
    { key: "Favorites", label: "♥ Favorites" },
    { key: "Tops", label: t('profile.categories.tops') },
    { key: "Bottoms", label: t('profile.categories.bottoms') },
    { key: "Shoes", label: t('profile.categories.shoes') },
  ];

  return (
    <View style={styles.container}>
      <SafeAreaView style={styles.safeArea}>
        <ScrollView showsVerticalScrollIndicator={false}>
          {/* Header */}
          <View style={styles.header}>
            <Text style={styles.username}>{username}</Text>
            <View style={styles.headerActions}>
              <LanguageSelector />
              <IconButton
                icon="videocam-outline"
                onPress={() => (navigation as any).navigate('WardrobeVideo')}
              />
              <IconButton
                icon="settings-outline"
                onPress={() => setShowSettings(true)}
              />
            </View>
          </View>


          {/* Profile Info */}
          <View style={styles.profileSection}>
            <View style={styles.avatarContainer}>
              <Image
                style={styles.avatar}
                source={{ uri: profileImage }}
              />
            </View>
            <View style={styles.statsContainer}>
              <View style={styles.statItem}>
                <Text style={styles.statNumber}>{filteredClothes.length}</Text>
                <Text style={styles.statLabel}>{t('profile.tabs.clothes')}</Text>
              </View>
              <View style={styles.statDivider} />
              <View style={styles.statItem}>
                <Text style={styles.statNumber}>{outifts.length}</Text>
                <Text style={styles.statLabel}>{t('profile.tabs.outfits')}</Text>
              </View>
              <View style={styles.statDivider} />
              <View style={styles.statItem}>
                <Text style={styles.statNumber}>{favorites.length}</Text>
                <Text style={styles.statLabel}>Favorites</Text>
              </View>
            </View>
          </View>

          {/* Tabs */}
          <View style={styles.tabsContainer}>
            {[t('profile.tabs.clothes'), t('profile.tabs.outfits'), t('profile.tabs.collections')].map((tab, idx) => {
              const tabKey = ['Clothes', 'Outfits', 'Collections'][idx];
              return (
                <TouchableOpacity
                  key={tabKey}
                  onPress={() => {
                    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                    setActiveTab(tabKey);
                  }}
                  style={styles.tab}
                >
                  <Text
                    style={[
                      styles.tabText,
                      activeTab == tabKey && styles.tabTextActive
                    ]}
                  >
                    {tab}
                  </Text>
                  {activeTab === tabKey && <View style={styles.tabIndicator} />}
                </TouchableOpacity>
              );
            })}
          </View>

          {/* Category Filters with Favorites - iOS 26 Style */}
          {activeTab === "Clothes" && (
            <ScrollView
              horizontal
              showsHorizontalScrollIndicator={false}
              style={styles.filtersScroll}
              contentContainerStyle={styles.filtersContent}
            >
              {categories.map((cat) => (
                <ChipButton
                  key={cat.key}
                  title={cat.label}
                  isActive={activeCategory === cat.key}
                  onPress={() => setActiveCategory(cat.key)}
                  style={cat.key === "Favorites" ? styles.favoritesChip : undefined}
                />
              ))}
            </ScrollView>
          )}

          {/* Content */}
          {activeTab == "Clothes" && (
            <View style={styles.gridContainer}>
              {filteredClothes.length == 0 ? (
                <View style={styles.emptyContainer}>
                  <Ionicons name="shirt-outline" size={48} color={colors.text.secondary} />
                  <Text style={styles.emptyText}>
                    {activeCategory === "Favorites"
                      ? "No favorites yet\nTap ♥ on items to add them"
                      : t('profile.noClothes')}
                  </Text>
                </View>
              ) : (
                <View style={styles.grid}>
                  {filteredClothes?.map((item, index) => (
                    <ClothingCard
                      key={item.uniqueId || `item-${index}`}
                      uniqueKey={item.uniqueId || `item-${index}`}
                      item={item}
                      index={index}
                      onPress={() => openEditModal(item)}
                      onLongPress={() => item.isSaved && deleteItem(item)}
                      onFavorite={() => toggleFavorite(item.uniqueId)}
                      isFavorite={favorites.includes(item.uniqueId)}
                    />
                  ))}
                </View>
              )}
            </View>
          )}

          {activeTab == "Outfits" && (
            <View style={styles.gridContainer}>
              {loading ? (
                <ActivityIndicator size={"large"} color={colors.text.primary} />
              ) : outifts.length === 0 ? (
                <Text style={styles.emptyText}>{t('profile.noOutfits')}</Text>
              ) : (
                <View style={styles.grid}>
                  {outifts?.map((outfit) => (
                    <View key={outfit._id} style={styles.outfitItem}>
                      <View style={[styles.outfitCard, shadows.soft]}>
                        <View style={styles.outfitImageContainer}>
                          {sortItems(outfit.items).map((item: any, index: number) => (
                            <Image
                              key={`${outfit._id}-${item.id}-${index}`}
                              source={{ uri: item.image }}
                              style={styles.outfitLayeredImage}
                              resizeMode="contain"
                            />
                          ))}
                        </View>
                        <View style={styles.outfitInfo}>
                          <Text style={styles.outfitDate}>
                            {outfit?.date}
                          </Text>
                          <Text style={styles.outfitOccasion}>
                            {outfit.occasion}
                          </Text>
                        </View>
                      </View>
                    </View>
                  ))}
                </View>
              )}
            </View>
          )}

          {activeTab === "Collections" && (
            <View style={styles.gridContainer}>
              <View style={styles.emptyContainer}>
                <Ionicons name="folder-outline" size={48} color={colors.text.secondary} />
                <Text style={styles.emptyText}>Collections coming soon!</Text>
              </View>
            </View>
          )}
        </ScrollView>

        {/* Edit Item Modal */}
        <Modal
          visible={showEditModal}
          animationType="slide"
          transparent={true}
          onRequestClose={() => setShowEditModal(false)}
        >
          <View style={styles.modalOverlay}>
            <View style={styles.editModal}>
              <View style={styles.modalHandle} />
              <View style={styles.editHeader}>
                <Text style={styles.editTitle}>Edit Item</Text>
                <TouchableOpacity onPress={() => setShowEditModal(false)}>
                  <Ionicons name="close" size={24} color={colors.text.primary} />
                </TouchableOpacity>
              </View>

              {editingItem && (
                <View style={styles.editContent}>
                  <Image
                    source={{ uri: editingItem.image }}
                    style={styles.editImage}
                    resizeMode="contain"
                  />

                  <View style={styles.editField}>
                    <Text style={styles.editLabel}>Type</Text>
                    <TextInput
                      style={styles.editInput}
                      value={editType}
                      onChangeText={setEditType}
                      placeholder="e.g., Jacket, Shirt, Pants"
                      placeholderTextColor={colors.text.secondary}
                    />
                  </View>

                  <View style={styles.editField}>
                    <Text style={styles.editLabel}>Color</Text>
                    <TextInput
                      style={styles.editInput}
                      value={editColor}
                      onChangeText={setEditColor}
                      placeholder="e.g., Black, Navy Blue"
                      placeholderTextColor={colors.text.secondary}
                    />
                  </View>

                  <View style={styles.editActions}>
                    <TouchableOpacity
                      style={styles.deleteButton}
                      onPress={() => {
                        setShowEditModal(false);
                        deleteItem(editingItem);
                      }}
                    >
                      <Ionicons name="trash-outline" size={20} color="#FFF" />
                      <Text style={styles.deleteButtonText}>Delete</Text>
                    </TouchableOpacity>

                    <TouchableOpacity
                      style={styles.saveButton}
                      onPress={saveEditedItem}
                    >
                      <Ionicons name="checkmark" size={20} color="#FFF" />
                      <Text style={styles.saveButtonText}>Save</Text>
                    </TouchableOpacity>
                  </View>
                </View>
              )}
            </View>
          </View>
        </Modal>

        {/* Settings Modal */}
        <Modal
          visible={showSettings}
          animationType="slide"
          transparent={true}
          onRequestClose={() => setShowSettings(false)}
        >
          <View style={styles.modalOverlay}>
            <View style={styles.settingsModal}>
              <View style={styles.modalHandle} />
              <View style={styles.settingsHeader}>
                <Text style={styles.settingsTitle}>Settings</Text>
                <TouchableOpacity onPress={() => setShowSettings(false)}>
                  <Ionicons name="close" size={24} color={colors.text.primary} />
                </TouchableOpacity>
              </View>

              <TouchableOpacity style={styles.settingsItem} onPress={handleEditProfile}>
                <Ionicons name="person-outline" size={22} color={colors.text.primary} />
                <Text style={styles.settingsItemText}>Edit Profile</Text>
                <Ionicons name="chevron-forward" size={20} color={colors.text.secondary} />
              </TouchableOpacity>

              <TouchableOpacity style={styles.settingsItem} onPress={handleNotifications}>
                <Ionicons name="notifications-outline" size={22} color={colors.text.primary} />
                <Text style={styles.settingsItemText}>Notifications</Text>
                <Ionicons name="chevron-forward" size={20} color={colors.text.secondary} />
              </TouchableOpacity>

              <TouchableOpacity style={styles.settingsItem} onPress={handlePrivacy}>
                <Ionicons name="lock-closed-outline" size={22} color={colors.text.primary} />
                <Text style={styles.settingsItemText}>Privacy</Text>
                <Ionicons name="chevron-forward" size={20} color={colors.text.secondary} />
              </TouchableOpacity>

              <TouchableOpacity style={styles.settingsItem} onPress={handleHelp}>
                <Ionicons name="help-circle-outline" size={22} color={colors.text.primary} />
                <Text style={styles.settingsItemText}>Help & Support</Text>
                <Ionicons name="chevron-forward" size={20} color={colors.text.secondary} />
              </TouchableOpacity>

              <View style={styles.settingsDivider} />

              <TouchableOpacity style={styles.logoutButton} onPress={handleLogout}>
                <Ionicons name="log-out-outline" size={22} color="#FF3B30" />
                <Text style={styles.logoutText}>Logout</Text>
              </TouchableOpacity>
            </View>
          </View>
        </Modal>
      </SafeAreaView>
    </View>
  );
};

export default ProfileScreen;

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
  username: {
    fontSize: 24,
    color: colors.text.primary,
    fontFamily: Platform.OS === 'ios' ? 'Didot' : 'serif',
    fontWeight: "600",
  },
  headerActions: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.m,
  },
  iconButton: {
    padding: spacing.xs,
  },
  profileSection: {
    paddingHorizontal: spacing.l,
    paddingVertical: spacing.xl,
    alignItems: "center",
  },
  avatarContainer: {
    marginBottom: spacing.l,
  },
  avatar: {
    width: 100,
    height: 100,
    borderRadius: 50,
    borderWidth: 2,
    borderColor: colors.border,
  },
  statsContainer: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.l,
  },
  statItem: {
    alignItems: "center",
  },
  statNumber: {
    fontSize: 24,
    color: colors.text.primary,
    fontWeight: "600",
    marginBottom: spacing.xs,
  },
  statLabel: {
    fontSize: 12,
    color: colors.text.secondary,
    textTransform: "uppercase",
    letterSpacing: 1,
  },
  statDivider: {
    width: 1,
    height: 40,
    backgroundColor: colors.border,
  },
  tabsContainer: {
    flexDirection: "row",
    paddingHorizontal: spacing.l,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  tab: {
    flex: 1,
    paddingVertical: spacing.m,
    alignItems: "center",
  },
  tabText: {
    fontSize: 14,
    color: colors.text.secondary,
    fontWeight: "500",
  },
  tabTextActive: {
    color: colors.text.primary,
    fontWeight: "600",
  },
  tabIndicator: {
    position: "absolute",
    bottom: -1,
    height: 2,
    width: "100%",
    backgroundColor: colors.text.primary,
  },
  filtersScroll: {
    marginTop: spacing.m,
  },
  filtersContent: {
    paddingHorizontal: spacing.l,
    gap: spacing.s,
  },
  filterChip: {
    paddingVertical: spacing.s,
    paddingHorizontal: spacing.m,
    borderRadius: 100,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.surface,
  },
  filterChipActive: {
    backgroundColor: colors.text.primary,
    borderColor: colors.text.primary,
  },
  favoritesChip: {
    borderColor: colors.favorite,
  },
  filterText: {
    fontSize: 14,
    color: colors.text.primary,
  },
  filterTextActive: {
    color: colors.surface,
  },
  gridContainer: {
    paddingHorizontal: spacing.l,
    paddingTop: spacing.l,
    paddingBottom: spacing.xxl,
  },
  grid: {
    flexDirection: "row",
    flexWrap: "wrap",
    marginHorizontal: -spacing.xs,
  },
  gridItem: {
    width: "33.33%",
    padding: spacing.xs,
  },
  itemCard: {
    backgroundColor: colors.surface,
    borderRadius: 12,
    overflow: "hidden",
  },
  favoriteButton: {
    position: "absolute",
    top: 8,
    right: 8,
    zIndex: 10,
    backgroundColor: "rgba(255,255,255,0.9)",
    borderRadius: 20,
    padding: 6,
  },
  savedBadge: {
    position: "absolute",
    top: 8,
    left: 8,
    zIndex: 10,
    backgroundColor: colors.text.primary,
    borderRadius: 4,
    paddingVertical: 2,
    paddingHorizontal: 6,
  },
  savedBadgeText: {
    color: "#FFF",
    fontSize: 8,
    fontWeight: "700",
    letterSpacing: 0.5,
  },
  itemImage: {
    width: "100%",
    aspectRatio: 0.75,
    backgroundColor: "#FFFFFF",
  },
  itemInfo: {
    padding: spacing.s,
  },
  itemType: {
    fontSize: 11,
    color: colors.text.primary,
    textTransform: "uppercase",
    letterSpacing: 0.5,
    fontWeight: "600",
  },
  itemColor: {
    fontSize: 10,
    color: colors.text.secondary,
    marginTop: 2,
  },
  outfitItem: {
    width: "50%",
    padding: spacing.xs,
  },
  outfitCard: {
    backgroundColor: colors.surface,
    borderRadius: 12,
    overflow: "hidden",
  },
  outfitImageContainer: {
    width: "100%",
    aspectRatio: 0.75,
    backgroundColor: colors.surfaceHighlight,
    position: "relative",
  },
  outfitLayeredImage: {
    width: "100%",
    height: "100%",
    position: "absolute",
  },
  outfitInfo: {
    padding: spacing.s,
  },
  outfitDate: {
    fontSize: 12,
    color: colors.text.primary,
    fontWeight: "600",
    marginBottom: 2,
  },
  outfitOccasion: {
    fontSize: 11,
    color: colors.text.secondary,
    textTransform: "uppercase",
  },
  emptyContainer: {
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: spacing.xxl,
  },
  emptyText: {
    fontSize: 14,
    color: colors.text.secondary,
    textAlign: "center",
    fontStyle: "italic",
    marginTop: spacing.m,
    lineHeight: 22,
  },
  // Modal Styles
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'flex-end',
  },
  modalHandle: {
    width: 40,
    height: 4,
    backgroundColor: colors.border,
    borderRadius: 2,
    alignSelf: "center",
    marginTop: spacing.s,
    marginBottom: spacing.m,
  },
  // Edit Modal
  editModal: {
    backgroundColor: colors.surface,
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    paddingBottom: 40,
    maxHeight: "80%",
  },
  editHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: spacing.l,
    paddingBottom: spacing.m,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  editTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: colors.text.primary,
  },
  editContent: {
    padding: spacing.l,
  },
  editImage: {
    width: 120,
    height: 160,
    alignSelf: "center",
    borderRadius: 12,
    backgroundColor: "#F5F5F5",
    marginBottom: spacing.l,
  },
  editField: {
    marginBottom: spacing.m,
  },
  editLabel: {
    fontSize: 12,
    color: colors.text.secondary,
    textTransform: "uppercase",
    letterSpacing: 1,
    marginBottom: spacing.xs,
  },
  editInput: {
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 12,
    paddingHorizontal: spacing.m,
    paddingVertical: spacing.m,
    fontSize: 16,
    color: colors.text.primary,
    backgroundColor: colors.surfaceHighlight,
  },
  editActions: {
    flexDirection: "row",
    gap: spacing.m,
    marginTop: spacing.l,
  },
  deleteButton: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: spacing.s,
    backgroundColor: colors.delete,
    paddingVertical: spacing.m,
    borderRadius: 12,
  },
  deleteButtonText: {
    color: "#FFF",
    fontSize: 16,
    fontWeight: "600",
  },
  saveButton: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: spacing.s,
    backgroundColor: colors.text.primary,
    paddingVertical: spacing.m,
    borderRadius: 12,
  },
  saveButtonText: {
    color: "#FFF",
    fontSize: 16,
    fontWeight: "600",
  },
  // Settings Modal
  settingsModal: {
    backgroundColor: colors.surface,
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    paddingBottom: 40,
  },
  settingsHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: spacing.l,
    paddingBottom: spacing.m,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  settingsTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: colors.text.primary,
  },
  settingsItem: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: spacing.l,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  settingsItemText: {
    flex: 1,
    fontSize: 16,
    color: colors.text.primary,
    marginLeft: spacing.m,
  },
  settingsDivider: {
    height: 8,
    backgroundColor: colors.background,
  },
  logoutButton: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: spacing.l,
  },
  logoutText: {
    fontSize: 16,
    color: '#FF3B30',
    marginLeft: spacing.m,
    fontWeight: '500',
  },
});