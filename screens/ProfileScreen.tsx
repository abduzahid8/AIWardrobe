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
} from "react-native";
import React, { useEffect, useState } from "react";
import useAuthStore from "../store/auth";
import { Ionicons } from "@expo/vector-icons";
import axios from "axios";
import { mpants, mshirts, pants, shoes, skirts, tops } from "../images";
import { useTranslation } from "react-i18next";
import LanguageSelector from "../components/LanguageSelector";
import { colors, shadows, spacing } from "../src/theme";
import AsyncStorage from "@react-native-async-storage/async-storage";

import { useNavigation } from "@react-navigation/native";

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

  // Fetch saved clothes from local storage
  const fetchSavedClothes = React.useCallback(async () => {
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

  useEffect(() => {
    fetchSavedClothes();
  }, [fetchSavedClothes]);

  const popularClothes = React.useMemo(() => [
    ...pants,
    ...tops,
    ...skirts,
    ...mpants,
    ...mshirts,
    ...shoes,
  ].filter((item) => item.image), []);

  // Combine saved clothes with popular clothes for display
  const allClothes = React.useMemo(() => {
    const saved = savedClothes.map(item => ({
      ...item,
      image: item.imageUrl || 'https://via.placeholder.com/150',
      isSaved: true
    }));
    return [...saved, ...popularClothes];
  }, [savedClothes, popularClothes]);

  const fetchOutfits = React.useCallback(async () => {
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
    if (activeCategory == "All") {
      return allClothes;  // Now includes saved items!
    }
    return allClothes.filter((item) => {
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
  }, [activeCategory, allClothes]);

  const sortItems = React.useCallback((items: any[]) => {
    const order = ["shirt", "pants", "skirts", "shoes"];
    return items.sort(
      (a: any, b: any) => order.indexOf(a.type) - order.indexOf(b.type)
    );
  }, []);


  return (
    <View style={styles.container}>
      <SafeAreaView style={styles.safeArea}>
        <ScrollView showsVerticalScrollIndicator={false}>
          {/* Header */}
          <View style={styles.header}>
            <Text style={styles.username}>{username}</Text>
            <View style={styles.headerActions}>
              <LanguageSelector />
              <TouchableOpacity
                style={styles.iconButton}
                onPress={() => (navigation as any).navigate('WardrobeVideo')}
              >
                <Ionicons name="videocam-outline" size={24} color={colors.text.primary} />
              </TouchableOpacity>
              <TouchableOpacity
                style={styles.iconButton}
                onPress={() => setShowSettings(true)}
              >
                <Ionicons name="settings-outline" size={24} color={colors.text.primary} />
              </TouchableOpacity>
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
                <Text style={styles.statNumber}>{followersCount}</Text>
                <Text style={styles.statLabel}>{t('profile.followers')}</Text>
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
                  onPress={() => setActiveTab(tabKey)}
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

          {/* Category Filters */}
          {activeTab === "Clothes" && (
            <ScrollView
              horizontal
              showsHorizontalScrollIndicator={false}
              style={styles.filtersScroll}
              contentContainerStyle={styles.filtersContent}
            >
              {[t('profile.categories.all'), t('profile.categories.tops'), t('profile.categories.bottoms'), t('profile.categories.shoes')].map((cat, idx) => {
                const catKey = ["All", "Tops", "Bottoms", "Shoes"][idx];
                return (
                  <TouchableOpacity
                    onPress={() => setActiveCategory(catKey)}
                    key={catKey}
                    style={[
                      styles.filterChip,
                      activeCategory == catKey && styles.filterChipActive
                    ]}
                  >
                    <Text
                      style={[
                        styles.filterText,
                        activeCategory == catKey && styles.filterTextActive
                      ]}
                    >
                      {cat}
                    </Text>
                  </TouchableOpacity>
                );
              })}
            </ScrollView>
          )}

          {/* Content */}
          {activeTab == "Clothes" && (
            <View style={styles.gridContainer}>
              {filteredClothes.length == 0 ? (
                <Text style={styles.emptyText}>{t('profile.noClothes')}</Text>
              ) : (
                <View style={styles.grid}>
                  {filteredClothes?.map((item, index) => (
                    <View key={index} style={styles.gridItem}>
                      <View style={[styles.itemCard, shadows.soft]}>
                        <Image
                          style={styles.itemImage}
                          source={{ uri: item?.image }}
                          resizeMode="contain"
                        />
                        <View style={styles.itemInfo}>
                          <Text style={styles.itemType}>
                            {item?.type}
                          </Text>
                        </View>
                      </View>
                    </View>
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
        </ScrollView>

        {/* Settings Modal */}
        <Modal
          visible={showSettings}
          animationType="slide"
          transparent={true}
          onRequestClose={() => setShowSettings(false)}
        >
          <View style={styles.modalOverlay}>
            <View style={styles.settingsModal}>
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
    gap: spacing.m,
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
    borderRadius: 0,
    overflow: "hidden",
  },
  itemImage: {
    width: "100%",
    aspectRatio: 0.75,
    backgroundColor: colors.surfaceHighlight,
  },
  itemInfo: {
    padding: spacing.s,
  },
  itemType: {
    fontSize: 11,
    color: colors.text.secondary,
    textTransform: "uppercase",
    letterSpacing: 0.5,
  },
  outfitItem: {
    width: "50%",
    padding: spacing.xs,
  },
  outfitCard: {
    backgroundColor: colors.surface,
    borderRadius: 0,
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
  emptyText: {
    fontSize: 14,
    color: colors.text.secondary,
    textAlign: "center",
    fontStyle: "italic",
    marginTop: spacing.xl,
  },
  // Settings Modal Styles
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'flex-end',
  },
  settingsModal: {
    backgroundColor: colors.surface,
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    paddingBottom: 40,
  },
  settingsHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: spacing.l,
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