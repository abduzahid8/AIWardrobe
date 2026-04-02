/**
 * ProfileScreen - User Profile
 * Minimalist Liquid Glass design with Looks and Trips tabs
 */

import React, { useState, useCallback, useMemo } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Dimensions,
  ScrollView,
  Image,
  TouchableOpacity,
  StatusBar,
  Modal,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation, useFocusEffect } from '@react-navigation/native';
import * as Haptics from 'expo-haptics';
import Animated, {
  FadeInDown,
  useAnimatedStyle,
  useSharedValue,
  withSpring,
} from 'react-native-reanimated';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { supabase } from '../lib/supabase';
import useAuthStore from '../store/auth';
import { LiquidGlass2026Theme } from '../constants/LiquidGlass2026Theme';
import { NavigationMenu } from '../src/components/NavigationMenu';

const { width, height } = Dimensions.get('window');
const { colors, spacing, radius, typography } = LiquidGlass2026Theme;

// PressableScale props type
interface PressableScaleProps {
  children: React.ReactNode;
  onPress: () => void;
  style?: object;
}

// Saved outfit type
interface OutfitType {
  _id: string;
  date?: string;
  occasion?: string;
  items: any[];
  image?: string; // If available directly, or derived from items
}

// Pressable with scale animation
const PressableScale = ({ children, onPress, style }: PressableScaleProps) => {
  const scale = useSharedValue(1);

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: scale.value }],
  }));

  return (
    <TouchableOpacity
      activeOpacity={1}
      onPressIn={() => {
        scale.value = withSpring(0.97);
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
      }}
      onPressOut={() => scale.value = withSpring(1)}
      onPress={onPress}
      style={style}
    >
      <Animated.View style={animatedStyle}>{children}</Animated.View>
    </TouchableOpacity>
  );
};

// Tab Button
const TabButton = ({ title, isActive, onPress }: { title: string; isActive: boolean; onPress: () => void }) => (
  <TouchableOpacity
    style={[styles.tabButton, isActive && styles.tabButtonActive]}
    onPress={() => {
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
      onPress();
    }}
    accessibilityLabel={`${title} tab`}
    accessibilityRole="tab"
    accessibilityState={{ selected: isActive }}
  >
    <Text style={[styles.tabText, isActive && styles.tabTextActive]}>{title}</Text>
  </TouchableOpacity>
);

const ProfileScreen = () => {
  const navigation = useNavigation();
  const { user, logout, deleteAccount } = useAuthStore();
  const [activeTab, setActiveTab] = useState<'looks' | 'trips'>('looks');
  const [showNavMenu, setShowNavMenu] = useState(false);

  // User data
  // User data
  const userName = user?.username || 'Username 14';
  const userAvatar = user?.profile_image || 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=200';
  const location = 'Monnaie, Ville De Paris';

  // Data states
  const [outfits, setOutfits] = useState<OutfitType[]>([]);
  const [loading, setLoading] = useState(false);

  useFocusEffect(useCallback(() => {
    fetchOutfits();
  }, [user?.id]));

  const fetchOutfits = async () => {
    if (!user?.id) return;
    setLoading(true);
    try {
      const { data, error } = await supabase
        .from('saved_outfits')
        .select('*')
        .eq('user_id', user.id)
        .order('created_at', { ascending: false });

      if (error) {
        console.log("Error fetching outfits", error);
        setOutfits([]);
        return;
      }

      if (data) {
        const mappedOutfits = data.map(o => ({
          _id: o.id,
          date: o.date,
          occasion: o.occasion,
          items: o.items || [],
          // If no explicit image, we might construct one from items later or show a placeholder
          image: o.image_url
        }));
        setOutfits(mappedOutfits);
      }
    } catch (error) {
      console.log("Error fetching outfits", error);
    } finally {
      setLoading(false);
    }
  };

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
            Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
            await AsyncStorage.removeItem("userToken");
            logout();
          }
        }
      ]
    );
  };

  const handleDeleteAccount = () => {
    Alert.alert(
      "Delete Account",
      "This will permanently delete your account and all data (clothing items, outfits, wear history). This action cannot be undone.",
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Delete Permanently",
          style: "destructive",
          onPress: () => {
            Alert.alert(
              "Final Confirmation",
              "Are you absolutely sure? All your data will be permanently erased.",
              [
                { text: "Keep My Account", style: "cancel" },
                {
                  text: "Yes, Delete Everything",
                  style: "destructive",
                  onPress: async () => {
                    try {
                      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
                      await deleteAccount();
                      await AsyncStorage.removeItem("userToken");
                    } catch (err: any) {
                      Alert.alert("Error", err.message || "Failed to delete account. Please try again.");
                    }
                  },
                },
              ]
            );
          },
        },
      ]
    );
  };

  const handleSettings = () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    // Navigate to settings or show modal
    Alert.alert("Settings", "Settings menu coming soon!");
  };

  return (
    <View style={styles.container}>
      <StatusBar barStyle="dark-content" backgroundColor={colors.background.primary} />
      <SafeAreaView style={styles.safeArea} edges={['top']}>

        {/* Header - Alta layout */}
        <View style={styles.header}>
          {/* Left: Empty spacer for balance */}
          <View style={styles.headerIcon} />

          {/* Right: Menu + Settings buttons */}
          <View style={styles.headerRight}>
            <TouchableOpacity style={styles.headerIcon} onPress={() => setShowNavMenu(true)} accessibilityLabel="Open navigation menu" accessibilityRole="button">
              <Ionicons name="menu" size={24} color={colors.text.primary} />
            </TouchableOpacity>
            <TouchableOpacity style={styles.headerIcon} onPress={handleSettings} accessibilityLabel="Settings" accessibilityRole="button">
              <Ionicons name="settings-outline" size={24} color={colors.text.primary} />
            </TouchableOpacity>
          </View>
        </View>

        <ScrollView
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          {/* Profile Section */}
          <View style={styles.profileSection}>
            <View style={styles.avatarCircle}>
              <Image
                source={{ uri: userAvatar }}
                style={styles.avatarImage}
                accessibilityLabel={`${userName}'s profile photo`}
              />
            </View>
            <Text style={styles.username} accessibilityRole="header">{userName}</Text>
            <View style={styles.locationRow}>
              <Text style={styles.location}>{location}</Text>
              <Ionicons name="locate-outline" size={14} color={colors.text.tertiary} />
            </View>
          </View>

          {/* Tabs */}
          <View style={styles.tabsContainer}>
            <TabButton
              title="Looks"
              isActive={activeTab === 'looks'}
              onPress={() => setActiveTab('looks')}
            />
            <TabButton
              title="Trips"
              isActive={activeTab === 'trips'}
              onPress={() => setActiveTab('trips')}
            />
          </View>

          {/* Tab Content */}
          {activeTab === 'looks' ? (
            outfits.length === 0 ? (
              <View style={styles.emptyState}>
                <Image
                  // Use a placeholder hanger icon if not available
                  source={{ uri: 'https://cdn-icons-png.flaticon.com/512/3531/3531853.png' }}
                  style={{ width: 120, height: 80, opacity: 0.2, marginBottom: 20, tintColor: '#CCC' }}
                  resizeMode="contain"
                />
                <Text style={styles.emptyTitle}>Style it, save it</Text>
                <Text style={styles.emptySubtitle}>Your saved looks live here. Create outfits with AI or add your own to get started!</Text>
                <TouchableOpacity
                  style={styles.actionButton}
                  onPress={() => (navigation as any).navigate('AIOutfit')}
                  accessibilityLabel="Create new look"
                  accessibilityRole="button"
                >
                  <Text style={styles.actionButtonText}>Create new look</Text>
                </TouchableOpacity>
              </View>
            ) : (
              <View style={styles.looksGrid}>
                {outfits.map((outfit, i) => (
                  <View key={outfit._id} style={styles.lookCard}>
                    {/* If we have a composed image for the outfit, show it. Otherwise verify items */}
                    {outfit.image ? (
                      <Image source={{ uri: outfit.image }} style={styles.lookImage} />
                    ) : (
                      <View style={styles.collageContainer}>
                        {outfit.items.slice(0, 4).map((item: any, idx) => (
                          <Image
                            key={idx}
                            source={{ uri: item.image || item.imageUrl }}
                            style={styles.collageImage}
                          />
                        ))}
                      </View>
                    )}
                  </View>
                ))}
              </View>
            )
          ) : (
            <View style={styles.emptyState}>
              <Ionicons name="airplane-outline" size={64} color={colors.text.tertiary} style={{ marginBottom: 16, opacity: 0.5 }} />
              <Text style={styles.emptyTitle}>No trips planned</Text>
              <Text style={styles.emptySubtitle}>Plan a trip to get packing suggestions</Text>
            </View>
          )}

          {/* Quick Links */}
          <View style={styles.quickLinksSection}>
            <TouchableOpacity
              style={styles.quickLink}
              onPress={() => (navigation as any).navigate('WardrobeAnalytics')}
              accessibilityLabel="Wardrobe Analytics"
              accessibilityRole="button"
            >
              <Ionicons name="analytics-outline" size={20} color={colors.text.secondary} />
              <Text style={styles.quickLinkText}>Wardrobe Analytics</Text>
              <Ionicons name="chevron-forward" size={16} color={colors.text.tertiary} />
            </TouchableOpacity>

            <View style={styles.quickLinkDivider} />

            <TouchableOpacity
              style={styles.quickLink}
              onPress={() => (navigation as any).navigate('PrivacyPolicy')}
              accessibilityLabel="Privacy Policy"
              accessibilityRole="button"
            >
              <Ionicons name="shield-checkmark-outline" size={20} color={colors.text.secondary} />
              <Text style={styles.quickLinkText}>Privacy Policy</Text>
              <Ionicons name="chevron-forward" size={16} color={colors.text.tertiary} />
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.quickLink}
              onPress={() => (navigation as any).navigate('TermsOfService')}
              accessibilityLabel="Terms of Service"
              accessibilityRole="button"
            >
              <Ionicons name="document-text-outline" size={20} color={colors.text.secondary} />
              <Text style={styles.quickLinkText}>Terms of Service</Text>
              <Ionicons name="chevron-forward" size={16} color={colors.text.tertiary} />
            </TouchableOpacity>
          </View>

          {/* Logout Option (Subtle at bottom) */}
          <TouchableOpacity style={styles.logoutButton} onPress={handleLogout} accessibilityLabel="Log out" accessibilityRole="button">
            <Text style={styles.logoutText}>Log Out</Text>
          </TouchableOpacity>

          {/* Delete Account */}
          <TouchableOpacity style={styles.deleteAccountButton} onPress={handleDeleteAccount} accessibilityLabel="Delete account" accessibilityRole="button">
            <Text style={styles.deleteAccountText}>Delete Account</Text>
          </TouchableOpacity>

          <View style={{ height: 100 }} />
        </ScrollView>

      </SafeAreaView>

      {/* Navigation Menu */}
      <NavigationMenu visible={showNavMenu} onClose={() => setShowNavMenu(false)} />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background.primary,
  },
  safeArea: {
    flex: 1,
  },

  // Header
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.screenPadding,
    paddingVertical: spacing.md,
  },
  headerIcon: {
    padding: spacing.xs,
  },
  headerRight: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  avatarPill: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: colors.background.secondary,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: radius.pill,
  },
  avatarPillText: {
    fontSize: 13,
    fontWeight: '600',
    color: colors.text.primary,
  },

  // Content
  scrollContent: {
    paddingTop: spacing.md,
  },

  // Profile
  profileSection: {
    alignItems: 'center',
    marginBottom: spacing.xl,
  },
  avatarCircle: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: colors.background.secondary,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: spacing.md,
    overflow: 'hidden',
    borderWidth: 2,
    borderColor: colors.background.primary,
    shadowColor: "#0A1931",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  avatarImage: {
    width: '100%',
    height: '100%',
  },
  username: {
    fontSize: 22,
    fontWeight: '700',
    color: colors.text.primary,
    marginBottom: 4,
  },
  locationRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  location: {
    fontSize: 14,
    color: colors.text.tertiary,
  },

  // Friends Card
  friendsCard: {
    marginHorizontal: spacing.screenPadding,
    backgroundColor: colors.background.secondary, // Light gray bg
    borderRadius: 16, // Use standard radius
    padding: spacing.lg, // Use sufficient padding
    marginBottom: spacing.xl,
    alignItems: 'center', // Center everything horizontally
  },
  friendsTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: colors.text.primary,
    marginBottom: 4,
    textAlign: 'center', // Center text
  },
  friendsSubtitle: {
    fontSize: 14,
    fontWeight: '400',
    color: colors.text.secondary,
    textAlign: 'center', // Center text
    marginBottom: spacing.lg,
    maxWidth: '100%',
  },
  avatarsRow: {
    flexDirection: 'row',
    marginBottom: spacing.lg,
    height: 120, // Taller for full body/larger images
    alignItems: 'flex-end',
    justifyContent: 'center', // Center images
    width: '100%',
    gap: spacing.md,
  },
  friendAvatar: {
    width: 80, // Wider to allow cover to work better without zooming in too much
    height: 120, // Full body height
    borderRadius: 0,
    resizeMode: 'cover', // Cover to fill height, relying on matching background color
  },
  avatarOverlap: {
    marginLeft: 0, // No overlap
  },
  addFriendsButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: '#0A1931',
    paddingVertical: 14, // Taller button
    borderRadius: radius.pill,
    alignSelf: 'center', // Center button
    width: 236, // Proportional width to match avatar spacing
  },
  addFriendsText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#FFF',
  },

  // Tabs
  tabsContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: spacing.xl,
    marginBottom: spacing.lg,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.subtle,
    marginHorizontal: spacing.screenPadding,
  },
  tabButton: {
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.sm,
  },
  tabButtonActive: {
    borderBottomWidth: 2,
    borderBottomColor: colors.text.primary,
  },
  tabText: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.text.tertiary,
  },
  tabTextActive: {
    color: colors.text.primary,
  },

  // Empty State
  emptyState: {
    alignItems: 'center',
    paddingVertical: spacing.xxl,
    paddingHorizontal: spacing.lg,
  },
  emptyTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: colors.text.primary,
    marginBottom: 8,
    marginTop: 16,
  },
  emptySubtitle: {
    fontSize: 14,
    color: colors.text.secondary,
    textAlign: 'center',
    marginBottom: spacing.xl,
    maxWidth: 300,
    lineHeight: 20,
  },
  actionButton: {
    backgroundColor: colors.background.secondary,
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: radius.pill,
  },
  actionButtonText: {
    color: colors.text.primary,
    fontWeight: '600',
    fontSize: 14,
  },

  // Looks Grid
  looksGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    paddingHorizontal: spacing.screenPadding,
    gap: spacing.sm,
  },
  lookCard: {
    width: (width - (spacing.screenPadding * 2) - spacing.sm) / 2,
    aspectRatio: 0.75,
    backgroundColor: colors.background.secondary,
    borderRadius: radius.lg,
    overflow: 'hidden',
    marginBottom: spacing.sm,
  },
  lookImage: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },
  collageContainer: {
    flex: 1,
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  collageImage: {
    width: '50%',
    height: '50%',
    resizeMode: 'cover',
  },

  // Logout
  logoutButton: {
    alignSelf: 'center',
    marginTop: spacing.xxl * 2,
    padding: spacing.md,
  },
  logoutText: {
    color: colors.text.tertiary,
    fontSize: 14,
    fontWeight: '500',
  },
  deleteAccountButton: {
    alignSelf: 'center',
    marginTop: spacing.md,
    padding: spacing.md,
  },
  deleteAccountText: {
    color: '#FF3B30',
    fontSize: 13,
    fontWeight: '500',
  },

  // Quick Links
  quickLinksSection: {
    marginHorizontal: spacing.screenPadding,
    marginTop: spacing.xl,
    backgroundColor: colors.background.secondary,
    borderRadius: radius.xl,
    overflow: 'hidden',
  },
  quickLink: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 14,
    paddingHorizontal: spacing.lg,
    gap: spacing.md,
  },
  quickLinkText: {
    flex: 1,
    fontSize: 15,
    fontWeight: '500',
    color: colors.text.primary,
  },
  quickLinkDivider: {
    height: 1,
    backgroundColor: colors.border.subtle,
    marginHorizontal: spacing.lg,
  },
});

export default ProfileScreen;