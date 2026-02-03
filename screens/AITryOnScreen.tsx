import {
  StyleSheet,
  Text,
  View,
  Image,
  TouchableOpacity,
  ActivityIndicator,
  SafeAreaView,
  Alert,
  ScrollView,
  Dimensions,
} from "react-native";
import React, { useState, useEffect, useCallback } from "react";
import { Ionicons } from "@expo/vector-icons";
import Animated from 'react-native-reanimated';
import { BlurView } from 'expo-blur';
import { useNavigation, useFocusEffect } from "@react-navigation/native";
import { useTranslation } from "react-i18next";
import * as ImagePicker from 'expo-image-picker';
import AsyncStorage from "@react-native-async-storage/async-storage";
import axios from "axios";
// @ts-ignore
import { API_URL } from "../api/config";
import AppColors from '../constants/AppColors';

const { width } = Dimensions.get('window');

// Wardrobe item interface
interface WardrobeItem {
  _id: string;
  id?: string;
  type?: string;
  itemType?: string;
  category?: string;
  color?: string;
  imageUrl?: string;
  image?: string;
}


const AITryOnScreen = () => {
  const navigation = useNavigation();
  const { t } = useTranslation();
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [resultImage, setResultImage] = useState<string | null>(null);

  // Image states
  const [humanImage, setHumanImage] = useState<string | null>(null);
  const [clothImage, setClothImage] = useState<string | null>(null);

  // Tab and wardrobe integration
  const [activeTab, setActiveTab] = useState<'upload' | 'wardrobe'>('upload');
  const [wardrobeItems, setWardrobeItems] = useState<WardrobeItem[]>([]);
  const [loadingWardrobe, setLoadingWardrobe] = useState(false);
  const [selectedWardrobeItem, setSelectedWardrobeItem] = useState<WardrobeItem | null>(null);

  // Load wardrobe items
  const loadWardrobeItems = useCallback(async () => {
    try {
      setLoadingWardrobe(true);
      const token = await AsyncStorage.getItem('userToken');

      if (token) {
        const response = await axios.get(`${API_URL}/clothing-items`, {
          headers: { Authorization: `Bearer ${token}` }
        });
        if (response.data) {
          const items = Array.isArray(response.data) ? response.data : response.data.items || [];
          // Filter to clothing items (tops, dresses, jackets)
          const tryableItems = items.filter((item: WardrobeItem) => {
            const category = (item.category || item.type || '').toLowerCase();
            return category.includes('shirt') || category.includes('top') ||
              category.includes('dress') || category.includes('jacket') ||
              category.includes('blouse') || category.includes('sweater') ||
              category.includes('upper');
          });
          setWardrobeItems(tryableItems.length > 0 ? tryableItems : items.slice(0, 20));
        }
      } else {
        // Try loading from local storage
        const localItems = await AsyncStorage.getItem('wardrobeItems');
        if (localItems) {
          setWardrobeItems(JSON.parse(localItems).slice(0, 20));
        }
      }
    } catch (error) {
      console.error('Failed to load wardrobe:', error);
    } finally {
      setLoadingWardrobe(false);
    }
  }, []);

  // Load wardrobe on focus and when switching to wardrobe tab
  useFocusEffect(
    useCallback(() => {
      if (activeTab === 'wardrobe') {
        loadWardrobeItems();
      }
    }, [activeTab, loadWardrobeItems])
  );

  useEffect(() => {
    if (activeTab === 'wardrobe' && wardrobeItems.length === 0) {
      loadWardrobeItems();
    }
  }, [activeTab, loadWardrobeItems, wardrobeItems.length]);

  // Handle selecting wardrobe item
  const handleSelectWardrobeItem = async (item: WardrobeItem) => {
    setSelectedWardrobeItem(item);
    const imageUrl = item.imageUrl || item.image;
    if (imageUrl) {
      // If it's a URL, fetch and convert to base64
      if (imageUrl.startsWith('http')) {
        try {
          // Use the image URL directly - the API should handle it
          setClothImage(imageUrl);
        } catch (error) {
          console.error('Failed to process image:', error);
          Alert.alert('Error', 'Failed to load this item. Please try uploading an image instead.');
        }
      } else {
        // Already base64 or local
        setClothImage(imageUrl);
      }
    }
  };

  // Функция выбора фото
  // Функция выбора фото
  const pickImage = async (setImageFunc: (uri: string) => void) => {
    const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (permissionResult.granted === false) {
      Alert.alert("Permission Required", "You need to allow access to your photos.");
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      // 👇 ИСПОЛЬЗУЕМ СТАРЫЙ ВАРИАНТ (с Options) - он рабочий
      // @ts-ignore
      mediaTypes: ['images'],
      allowsEditing: true,
      aspect: [3, 4],
      quality: 0.5,
      base64: true,
    });

    if (!result.canceled && result.assets && result.assets[0].base64) {
      const base64Image = `data:image/jpeg;base64,${result.assets[0].base64}`;
      setImageFunc(base64Image);
    }
  };

  const handleTryOn = async () => {
    if (!humanImage || !clothImage) {
      Alert.alert(t('aiTryOn.errors.missingPhotos'), t('aiTryOn.errors.missingPhotos'));
      return;
    }

    setLoading(true);
    setResultImage(null);

    try {
      // Add timeout to fetch - HF Spaces can take up to 3 minutes
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 180000); // 180 second timeout for HF Spaces

      const response = await fetch(`${API_URL}/tryon`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          person_image: humanImage,
          garment_image: clothImage,
          garment_type: "upper_body",
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Server error: ${response.status} ${errorText}`);
      }

      const data = await response.json();

      // Debug logging
      console.log("📦 Try-On Response:", {
        success: data.success,
        methodUsed: data.methodUsed,
        hasResultImage: !!data.resultImage,
        resultImageLength: data.resultImage?.length || 0,
        resultImagePrefix: data.resultImage?.substring(0, 50) || 'NONE'
      });

      if (data.error) throw new Error(data.error);
      if (!data.success || !data.resultImage) {
        throw new Error(data.methodUsed === "replicate"
          ? "AI service requires credits. Please try again later."
          : "No image returned from server");
      }

      console.log("✅ Setting result image, length:", data.resultImage.length);
      setResultImage(data.resultImage);
    } catch (error: unknown) {
      console.error("Try-On Error:", error);

      let errorMessage = t('aiTryOn.errors.tryOnFailed');

      const err = error as Error & { name?: string };
      if (err.name === 'AbortError') {
        errorMessage = "Request timed out. Please try again.";
      } else if (err.message?.includes('Network request failed')) {
        errorMessage = "Network error. Check your internet connection.";
      } else if (err.message?.includes('Server error')) {
        errorMessage = "Server is currently unavailable. Please try again later.";
      } else if (err.message?.includes('credits')) {
        errorMessage = "AI service is temporarily limited. Please try again in a minute.";
      }

      Alert.alert(t('common.error'), errorMessage);
    } finally {
      setLoading(false);
    }
  };

  // Save result to wardrobe
  const handleSaveToWardrobe = async () => {
    if (!resultImage) return;

    setSaving(true);

    try {
      const token = await AsyncStorage.getItem('userToken');

      if (!token) {
        Alert.alert('Login Required', 'Please login to save items to your wardrobe');
        setSaving(false);
        return;
      }

      await axios.post(`${API_URL}/clothing-items`, {
        type: 'AI Try-On Result',
        color: 'Mixed',
        style: 'Casual',
        description: 'Virtual try-on generated outfit',
        season: 'All Seasons',
        imageUrl: resultImage
      }, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });

      Alert.alert(
        'Saved! 🎉',
        'Your try-on result has been saved to your wardrobe!',
        [{
          text: 'View Wardrobe',
          onPress: () => (navigation as any).navigate('Profile')
        },
        { text: 'OK' }]
      );
    } catch (error: unknown) {
      console.error('Save error:', error);
      Alert.alert('Error', 'Failed to save. Please try again.');
    } finally {
      setSaving(false);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <Ionicons name="chevron-back" size={28} color="black" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>{t('aiTryOn.title')} ✨</Text>
        <View style={{ width: 28 }} />
      </View>



      <ScrollView contentContainerStyle={styles.scrollContent}>

        {/* Блок выбора картинок */}
        <View style={styles.uploadSection}>

          {/* Человек */}
          <View style={styles.uploadColumn}>
            <Text style={styles.label}>1. {t('aiTryOn.you')}</Text>
            <TouchableOpacity
              style={styles.uploadBox}
              onPress={() => pickImage(setHumanImage)}
              activeOpacity={0.7}
            >
              {humanImage ? (
                <Image source={{ uri: humanImage }} style={styles.uploadedImage} />
              ) : (
                <View style={styles.uploadPlaceholder}>
                  <View style={styles.iconCircle}>
                    <Ionicons name="person" size={32} color="#000" />
                  </View>
                  <Text style={styles.uploadText}>{t('aiTryOn.selectPhoto')}</Text>
                </View>
              )}
            </TouchableOpacity>
          </View>

          {/* Clothing Selection with Tabs */}
          <View style={styles.uploadColumn}>
            <Text style={styles.label}>2. {t('aiTryOn.clothes')}</Text>

            {/* Tab Switcher */}
            <View style={styles.tabContainer}>
              <TouchableOpacity
                style={[styles.tab, activeTab === 'upload' && styles.tabActive]}
                onPress={() => {
                  setActiveTab('upload');
                  setSelectedWardrobeItem(null);
                }}
              >
                <Ionicons
                  name="cloud-upload-outline"
                  size={14}
                  color={activeTab === 'upload' ? '#fff' : '#666'}
                />
                <Text style={[styles.tabText, activeTab === 'upload' && styles.tabTextActive]}>
                  Upload
                </Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.tab, activeTab === 'wardrobe' && styles.tabActive]}
                onPress={() => setActiveTab('wardrobe')}
              >
                <Ionicons
                  name="shirt-outline"
                  size={14}
                  color={activeTab === 'wardrobe' ? '#fff' : '#666'}
                />
                <Text style={[styles.tabText, activeTab === 'wardrobe' && styles.tabTextActive]}>
                  My Wardrobe
                </Text>
              </TouchableOpacity>
            </View>

            {activeTab === 'upload' ? (
              <TouchableOpacity
                style={styles.uploadBox}
                onPress={() => pickImage(setClothImage)}
                activeOpacity={0.7}
              >
                {clothImage && !selectedWardrobeItem ? (
                  <Image source={{ uri: clothImage }} style={styles.uploadedImage} />
                ) : (
                  <View style={styles.uploadPlaceholder}>
                    <View style={styles.iconCircle}>
                      <Ionicons name="shirt" size={32} color="#000" />
                    </View>
                    <Text style={styles.uploadText}>{t('aiTryOn.selectItem')}</Text>
                  </View>
                )}
              </TouchableOpacity>
            ) : (
              <View style={styles.wardrobeSection}>
                {loadingWardrobe ? (
                  <View style={styles.wardrobeLoading}>
                    <ActivityIndicator size="small" color={AppColors.primary} />
                    <Text style={styles.wardrobeLoadingText}>Loading your closet...</Text>
                  </View>
                ) : wardrobeItems.length === 0 ? (
                  <View style={styles.wardrobeEmpty}>
                    <Ionicons name="shirt-outline" size={32} color="#ccc" />
                    <Text style={styles.wardrobeEmptyText}>No items in your wardrobe</Text>
                    <TouchableOpacity
                      style={styles.scanButton}
                      onPress={() => (navigation as any).navigate('WardrobeVideo')}
                    >
                      <Text style={styles.scanButtonText}>Scan Wardrobe</Text>
                    </TouchableOpacity>
                  </View>
                ) : (
                  <ScrollView
                    horizontal
                    showsHorizontalScrollIndicator={false}
                    contentContainerStyle={styles.wardrobeScroll}
                  >
                    {wardrobeItems.map((item) => {
                      const imageUrl = item.imageUrl || item.image;
                      const isSelected = selectedWardrobeItem?._id === item._id;
                      return (
                        <TouchableOpacity
                          key={item._id || item.id}
                          style={[
                            styles.wardrobeItemCard,
                            isSelected && styles.wardrobeItemCardSelected
                          ]}
                          onPress={() => handleSelectWardrobeItem(item)}
                        >
                          {imageUrl ? (
                            <Image
                              source={{ uri: imageUrl }}
                              style={styles.wardrobeItemImage}
                            />
                          ) : (
                            <View style={styles.wardrobeItemPlaceholder}>
                              <Ionicons name="shirt-outline" size={24} color="#ccc" />
                            </View>
                          )}
                          {isSelected && (
                            <View style={styles.selectedBadge}>
                              <Ionicons name="checkmark-circle" size={20} color="#34C759" />
                            </View>
                          )}
                        </TouchableOpacity>
                      );
                    })}
                  </ScrollView>
                )}

                {selectedWardrobeItem && (
                  <View style={styles.selectedInfo}>
                    <Ionicons name="checkmark-circle" size={16} color="#34C759" />
                    <Text style={styles.selectedInfoText}>
                      {selectedWardrobeItem.type || selectedWardrobeItem.category || 'Item'} selected
                    </Text>
                  </View>
                )}
              </View>
            )}
          </View>

        </View>

        {/* Результат */}
        <Text style={styles.label}>3. {t('aiTryOn.result')}</Text>
        <View style={styles.resultContainer}>
          {loading ? (
            <View style={styles.loadingBox}>
              <ActivityIndicator size="large" color="#000" />
              <Text style={{ marginTop: 15, color: "#555", textAlign: 'center' }}>
                {t('aiTryOn.generating')}{"\n"}
                {t('aiTryOn.takesTime')}
              </Text>
            </View>
          ) : resultImage ? (
            <Image source={{ uri: resultImage }} style={styles.resultImage} />
          ) : (
            <View style={styles.placeholder}>
              <Ionicons name="sparkles-outline" size={40} color="#ccc" />
              <Text style={{ color: "#aaa", marginTop: 10 }}>{t('aiTryOn.resultHere')}</Text>
            </View>
          )}
        </View>

        <TouchableOpacity style={styles.button} onPress={handleTryOn} disabled={loading}>
          <Text style={styles.buttonText}>
            {loading ? t('aiTryOn.processing') : `${t('aiTryOn.generate')} ⚡️`}
          </Text>
        </TouchableOpacity>

        {/* Save to Wardrobe Button - only show when result exists */}
        {resultImage && (
          <TouchableOpacity
            style={styles.saveButton}
            onPress={handleSaveToWardrobe}
            disabled={saving}
          >
            <Ionicons name="heart" size={20} color="#fff" style={{ marginRight: 8 }} />
            <Text style={styles.saveButtonText}>
              {saving ? 'Saving...' : 'Save to Wardrobe'}
            </Text>
          </TouchableOpacity>
        )}

      </ScrollView>
    </SafeAreaView>
  );
};

export default AITryOnScreen;

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#f8f9fa" }, // Чуть серый фон для контраста
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    padding: 20,
    backgroundColor: "#fff",
    borderBottomWidth: 1,
    borderColor: "#eee"
  },
  headerTitle: { fontSize: 22, fontWeight: "800", letterSpacing: 0.5 },
  scrollContent: { padding: 20 },

  uploadSection: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 30
  },
  uploadColumn: {
    width: '48%',
  },
  label: {
    fontWeight: "700",
    marginBottom: 10,
    color: "#1a1a1a",
    fontSize: 16,
    marginLeft: 4
  },

  // Стиль кнопок загрузки (карточек)
  uploadBox: {
    width: '100%',
    height: 240, // Высокие кнопки
    backgroundColor: '#fff',
    borderRadius: 24,
    borderWidth: 2,
    borderColor: '#e5e7eb',
    borderStyle: 'dashed',
    overflow: 'hidden',
    // Тени
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 3,
  },
  uploadPlaceholder: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#fafafa'
  },
  iconCircle: {
    width: 60,
    height: 60,
    backgroundColor: '#fff',
    borderRadius: 30,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#eee'
  },
  uploadText: {
    color: '#666',
    fontWeight: '600',
    fontSize: 14
  },
  uploadedImage: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover'
  },

  // Результат
  resultContainer: {
    width: "100%",
    height: 450,
    borderRadius: 24,
    overflow: "hidden",
    backgroundColor: "#fff",
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 20,
    borderWidth: 1,
    borderColor: "#eee",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.1,
    shadowRadius: 20,
    elevation: 5,
  },
  resultImage: { width: "100%", height: "100%", resizeMode: "cover" },
  placeholder: { alignItems: "center" },
  loadingBox: { alignItems: "center" },

  // Главная кнопка
  button: {
    backgroundColor: "#000",
    paddingVertical: 20,
    borderRadius: 30,
    width: "100%",
    alignItems: "center",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.3,
    shadowRadius: 10,
    elevation: 8,
    marginBottom: 40
  },
  buttonText: { color: "#fff", fontSize: 18, fontWeight: "bold", letterSpacing: 1 },
  saveButton: {
    backgroundColor: "#E91E63",
    paddingVertical: 16,
    borderRadius: 30,
    width: "100%",
    alignItems: "center",
    flexDirection: "row",
    justifyContent: "center",
    shadowColor: "#E91E63",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 5,
    marginBottom: 40
  },
  saveButtonText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "bold"
  },

  // Tab Switcher
  tabContainer: {
    flexDirection: 'row',
    marginBottom: 10,
    borderRadius: 12,
    backgroundColor: '#f0f0f0',
    padding: 4,
  },
  tab: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 8,
    borderRadius: 8,
    gap: 4,
  },
  tabActive: {
    backgroundColor: '#000',
  },
  tabText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#666',
  },
  tabTextActive: {
    color: '#fff',
  },

  // Wardrobe Section
  wardrobeSection: {
    minHeight: 200,
  },
  wardrobeLoading: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 40,
  },
  wardrobeLoadingText: {
    marginTop: 10,
    color: '#666',
    fontSize: 14,
  },
  wardrobeEmpty: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 30,
    backgroundColor: '#fafafa',
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#eee',
    borderStyle: 'dashed',
  },
  wardrobeEmptyText: {
    marginTop: 8,
    color: '#999',
    fontSize: 14,
  },
  scanButton: {
    marginTop: 16,
    backgroundColor: '#000',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 20,
  },
  scanButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
  wardrobeScroll: {
    paddingVertical: 10,
  },
  wardrobeItemCard: {
    width: 80,
    height: 110,
    borderRadius: 12,
    backgroundColor: '#fff',
    marginRight: 10,
    overflow: 'hidden',
    borderWidth: 2,
    borderColor: '#eee',
  },
  wardrobeItemCardSelected: {
    borderColor: '#34C759',
    borderWidth: 2,
  },
  wardrobeItemImage: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },
  wardrobeItemPlaceholder: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#f5f5f5',
  },
  selectedBadge: {
    position: 'absolute',
    top: 4,
    right: 4,
    backgroundColor: '#fff',
    borderRadius: 10,
  },
  selectedInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 10,
    paddingHorizontal: 12,
    paddingVertical: 8,
    backgroundColor: '#E8F5E9',
    borderRadius: 8,
  },
  selectedInfoText: {
    marginLeft: 6,
    fontSize: 13,
    color: '#2E7D32',
    fontWeight: '500',
  },

});