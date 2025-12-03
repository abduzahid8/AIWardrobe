import {
  StyleSheet,
  Text,
  View,
  Image,
  TouchableOpacity,
  ActivityIndicator,
  SafeAreaView,
  Alert,
  ScrollView
} from "react-native";
import React, { useState } from "react";
import { Ionicons } from "@expo/vector-icons";
import { useNavigation } from "@react-navigation/native";
import { useTranslation } from "react-i18next";
import * as ImagePicker from 'expo-image-picker';
// @ts-ignore
import { API_URL } from "../api/config";


const AITryOnScreen = () => {
  const navigation = useNavigation();
  const { t } = useTranslation();
  const [loading, setLoading] = useState(false);
  const [resultImage, setResultImage] = useState<string | null>(null);

  // Храним сами картинки (Base64 или URI)
  const [humanImage, setHumanImage] = useState<string | null>(null);
  const [clothImage, setClothImage] = useState<string | null>(null);

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
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
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
      const response = await fetch(`${API_URL}/try-on`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          human_image: humanImage,
          garment_image: clothImage,
          description: "clothing",
        }),
      });

      const data = await response.json();

      if (data.error) throw new Error(data.error);

      setResultImage(data.image);
    } catch (error: any) {
      console.log(error);
      Alert.alert(t('common.error'), t('aiTryOn.errors.tryOnFailed'));
    } finally {
      setLoading(false);
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

          {/* Одежда */}
          <View style={styles.uploadColumn}>
            <Text style={styles.label}>2. {t('aiTryOn.clothes')}</Text>
            <TouchableOpacity
              style={styles.uploadBox}
              onPress={() => pickImage(setClothImage)}
              activeOpacity={0.7}
            >
              {clothImage ? (
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
});