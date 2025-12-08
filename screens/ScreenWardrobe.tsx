import React, { useState, useRef, useEffect } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
  StyleSheet,
} from "react-native";
import { Camera, CameraView } from "expo-camera";
import { Ionicons } from "@expo/vector-icons";
import axios from "axios";
import * as ImagePicker from "expo-image-picker";
import * as Device from 'expo-device';
import { useTranslation } from 'react-i18next';
import { API_URL } from "../api/config";
import { useNavigation } from '@react-navigation/native';

export default function ScanWardrobeScreen() {
  const navigation = useNavigation<any>();
  const { t } = useTranslation();
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const cameraRef = useRef<any>(null);

  const [isRecording, setIsRecording] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');
    })();
  }, []);

  if (hasPermission === null) {
    return <View />;
  }
  if (hasPermission === false) {
    return (
      <View style={styles.center}>
        <Text style={{ marginBottom: 20 }}>{t('wardrobe.cameraAccess')}</Text>
        <TouchableOpacity onPress={async () => {
          const { status } = await Camera.requestCameraPermissionsAsync();
          setHasPermission(status === 'granted');
        }} style={styles.btn}>
          <Text style={styles.btnText}>{t('wardrobe.allow')}</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // --- ФУНКЦИЯ ОТПРАВКИ И АНАЛИЗА (ГЛАВНАЯ) ---
  const handleAnalyzeVideo = async (uri: string) => {
    if (!uri) return;

    setIsAnalyzing(true);

    try {
      const formData = new FormData();
      // @ts-ignore
      formData.append('video', {
        uri: uri,
        type: 'video/mp4',
        name: 'upload.mp4',
      });

      console.log("🚀 Sending video to:", `${API_URL}/scan-wardrobe`);

      const response = await axios.post(`${API_URL}/scan-wardrobe`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 180000,
      });
      console.log("✅ AI Response:", response.data);

      // Переходим на экран проверки результатов
      if (response.data.detectedItems) {
        navigation.navigate('ReviewScan', {
          items: response.data.detectedItems
        });
      } else {
        Alert.alert(t('wardrobe.error'), t('wardrobe.noItemsFound'));
      }

    } catch (error: any) {
      console.error("Analysis error:", error);
      Alert.alert(t('wardrobe.error'), t('wardrobe.recognitionError'));
    } finally {
      setIsAnalyzing(false);
    }
  };

  // 1. Выбор видео из галереи
  const pickVideoFromGallery = async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        // @ts-ignore
        mediaTypes: ['videos'],
        allowsEditing: true, // Разрешаем редактирование (это часто включает сжатие)
        quality: 0.5,        // Сжимаем качество
        videoExportPreset: ImagePicker.VideoExportPreset.H264_640x480, // Принудительно 640x480 (SD)
      });

      if (!result.canceled && result.assets[0].uri) {
        handleAnalyzeVideo(result.assets[0].uri);
      }
    } catch (error) {
      Alert.alert(t('wardrobe.error'), t('wardrobe.selectVideoError'));
    }
  };

  // 2. Запись видео
  const startRecording = async () => {
    if (!Device.isDevice) {
      Alert.alert(t('wardrobe.error'), t('wardrobe.cameraNotSimulator'));
      return;
    }

    if (cameraRef.current) {
      try {
        setIsRecording(true);
        const video = await cameraRef.current.recordAsync({
          maxDuration: 10,
          quality: "480p",
        });

        // Когда запись закончится (через 10 сек или вручную)
        if (video) {
          handleAnalyzeVideo(video.uri); // <-- Вызываем новую функцию
        }
      } catch (e) {
        console.error(e);
        setIsRecording(false);
      }
    }
  };

  const stopRecording = () => {
    if (cameraRef.current && isRecording) {
      cameraRef.current.stopRecording();
      setIsRecording(false);
    }
  };

  // --- ЭКРАН ЗАГРУЗКИ (Показываем поверх камеры) ---
  if (isAnalyzing) {
    return (
      <View style={{ flex: 1, backgroundColor: 'black', justifyContent: 'center', alignItems: 'center' }}>
        <ActivityIndicator size="large" color="#fff" />
        <Text style={{ color: 'white', marginTop: 16, fontSize: 18, fontWeight: 'bold' }}>{t('wardrobe.aiAnalyzing')}</Text>
        <Text style={{ color: '#aaa', marginTop: 5 }}>{t('wardrobe.mayTakeMinute')}</Text>
      </View>
    );
  }

  // --- ЭКРАН КАМЕРЫ ---
  return (
    <View style={{ flex: 1, backgroundColor: 'black' }}>
      <CameraView
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
        mode="video"
      />

      <View style={styles.overlay}>
        <View style={styles.tipContainer}>
          <Text style={{ color: 'white', fontWeight: '600' }}>
            {isRecording ? `🔴 ${t('wardrobe.recording')}` : t('wardrobe.holdToScan')}
          </Text>
        </View>

        <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'space-around', width: '100%', paddingHorizontal: 20 }}>
          {/* ГАЛЕРЕЯ */}
          <TouchableOpacity onPress={pickVideoFromGallery} style={styles.iconBtn}>
            <Ionicons name="images" size={28} color="white" />
          </TouchableOpacity>

          {/* ЗАПИСЬ */}
          <TouchableOpacity
            onLongPress={startRecording}
            onPressOut={stopRecording}
            style={styles.recordBtnOuter}
          >
            <View style={[
              styles.recordBtnInner,
              { backgroundColor: isRecording ? '#ef4444' : 'white', transform: [{ scale: isRecording ? 0.7 : 1 }] }
            ]} />
          </TouchableOpacity>

          <View style={{ width: 50 }} />
        </View>
      </View>

      <TouchableOpacity onPress={() => navigation.goBack()} style={styles.closeBtn}>
        <Ionicons name="close" size={28} color="white" />
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  btn: { backgroundColor: '#3b82f6', padding: 16, marginTop: 16, borderRadius: 8 },
  btnText: { color: 'white', fontWeight: 'bold' },
  overlay: { position: 'absolute', bottom: 50, left: 0, right: 0, alignItems: 'center', zIndex: 10 },
  tipContainer: { backgroundColor: 'rgba(0,0,0,0.6)', paddingHorizontal: 16, paddingVertical: 8, borderRadius: 20, marginBottom: 30 },
  recordBtnOuter: { width: 80, height: 80, borderRadius: 40, borderWidth: 5, borderColor: 'white', justifyContent: 'center', alignItems: 'center' },
  recordBtnInner: { width: 64, height: 64, borderRadius: 32 },
  iconBtn: { width: 50, height: 50, borderRadius: 25, backgroundColor: 'rgba(0,0,0,0.4)', justifyContent: 'center', alignItems: 'center' },
  closeBtn: { position: 'absolute', top: 60, left: 20, zIndex: 10, backgroundColor: 'rgba(0,0,0,0.4)', borderRadius: 20, padding: 8 }
});