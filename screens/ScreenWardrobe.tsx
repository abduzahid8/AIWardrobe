import React, { useState, useRef } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
  SafeAreaView,
  FlatList,
  StyleSheet,
} from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import { Ionicons } from "@expo/vector-icons";
import axios from "axios";
import * as ImagePicker from "expo-image-picker";
// 👇 Импорт для проверки устройства
import * as Device from 'expo-device';
// @ts-ignore
import { API_URL } from "../api/config";

export default function ScanWardrobeScreen({ navigation }: any) {
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<any>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [loading, setLoading] = useState(false);
  const [scannedItems, setScannedItems] = useState<any[]>([]);

  // Запрос прав
  if (!permission) return <View />;
  if (!permission.granted) {
    return (
      <View style={styles.center}>
        <Text style={{ marginBottom: 20 }}>Нужен доступ к камере</Text>
        <TouchableOpacity onPress={requestPermission} style={styles.btn}>
          <Text style={styles.btnText}>Разрешить</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // --- ФУНКЦИИ ---

  // 1. Выбор видео из галереи (РАБОТАЕТ НА СИМУЛЯТОРЕ)
  const pickVideoFromGallery = async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Videos,
        allowsEditing: true,
        quality: 1,
      });

      if (!result.canceled && result.assets[0].uri) {
        handleUpload(result.assets[0].uri);
      }
    } catch (error) {
      Alert.alert("Ошибка", "Не удалось выбрать видео");
    }
  };

  // 2. Запись видео (ТОЛЬКО РЕАЛЬНЫЙ ТЕЛЕФОН)
  const startRecording = async () => {
    // Проверка на симулятор
    if (!Device.isDevice) {
        Alert.alert(
            "Ошибка Симулятора", 
            "Камера не пишет видео на симуляторе. Нажмите на кнопку ГАЛЕРЕИ (слева), чтобы загрузить готовое видео."
        );
        return;
    }

    if (cameraRef.current) {
      try {
        setIsRecording(true);
        const video = await cameraRef.current.recordAsync({
          maxDuration: 10,
          quality: "480p",
        });
        
        handleUpload(video.uri);
      } catch (e) {
        console.error(e);
        setIsRecording(false);
        Alert.alert("Ошибка", "Не удалось записать видео.");
      }
    }
  };

  const stopRecording = () => {
    if (cameraRef.current && isRecording) {
      cameraRef.current.stopRecording();
      setIsRecording(false);
    }
  };

  // 3. Отправка на сервер
  const handleUpload = async (uri: string) => {
    setLoading(true);
    setIsRecording(false);

    const formData = new FormData();
    // @ts-ignore
    formData.append("video", {
      uri: uri,
      type: "video/mp4",
      name: "scan.mp4",
    });

    try {
      console.log("🚀 Отправка видео...");
      const response = await axios.post(`${API_URL}/scan-wardrobe`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 60000,
      });

      console.log("✅ Найдено:", response.data.items);
      setScannedItems(response.data.items);
      Alert.alert("Успех", `Найдено ${response.data.items.length} вещей!`);

    } catch (error: any) {
      console.error("Upload error:", error);
      Alert.alert("Ошибка", "Сбой анализа. Проверьте сервер.");
    } finally {
      setLoading(false);
    }
  };

  // --- ЭКРАН РЕЗУЛЬТАТОВ ---
  if (scannedItems.length > 0) {
    return (
      <SafeAreaView style={{ flex: 1, backgroundColor: 'white', padding: 16 }}>
        <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 16 }}>
            <TouchableOpacity onPress={() => setScannedItems([])} style={{ marginRight: 10 }}>
                <Ionicons name="arrow-back" size={24} color="black" />
            </TouchableOpacity>
            <Text style={{ fontSize: 24, fontWeight: 'bold' }}>Найдено ✨</Text>
        </View>
        
        <FlatList
          data={scannedItems}
          keyExtractor={(item, index) => index.toString()}
          renderItem={({ item }) => (
            <View style={{ backgroundColor: '#f9fafb', padding: 16, borderRadius: 12, marginBottom: 12, borderWidth: 1, borderColor: '#e5e7eb' }}>
              <View style={{ flexDirection: 'row', justifyContent: 'space-between' }}>
                <Text style={{ fontSize: 18, fontWeight: 'bold' }}>{item.itemType}</Text>
                <Text style={{ color: '#6b7280', fontWeight: '600' }}>{item.season}</Text>
              </View>
              <Text style={{ color: '#4b5563', marginTop: 4 }}>{item.color} • {item.style}</Text>
              <Text style={{ fontSize: 12, color: '#9ca3af', marginTop: 4 }}>{item.description}</Text>
            </View>
          )}
        />
        <TouchableOpacity 
          onPress={() => navigation.navigate("Home")}
          style={{ backgroundColor: 'black', padding: 16, borderRadius: 12, marginTop: 16 }}
        >
          <Text style={{ color: 'white', textAlign: 'center', fontWeight: 'bold', fontSize: 16 }}>Добавить всё</Text>
        </TouchableOpacity>
      </SafeAreaView>
    );
  }

  // --- ЭКРАН ЗАГРУЗКИ ---
  if (loading) {
    return (
      <View style={{ flex: 1, backgroundColor: 'black', justifyContent: 'center', alignItems: 'center' }}>
        <ActivityIndicator size="large" color="#fff" />
        <Text style={{ color: 'white', marginTop: 16, fontSize: 18, fontWeight: 'bold' }}>ИИ смотрит видео...</Text>
      </View>
    );
  }

  // --- ЭКРАН КАМЕРЫ ---
  return (
    <View style={{ flex: 1, backgroundColor: 'black' }}>
      
      {/* 1. ИСПРАВЛЕНИЕ: CameraView пустой, без детей */}
      <CameraView 
        ref={cameraRef}
        style={StyleSheet.absoluteFill} 
        mode="video"
      />

      {/* 2. Интерфейс поверх камеры (Overlay) */}
      <View style={styles.overlay}>
          
          <View style={styles.tipContainer}>
             <Text style={{ color: 'white', fontWeight: '600' }}>
              {isRecording ? "🔴 Запись..." : "Удерживайте для сканирования"}
            </Text>
          </View>
          
          <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'space-around', width: '100%', paddingHorizontal: 20 }}>
            
            {/* Кнопка ГАЛЕРЕИ (Нажимайте её на симуляторе!) */}
            <TouchableOpacity onPress={pickVideoFromGallery} style={styles.iconBtn}>
                <Ionicons name="images" size={28} color="white" />
            </TouchableOpacity>

            {/* Кнопка ЗАПИСИ */}
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
      
      <TouchableOpacity 
        onPress={() => navigation.goBack()}
        style={styles.closeBtn}
      >
        <Ionicons name="close" size={28} color="white" />
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  btn: { backgroundColor: '#3b82f6', padding: 16, marginTop: 16, borderRadius: 8 },
  btnText: { color: 'white', fontWeight: 'bold' },
  overlay: {
    position: 'absolute',
    bottom: 50,
    left: 0,
    right: 0,
    alignItems: 'center',
    zIndex: 10,
  },
  tipContainer: {
    backgroundColor: 'rgba(0,0,0,0.6)',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    marginBottom: 30
  },
  recordBtnOuter: {
    width: 80,
    height: 80,
    borderRadius: 40,
    borderWidth: 5,
    borderColor: 'white',
    justifyContent: 'center',
    alignItems: 'center'
  },
  recordBtnInner: {
    width: 64,
    height: 64,
    borderRadius: 32,
  },
  iconBtn: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: 'rgba(0,0,0,0.4)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  closeBtn: {
    position: 'absolute',
    top: 60,
    left: 20,
    zIndex: 10,
    backgroundColor: 'rgba(0,0,0,0.4)',
    borderRadius: 20,
    padding: 8
  }
});