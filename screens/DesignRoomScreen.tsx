import {
  Dimensions,
  Image,
  SafeAreaView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  Platform,
} from "react-native";
import React, { useEffect, useState } from "react";
import { useNavigation, useRoute } from "@react-navigation/native";
import moment from "moment";
import Animated, {
  useAnimatedStyle,
  useSharedValue,
} from "react-native-reanimated";
import { Gesture, GestureDetector } from "react-native-gesture-handler";
import { LinearGradient } from "expo-linear-gradient";
import { Ionicons } from "@expo/vector-icons";
import { colors, shadows, spacing } from "../src/theme";

const { width, height } = Dimensions.get("window");

interface ClothingItem {
  id: number;
  image: string;
  x: number;
  y: number;
  type?: "pants" | "shoes" | "shirt" | "skirts";
  gender?: "m" | "f" | "unisex";
}

const DraggableClothingItem = ({ item }: { item: ClothingItem }) => {
  const translateX = useSharedValue(item?.x);
  const translateY = useSharedValue(item?.y);

  const panGesture = Gesture.Pan()
    .onUpdate((e) => {
      translateX.value = e.translationX + item.x;
      translateY.value = e.translationY + item.y;
    })
    .onEnd(() => {
      item.x = translateX.value;
      item.y = translateY.value;
    });

  const animateStyle = useAnimatedStyle(() => ({
    transform: [
      { translateX: translateX.value },
      { translateY: translateY.value },
    ],
    position: "absolute",
    zIndex: item.type === "shirt" || item.type === "skirts" ? 20 : 10,
  }));

  return (
    <GestureDetector gesture={panGesture}>
      <Animated.View style={animateStyle}>
        <Image
          resizeMode="contain"
          style={{ width: 240, height: item?.type == "shoes" ? 180 : 240 }}
          source={{ uri: item?.image }}
        />
      </Animated.View>
    </GestureDetector>
  );
};

const DesignRoomScreen = () => {
  const route = useRoute();
  const params = (route.params || {}) as {
    selectedItems?: ClothingItem[];
    date?: string;
    savedOutfits?: { [key: string]: any[] };
  };
  const { selectedItems = [], date = moment().format("YYYY-MM-DD"), savedOutfits = {} } = params;
  const [clothes, setClothes] = useState<ClothingItem[]>([]);
  const navigation = useNavigation();

  useEffect(() => {
    const initialClothes = selectedItems.map((item) => {
      const xPosition = width / 2 - 120;
      let yPosition;
      const shirtItem = selectedItems.find((i) => i.type === "shirt");
      const pantsItem = selectedItems.find((i) => i.type == "pants");
      const shoesItems = selectedItems.find((i) => i.type == "shoes");

      if (item.type === "shirt" || item.type === "skirts") {
        yPosition = height / 2 - 240 - 100;
      } else if (item.type === "pants") {
        yPosition = shirtItem ? height / 2 - 100 : height / 2;
      } else if (item.type === "shoes") {
        yPosition = pantsItem || shirtItem ? height / 2 + 100 : height / 2 + 60;
      } else {
        yPosition = height / 2; // Default
      }

      return { ...item, x: xPosition, y: yPosition };
    });
    setClothes(initialClothes);
  }, [JSON.stringify(selectedItems)]);

  return (
    <View style={styles.container}>
      <LinearGradient
        colors={['#ffffff', '#f0f4ff', '#e6eeff']}
        style={StyleSheet.absoluteFill}
      />
      <SafeAreaView style={styles.safeArea}>
        {/* Header */}
        <View style={styles.header}>
          <View>
            <Text style={styles.headerTitle}>Design Room</Text>
            <Text style={styles.headerDate}>{moment(date).format("MMMM Do")}</Text>
          </View>
          <TouchableOpacity
            onPress={() =>
              (navigation.navigate as any)("NewOutfit", {
                selectedItems,
                date,
                savedOutfits,
              })
            }
            style={styles.nextButton}
          >
            <Text style={styles.nextButtonText}>Next</Text>
            <Ionicons name="arrow-forward" size={16} color="#fff" />
          </TouchableOpacity>
        </View>

        {/* Canvas Area */}
        <View style={styles.canvas}>
          {clothes?.map((item) => (
            <DraggableClothingItem key={item.id} item={item} />
          ))}
          {clothes.length === 0 && (
            <View style={styles.emptyState}>
              <Ionicons name="shirt-outline" size={48} color="#cbd5e1" />
              <Text style={styles.emptyStateText}>
                Add clothes to start designing
              </Text>
            </View>
          )}
        </View>

        {/* Toolbar */}
        <View style={styles.toolbarContainer}>
          <View style={[styles.toolbar, shadows.medium]}>
            <TouchableOpacity style={styles.toolButton}>
              <View style={styles.toolIconBg}>
                <Ionicons name="add" size={24} color="#1e293b" />
              </View>
              <Text style={styles.toolLabel}>Add</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.toolButton}
              onPress={() => (navigation as any).navigate('WardrobeVideo')}
            >
              <View style={[styles.toolIconBg, styles.activeToolIcon]}>
                <Ionicons name="videocam" size={24} color="#fff" />
              </View>
              <Text style={styles.toolLabel}>Scan Video</Text>
            </TouchableOpacity>

            <TouchableOpacity style={styles.toolButton}>
              <View style={styles.toolIconBg}>
                <Ionicons name="happy-outline" size={24} color="#1e293b" />
              </View>
              <Text style={styles.toolLabel}>Stickers</Text>
            </TouchableOpacity>

            <TouchableOpacity style={styles.toolButton}>
              <View style={styles.toolIconBg}>
                <Ionicons name="image-outline" size={24} color="#1e293b" />
              </View>
              <Text style={styles.toolLabel}>Bg</Text>
            </TouchableOpacity>
          </View>
        </View>
      </SafeAreaView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  safeArea: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 16,
    zIndex: 50,
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: '700',
    color: '#1e293b',
    fontFamily: Platform.OS === 'ios' ? 'Didot' : 'serif',
  },
  headerDate: {
    fontSize: 14,
    color: '#64748b',
    fontWeight: '500',
  },
  nextButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#1e293b',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    gap: 4,
  },
  nextButtonText: {
    color: '#fff',
    fontWeight: '600',
    fontSize: 14,
  },
  canvas: {
    flex: 1,
    position: 'relative',
    zIndex: 10,
  },
  emptyState: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    opacity: 0.5,
  },
  emptyStateText: {
    marginTop: 12,
    fontSize: 16,
    color: '#64748b',
    fontWeight: '500',
  },
  toolbarContainer: {
    paddingHorizontal: 20,
    paddingBottom: 20,
    zIndex: 50,
  },
  toolbar: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    backgroundColor: '#fff',
    borderRadius: 24,
    paddingVertical: 12,
    paddingHorizontal: 8,
  },
  toolButton: {
    alignItems: 'center',
    gap: 4,
  },
  toolIconBg: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: '#f1f5f9',
    alignItems: 'center',
    justifyContent: 'center',
  },
  activeToolIcon: {
    backgroundColor: '#4f46e5',
    shadowColor: '#4f46e5',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 4,
  },
  toolLabel: {
    fontSize: 12,
    fontWeight: '500',
    color: '#475569',
  },
});

export default DesignRoomScreen;
