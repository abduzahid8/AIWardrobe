import React, { useEffect, useRef, useState, useMemo } from 'react';
import {
  View,
  Text,
  Animated,
  Dimensions,
  StyleSheet,
  Easing,
  Image,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { BlurView } from 'expo-blur';
import Svg, { Line, Defs, LinearGradient as SvgGradient, Stop } from 'react-native-svg';

const { width, height } = Dimensions.get('window');

interface ClothingItem {
  id: string;
  image: string;
  name?: string;
  type?: string;
  category?: string;
}

interface ThinkingStep {
  id: number;
  text: string;
  icon: any;
}

const THINKING_STEPS: ThinkingStep[] = [
  { id: 1, text: 'Scanning your wardrobe...', icon: 'scan-outline' },
  { id: 2, text: 'Analyzing style DNA...', icon: 'color-palette-outline' },
  { id: 3, text: 'Matching combinations...', icon: 'git-merge-outline' },
  { id: 4, text: 'Optimizing outfits...', icon: 'options-outline' },
  { id: 5, text: 'Finalizing your looks...', icon: 'sparkles' },
];

interface AIThinkingAnimationProps {
  styleName?: string;
  clothingItems?: ClothingItem[];
}

// Generate helix DNA strand positions
const generateHelixItems = (count: number, clothingImages: string[]) => {
  const items = [];
  const strands = 2; // Double helix
  const itemsPerStrand = Math.ceil(count / strands);
  
  for (let strand = 0; strand < strands; strand++) {
    for (let i = 0; i < itemsPerStrand; i++) {
      const progress = i / itemsPerStrand;
      const angle = progress * Math.PI * 4 + (strand * Math.PI); // 2 full rotations
      const y = height * 0.85 - progress * height * 0.7; // Start from bottom, go up
      const radius = 90 + Math.sin(progress * Math.PI) * 40; // Varying radius
      
      items.push({
        id: strand * itemsPerStrand + i,
        imageIndex: (strand * itemsPerStrand + i) % clothingImages.length,
        angle,
        baseY: y,
        radius,
        strand,
        progress,
        size: 55 + Math.random() * 25,
        delay: i * 100 + strand * 200,
      });
    }
  }
  return items;
};

// Generate energy nodes that pulse between items
const generateConnections = (count: number) => {
  return Array.from({ length: count }, (_, i) => ({
    id: i,
    startItem: i,
    endItem: (i + 1) % count,
    delay: i * 150,
    speed: 1200 + Math.random() * 800,
  }));
};

export const AIThinkingAnimation: React.FC<AIThinkingAnimationProps> = ({ 
  styleName,
  clothingItems = [],
}) => {
  const [currentStep, setCurrentStep] = useState(0);
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const rotateAnim = useRef(new Animated.Value(0)).current;
  const progressAnim = useRef(new Animated.Value(0)).current;
  const stepFadeAnim = useRef(new Animated.Value(1)).current;

  // Default fallback images
  const clothingImages = useMemo(() => {
    if (clothingItems.length > 0) return clothingItems.map(item => item.image).filter(Boolean);
    return [
      'https://images.unsplash.com/photo-1596755094514-f87e34085b2c?w=200',
      'https://images.unsplash.com/photo-1576566588028-4147f3842f27?w=200',
      'https://images.unsplash.com/photo-1591195853828-11db59a44f6b?w=200',
      'https://images.unsplash.com/photo-1542272454315-4c01d7abdf4a?w=200',
      'https://images.unsplash.com/photo-1549298916-b41fb501e0de?w=200',
      'https://images.unsplash.com/photo-1603252109303-275144992cd7?w=200',
      'https://images.unsplash.com/photo-1551028719-00167b16eac5?w=200',
      'https://images.unsplash.com/photo-1591047139829-d91aecb6caea?w=200',
    ];
  }, [clothingItems]);

  // Generate helix DNA strands
  const helixItems = useMemo(() => generateHelixItems(20, clothingImages), [clothingImages]);
  const connections = useMemo(() => generateConnections(12), []);

  // Animation refs
  const helixRotateAnim = useRef(new Animated.Value(0)).current;
  const helixFloatAnim = useRef(new Animated.Value(0)).current;
  const connectionAnims = useRef(connections.map(() => new Animated.Value(0))).current;

  useEffect(() => {
    // Continuous helix rotation
    Animated.loop(
      Animated.timing(helixRotateAnim, {
        toValue: 1,
        duration: 8000,
        easing: Easing.linear,
        useNativeDriver: true,
      })
    ).start();

    // Floating up/down motion
    Animated.loop(
      Animated.sequence([
        Animated.timing(helixFloatAnim, {
          toValue: 1,
          duration: 3000,
          easing: Easing.inOut(Easing.sin),
          useNativeDriver: true,
        }),
        Animated.timing(helixFloatAnim, {
          toValue: 0,
          duration: 3000,
          easing: Easing.inOut(Easing.sin),
          useNativeDriver: true,
        }),
      ])
    ).start();

    // Pulse center
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, { toValue: 1.15, duration: 800, easing: Easing.inOut(Easing.ease), useNativeDriver: true }),
        Animated.timing(pulseAnim, { toValue: 1, duration: 800, easing: Easing.inOut(Easing.ease), useNativeDriver: true }),
      ])
    ).start();

    // Progress
    Animated.timing(progressAnim, { toValue: 1, duration: 9000, easing: Easing.linear, useNativeDriver: false }).start();

    // Energy connections pulsing
    connectionAnims.forEach((anim: Animated.Value, index: number) => {
      const conn = connections[index];
      Animated.loop(
        Animated.sequence([
          Animated.delay(conn.delay),
          Animated.timing(anim, { toValue: 1, duration: conn.speed, easing: Easing.out(Easing.quad), useNativeDriver: true }),
          Animated.timing(anim, { toValue: 0, duration: 0, useNativeDriver: true }),
        ])
      ).start();
    });

    // Steps
    const stepInterval = setInterval(() => {
      setCurrentStep((prev) => {
        if (prev >= THINKING_STEPS.length - 1) return prev;
        Animated.sequence([
          Animated.timing(stepFadeAnim, { toValue: 0, duration: 150, useNativeDriver: true }),
          Animated.timing(stepFadeAnim, { toValue: 1, duration: 150, useNativeDriver: true }),
        ]).start();
        return prev + 1;
      });
    }, 1800);

    return () => clearInterval(stepInterval);
  }, []);

  const helixRotate = helixRotateAnim.interpolate({ inputRange: [0, 1], outputRange: ['0deg', '360deg'] });
  const progressWidth = progressAnim.interpolate({ inputRange: [0, 1], outputRange: ['0%', '100%'] });

  const getHelixStyle = (item: any) => {
    const baseX = width / 2 + Math.cos(item.angle) * item.radius;
    const baseY = item.baseY;
    const zScale = (Math.sin(item.angle) + 1) / 2 * 0.4 + 0.6;
    const opacity = (Math.sin(item.angle) + 1) / 2 * 0.5 + 0.5;

    return {
      position: 'absolute' as const,
      left: baseX - item.size / 2,
      top: baseY - item.size / 2,
      transform: [{ scale: zScale }],
      opacity,
      zIndex: Math.floor(zScale * 10),
    };
  };

  const currentStepData = THINKING_STEPS[currentStep];

  return (
    <View style={styles.container}>
      <LinearGradient colors={['rgba(43, 92, 233, 0.08)', 'rgba(236, 72, 153, 0.05)', 'transparent']} style={StyleSheet.absoluteFill} />

      {/* DNA Helix container */}
      <Animated.View style={[styles.helixContainer, { transform: [{ rotateY: helixRotate }] }]}>
        {helixItems.map((item: any) => (
          <View key={item.id} style={[styles.helixItem, getHelixStyle(item), { width: item.size, height: item.size * 1.2 }]}>
            <Image source={{ uri: clothingImages[item.imageIndex] }} style={styles.clothingImage} resizeMode="cover" />
          </View>
        ))}
      </Animated.View>

      {/* Energy connection lines */}
      <Svg width={width} height={height} style={StyleSheet.absoluteFill} pointerEvents="none">
        <Defs>
          <SvgGradient id="energyGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <Stop offset="0%" stopColor="#3B82F6" stopOpacity="0" />
            <Stop offset="50%" stopColor="#2B5CE9" stopOpacity="0.6" />
            <Stop offset="100%" stopColor="#8B5CF6" stopOpacity="0" />
          </SvgGradient>
        </Defs>
        {connections.map((conn: any, index: number) => {
          const startItem = helixItems[conn.startItem % helixItems.length];
          const endItem = helixItems[conn.endItem % helixItems.length];
          if (!startItem || !endItem) return null;
          const x1 = width / 2 + Math.cos(startItem.angle) * startItem.radius;
          const y1 = startItem.baseY;
          const x2 = width / 2 + Math.cos(endItem.angle) * endItem.radius;
          const y2 = endItem.baseY;
          return (
            <AnimatedLine
              key={conn.id}
              x1={x1} y1={y1} x2={x2} y2={y2}
              stroke="url(#energyGradient)"
              strokeWidth={2}
              opacity={connectionAnims[index]}
            />
          );
        })}
      </Svg>

      {/* Center AI Core */}
      <Animated.View style={[styles.centerBrain, { transform: [{ scale: pulseAnim }] }]}>
        <BlurView intensity={95} tint="light" style={StyleSheet.absoluteFill}>
          <LinearGradient colors={['rgba(255,255,255,0.98)', 'rgba(255,255,255,0.9)']} style={StyleSheet.absoluteFill} />
        </BlurView>
        <View style={styles.brainInner}>
          <View style={styles.aiIconContainer}>
            <Ionicons name="flask" size={32} color="#2B5CE9" />
          </View>
          <Text style={styles.brainText}>STYLE DNA</Text>
        </View>
        <View style={styles.glowRing} />
        <View style={styles.glowRingOuter} />
      </Animated.View>

      {/* Hexagon pattern background */}
      <View style={styles.hexPattern} pointerEvents="none">
        {[...Array(6)].map((_, i) => (
          <Animated.View key={i} style={[styles.hexagon, { opacity: 0.05 + i * 0.02 }]} />
        ))}
      </View>

      {/* Thinking text */}
      <View style={styles.textContainer}>
        <Animated.View style={{ opacity: stepFadeAnim }}>
          <View style={styles.stepRow}>
            <Ionicons name={currentStepData.icon} size={20} color="#2B5CE9" style={styles.stepIcon} />
            <Text style={styles.stepText}>{currentStepData.text}</Text>
          </View>
        </Animated.View>
        {styleName && <Text style={styles.styleTag}>Synthesizing <Text style={styles.styleTagHighlight}>{styleName}</Text> DNA</Text>}
      </View>

      {/* Progress bar */}
      <View style={styles.progressContainer}>
        <View style={styles.progressBackground}>
          <Animated.View style={[styles.progressFill, { width: progressWidth }]} />
        </View>
        <Text style={styles.progressText}>{Math.round((currentStep / (THINKING_STEPS.length - 1)) * 100)}%</Text>
      </View>
    </View>
  );
};

const AnimatedLine = Animated.createAnimatedComponent(Line);

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  helixContainer: {
    position: 'absolute',
    width: width,
    height: height,
  },
  helixItem: {
    position: 'absolute',
    borderRadius: 14,
    overflow: 'hidden',
    backgroundColor: '#fff',
    shadowColor: '#2B5CE9',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.25,
    shadowRadius: 16,
    elevation: 10,
    borderWidth: 3,
    borderColor: 'rgba(255,255,255,0.9)',
  },
  clothingImage: {
    width: '100%',
    height: '100%',
  },
  centerBrain: {
    width: 120,
    height: 120,
    borderRadius: 60,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 3,
    borderColor: 'rgba(43, 92, 233, 0.4)',
    overflow: 'hidden',
    shadowColor: '#2B5CE9',
    shadowOffset: { width: 0, height: 12 },
    shadowOpacity: 0.4,
    shadowRadius: 24,
    elevation: 15,
    backgroundColor: 'rgba(255,255,255,0.95)',
  },
  brainInner: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  aiIconContainer: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: 'rgba(43, 92, 233, 0.12)',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 6,
    borderWidth: 1,
    borderColor: 'rgba(43, 92, 233, 0.2)',
  },
  brainText: {
    fontSize: 10,
    fontWeight: '900',
    color: '#2B5CE9',
    letterSpacing: 1.2,
  },
  glowRing: {
    position: 'absolute',
    width: 160,
    height: 160,
    borderRadius: 80,
    borderWidth: 2,
    borderColor: 'rgba(43, 92, 233, 0.15)',
    zIndex: -1,
  },
  glowRingOuter: {
    position: 'absolute',
    width: 220,
    height: 220,
    borderRadius: 110,
    borderWidth: 1,
    borderColor: 'rgba(43, 92, 233, 0.08)',
    zIndex: -2,
  },
  hexPattern: {
    position: 'absolute',
    width: 400,
    height: 400,
    alignItems: 'center',
    justifyContent: 'center',
  },
  hexagon: {
    position: 'absolute',
    width: 400,
    height: 400,
    borderRadius: 200,
    borderWidth: 1,
    borderColor: 'rgba(43, 92, 233, 0.1)',
  },
  textContainer: {
    position: 'absolute',
    bottom: height * 0.18,
    alignItems: 'center',
  },
  stepRow: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255,255,255,0.95)',
    paddingHorizontal: 28,
    paddingVertical: 16,
    borderRadius: 30,
    borderWidth: 1,
    borderColor: 'rgba(43, 92, 233, 0.25)',
    shadowColor: '#2B5CE9',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.15,
    shadowRadius: 20,
    elevation: 10,
  },
  stepIcon: {
    marginRight: 12,
  },
  stepText: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1F2937',
  },
  styleTag: {
    fontSize: 14,
    color: '#6B7280',
    marginTop: 16,
    fontWeight: '500',
  },
  styleTagHighlight: {
    color: '#2B5CE9',
    fontWeight: '800',
  },
  progressContainer: {
    position: 'absolute',
    bottom: height * 0.08,
    width: width - 80,
    flexDirection: 'row',
    alignItems: 'center',
  },
  progressBackground: {
    flex: 1,
    height: 8,
    backgroundColor: 'rgba(229, 231, 235, 0.8)',
    borderRadius: 4,
    overflow: 'hidden',
    marginRight: 12,
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#2B5CE9',
    borderRadius: 4,
  },
  progressText: {
    fontSize: 14,
    fontWeight: '700',
    color: '#2B5CE9',
    minWidth: 40,
  },
});

export default AIThinkingAnimation;
