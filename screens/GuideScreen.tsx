import React, { useState, useEffect } from 'react';
import { View, StyleSheet, ScrollView, TouchableOpacity, Image, Linking, ActivityIndicator, Platform,  } from 'react-native'
import { ScaledText } from '../components/ui/ScaledText';
import { useTranslation } from 'react-i18next';
import { supabase } from '../lib/supabase';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { CachedImage } from '../components/ui/CachedImage';

interface GuideContent {
  title: string;
  subtitle: string;
  cta_text: string;
  cta_url: string | null;
  hero_image_url: string | null;
  background_color: string;
}

export default function GuideScreen() {
  const { t } = useTranslation();
  const insets = useSafeAreaInsets();
  const [content, setContent] = useState<GuideContent | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchGuideContent();
  }, []);

  const fetchGuideContent = async () => {
    try {
      const { data, error } = await supabase
        .from('guide_page')
        .select('*')
        .eq('is_active', true)
        .single();

      if (error) throw error;
      setContent(data);
    } catch (error) {
      console.error('Error fetching guide content:', error);
      // Fall back to i18n strings
      setContent({
        title: t('guide.title'),
        subtitle: t('guide.subtitle'),
        cta_text: t('guide.cta'),
        cta_url: t('guide.ctaUrl'),
        hero_image_url: null,
        background_color: '#F5F5F5',
      });
    } finally {
      setLoading(false);
    }
  };

  const handleCTAPress = async () => {
    const url = content?.cta_url || t('guide.ctaUrl');
    if (url) {
      try {
        const supported = await Linking.canOpenURL(url);
        if (supported) {
          await Linking.openURL(url);
        }
      } catch {
        console.warn('Failed to open URL:', url);
      }
    }
  };

  if (loading) {
    return (
      <View style={[styles.container, { backgroundColor: '#F5F5F5' }]}>
        <ActivityIndicator size="large" color="#000" />
      </View>
    );
  }

  const bgColor = content?.background_color || '#F5F5F5';

  return (
    <ScrollView
      style={[styles.container, { backgroundColor: bgColor }]}
      contentContainerStyle={[styles.contentContainer, { paddingBottom: insets.bottom + 20 }]}
    >
      {/* Hero Section with Phone Mockup */}
      <View style={styles.heroSection}>
        {content?.hero_image_url ? (
          <CachedImage
            uri={content.hero_image_url}
            style={styles.heroImage}
            contentFit="contain"
            fadeIn={false}
          />
        ) : (
          <View style={styles.phoneMockup}>
            <View style={styles.phoneScreen}>
              <View style={styles.mannequinContainer}>
                {/* Placeholder for mannequin/drawing illustration */}
                <Image
                  source={require('../assets/images/mannequin_front.png')}
                  style={styles.mannequinImage}
                  resizeMode="contain"
                />
                <View style={styles.handOverlay}>
                  <ScaledText style={styles.handIcon}>✏️</ScaledText>
                </View>
              </View>
            </View>
          </View>
        )}

        {/* Surrounding sketches - decorative elements */}
        <View style={styles.sketchContainer}>
          <View style={[styles.sketch, styles.sketch1]} />
          <View style={[styles.sketch, styles.sketch2]} />
          <View style={[styles.sketch, styles.sketch3]} />
          <View style={[styles.sketch, styles.sketch4]} />
          <View style={[styles.sketch, styles.sketch5]} />
          <View style={[styles.sketch, styles.sketch6]} />
        </View>
      </View>

      {/* Text Section */}
      <View style={styles.textSection}>
        <ScaledText style={styles.title}>{content?.title || t('guide.title')}</ScaledText>
        <ScaledText style={styles.subtitle}>{content?.subtitle || t('guide.subtitle')}</ScaledText>

        <TouchableOpacity style={styles.ctaButton} onPress={handleCTAPress}>
          <ScaledText style={styles.ctaText}>{content?.cta_text || t('guide.cta')}</ScaledText>
          <View style={styles.ctaUnderline} />
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  contentContainer: {
    flexGrow: 1,
  },
  heroSection: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingTop: 60,
    paddingBottom: 40,
    position: 'relative',
  },
  heroImage: {
    width: 300,
    height: 400,
  },
  phoneMockup: {
    width: 200,
    height: 380,
    backgroundColor: '#1a1a1a',
    borderRadius: 30,
    padding: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 20 },
    shadowOpacity: 0.15,
    shadowRadius: 30,
    elevation: 10,
  },
  phoneScreen: {
    flex: 1,
    backgroundColor: '#fff',
    borderRadius: 22,
    overflow: 'hidden',
  },
  mannequinContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  mannequinImage: {
    width: '100%',
    height: '80%',
  },
  handOverlay: {
    position: 'absolute',
    bottom: 40,
    right: 30,
    backgroundColor: 'rgba(255,255,255,0.9)',
    borderRadius: 20,
    padding: 8,
  },
  handIcon: {
    fontSize: 24,
  },
  sketchContainer: {
    position: 'absolute',
    width: '100%',
    height: '100%',
  },
  sketch: {
    position: 'absolute',
    backgroundColor: '#ddd',
    borderRadius: 8,
  },
  sketch1: {
    width: 60,
    height: 80,
    top: 80,
    left: 40,
    transform: [{ rotate: '-15deg' }],
  },
  sketch2: {
    width: 50,
    height: 70,
    top: 120,
    right: 50,
    transform: [{ rotate: '20deg' }],
  },
  sketch3: {
    width: 40,
    height: 60,
    top: 200,
    left: 60,
    transform: [{ rotate: '10deg' }],
  },
  sketch4: {
    width: 55,
    height: 75,
    top: 250,
    right: 40,
    transform: [{ rotate: '-10deg' }],
  },
  sketch5: {
    width: 45,
    height: 65,
    bottom: 100,
    left: 50,
    transform: [{ rotate: '5deg' }],
  },
  sketch6: {
    width: 50,
    height: 70,
    bottom: 80,
    right: 60,
    transform: [{ rotate: '-5deg' }],
  },
  textSection: {
    paddingHorizontal: 30,
    alignItems: 'center',
    paddingBottom: 60,
  },
  title: {
    fontSize: 18,
    fontWeight: '500',
    color: '#333',
    textAlign: 'center',
    marginBottom: 16,
    letterSpacing: 1,
  },
  subtitle: {
    fontSize: 28,
    fontWeight: '400',
    color: '#000',
    textAlign: 'center',
    lineHeight: 38,
    marginBottom: 40,
    fontFamily: Platform.select({
      ios: 'Georgia',
      android: 'serif',
    }),
  },
  ctaButton: {
    alignItems: 'center',
    paddingVertical: 10,
  },
  ctaText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#000',
    letterSpacing: 2,
  },
  ctaUnderline: {
    height: 2,
    backgroundColor: '#000',
    marginTop: 4,
    width: '100%',
  },
});
