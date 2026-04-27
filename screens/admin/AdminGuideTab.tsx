/**
 * AdminGuideTab — Edit guide page content and upload hero image
 */
import React, { useState, useEffect } from 'react';
import {
  ActivityIndicator,
  Alert,
  Image,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import { supabase } from '../../lib/supabase';
import { useTranslation } from 'react-i18next';
import { createLogger } from '../../src/utils/logger';
import * as FileSystem from 'expo-file-system/legacy';

const logger = createLogger('AdminGuideTab');

interface GuideContent {
  id: string;
  title: string;
  subtitle: string;
  cta_text: string;
  cta_url: string | null;
  hero_image_url: string | null;
  background_color: string;
  is_active: boolean;
}

export const AdminGuideTab = () => {
  const { t } = useTranslation();
  const [content, setContent] = useState<GuideContent | null>(null);
  const [loading, setLoading] = useState(true);
  const [title, setTitle] = useState('');
  const [subtitle, setSubtitle] = useState('');
  const [ctaText, setCtaText] = useState('');
  const [ctaUrl, setCtaUrl] = useState('');
  const [backgroundColor, setBackgroundColor] = useState('#F5F5F5');
  const [isActive, setIsActive] = useState(true);
  const [localImage, setLocalImage] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

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
      setTitle(data.title);
      setSubtitle(data.subtitle);
      setCtaText(data.cta_text);
      setCtaUrl(data.cta_url || '');
      setBackgroundColor(data.background_color);
      setIsActive(data.is_active);
    } catch (error) {
      logger.error('Error fetching guide content:', error);
      Alert.alert(t('common.error'), t('admin.guide.fetchFailed'));
    } finally {
      setLoading(false);
    }
  };

  const pickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: 'images' as ImagePicker.MediaType,
      quality: 0.85,
      allowsEditing: true,
    });
    if (!result.canceled && result.assets[0] && result.assets[0].uri) {
      setLocalImage(result.assets[0].uri);
    }
  };

  const uploadImage = async (uri: string): Promise<string | null> => {
    try {
      const fileExt = uri.split('.').pop()?.toLowerCase() || 'jpg';
      const safeExt = ['jpg', 'jpeg', 'png', 'webp'].includes(fileExt) ? fileExt : 'jpg';
      const fileName = `guide-hero-${Date.now()}.${safeExt}`;
      const filePath = `guide/${fileName}`;
      const contentType = `image/${safeExt === 'jpg' ? 'jpeg' : safeExt}`;

      const { data: { session } } = await supabase.auth.getSession();
      const token = session?.access_token || process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY;
      const supabaseUrl = process.env.EXPO_PUBLIC_SUPABASE_URL;

      const uploadResult = await FileSystem.uploadAsync(
        `${supabaseUrl}/storage/v1/object/guide-images/${filePath}`,
        uri,
        {
          httpMethod: 'POST',
          headers: {
            Authorization: `Bearer ${token}`,
            apikey: process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY!,
            'Content-Type': contentType,
            'x-upsert': 'true',
          },
        }
      );

      if (uploadResult.status !== 200) {
        let errorMsg = 'Upload failed';
        try {
          const parsed = JSON.parse(uploadResult.body);
          errorMsg = parsed.message || parsed.error || errorMsg;
        } catch (e) {}
        throw new Error(errorMsg);
      }

      const { data } = supabase.storage.from('guide-images').getPublicUrl(filePath);
      return data.publicUrl;
    } catch (err: any) {
      logger.error('Upload exception', err);
      Alert.alert(t('admin.guide.uploadError'), err.message || 'Exception occurred');
      return null;
    }
  };

  const handleSubmit = async () => {
    if (!content) return;
    setSubmitting(true);
    try {
      let finalImageUrl = content.hero_image_url;
      if (localImage) {
        const uploaded = await uploadImage(localImage);
        if (!uploaded) {
          setSubmitting(false);
          return;
        }
        finalImageUrl = uploaded;
      }

      const { error } = await supabase
        .from('guide_page')
        .update({
          title: title.trim(),
          subtitle: subtitle.trim(),
          cta_text: ctaText.trim(),
          cta_url: ctaUrl.trim() || null,
          hero_image_url: finalImageUrl,
          background_color: backgroundColor.trim(),
          is_active: isActive,
        })
        .eq('id', content.id);

      if (error) {
        Alert.alert(t('common.error'), error.message);
        return;
      }

      Alert.alert(t('common.success'), t('admin.guide.saved'));
      setContent({
        ...content,
        title: title.trim(),
        subtitle: subtitle.trim(),
        cta_text: ctaText.trim(),
        cta_url: ctaUrl.trim() || null,
        hero_image_url: finalImageUrl,
        background_color: backgroundColor.trim(),
        is_active: isActive,
      });
      setLocalImage(null);
    } catch (err) {
      logger.error('Submit error', err);
      Alert.alert(t('common.error'), t('admin.guide.saveFailed'));
    } finally {
      setSubmitting(false);
    }
  };

  if (loading) {
    return (
      <View style={s.loadingContainer}>
        <ActivityIndicator size="large" color="#007AFF" />
      </View>
    );
  }

  return (
    <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined} style={{ flex: 1 }}>
      <ScrollView style={s.scroll} keyboardShouldPersistTaps="handled">
        <TouchableOpacity style={s.imagePicker} onPress={pickImage}>
          {localImage ? (
            <Image source={{ uri: localImage }} style={s.imagePreview} />
          ) : content?.hero_image_url ? (
            <Image source={{ uri: content.hero_image_url }} style={s.imagePreview} />
          ) : (
            <View style={s.imagePlaceholder}>
              <Ionicons name="image" size={32} color="#8E8E93" />
              <Text style={s.imagePlaceholderText}>{t('admin.guide.pickImage')}</Text>
            </View>
          )}
        </TouchableOpacity>

        <Text style={s.label}>{t('admin.guide.title')}</Text>
        <TextInput
          style={s.input}
          value={title}
          onChangeText={setTitle}
          placeholder={t('admin.guide.titlePlaceholder')}
        />

        <Text style={s.label}>{t('admin.guide.subtitle')}</Text>
        <TextInput
          style={[s.input, s.textArea]}
          value={subtitle}
          onChangeText={setSubtitle}
          placeholder={t('admin.guide.subtitlePlaceholder')}
          multiline
          numberOfLines={3}
        />

        <Text style={s.label}>{t('admin.guide.ctaText')}</Text>
        <TextInput
          style={s.input}
          value={ctaText}
          onChangeText={setCtaText}
          placeholder={t('admin.guide.ctaTextPlaceholder')}
        />

        <Text style={s.label}>{t('admin.guide.ctaUrl')}</Text>
        <TextInput
          style={s.input}
          value={ctaUrl}
          onChangeText={setCtaUrl}
          placeholder={t('admin.guide.ctaUrlPlaceholder')}
          autoCapitalize="none"
          autoCorrect={false}
          keyboardType="url"
        />

        <Text style={s.label}>{t('admin.guide.backgroundColor')}</Text>
        <View style={s.colorRow}>
          <TextInput
            style={[s.input, s.colorInput]}
            value={backgroundColor}
            onChangeText={setBackgroundColor}
            placeholder="#F5F5F5"
            autoCapitalize="characters"
            maxLength={7}
          />
          <View style={[s.colorPreview, { backgroundColor }]} />
        </View>

        <View style={s.switchRow}>
          <Text style={s.label}>{t('admin.guide.isActive')}</Text>
          <Switch value={isActive} onValueChange={setIsActive} />
        </View>

        <TouchableOpacity
          style={[s.submitBtn, submitting && s.submitBtnDisabled]}
          onPress={handleSubmit}
          disabled={submitting}
        >
          {submitting ? (
            <ActivityIndicator color="#FFF" />
          ) : (
            <Text style={s.submitBtnText}>{t('admin.guide.save')}</Text>
          )}
        </TouchableOpacity>
        <View style={{ height: 40 }} />
      </ScrollView>
    </KeyboardAvoidingView>
  );
};

const s = StyleSheet.create({
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  scroll: {
    flex: 1,
    paddingHorizontal: 20,
  },
  imagePicker: {
    height: 200,
    borderRadius: 16,
    backgroundColor: '#E5E5EA',
    marginBottom: 16,
    overflow: 'hidden',
  },
  imagePreview: {
    width: '100%',
    height: '100%',
    resizeMode: 'contain',
  },
  imagePlaceholder: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  imagePlaceholderText: {
    fontSize: 14,
    color: '#8E8E93',
    marginTop: 6,
  },
  label: {
    fontSize: 13,
    fontWeight: '600',
    color: '#636366',
    marginTop: 12,
    marginBottom: 4,
  },
  input: {
    backgroundColor: '#FFF',
    borderRadius: 10,
    paddingHorizontal: 14,
    paddingVertical: 10,
    fontSize: 15,
    color: '#1C1C1E',
    borderWidth: 1,
    borderColor: '#E5E5EA',
  },
  textArea: {
    minHeight: 70,
    textAlignVertical: 'top',
  },
  colorRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  colorInput: {
    flex: 1,
  },
  colorPreview: {
    width: 40,
    height: 40,
    borderRadius: 8,
    marginLeft: 10,
    borderWidth: 1,
    borderColor: '#E5E5EA',
  },
  switchRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 16,
  },
  submitBtn: {
    backgroundColor: '#007AFF',
    borderRadius: 14,
    paddingVertical: 16,
    alignItems: 'center',
    marginTop: 24,
  },
  submitBtnDisabled: {
    opacity: 0.6,
  },
  submitBtnText: {
    fontSize: 17,
    fontWeight: '600',
    color: '#FFF',
  },
});
