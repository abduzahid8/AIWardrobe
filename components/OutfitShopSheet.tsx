import { shopPriceLabel } from '../src/utils/shopPriceLabel';
import React, { useCallback } from 'react';
import {
  View,
  StyleSheet,
  Modal,
  TouchableOpacity,
  Linking,
  ScrollView,
  Pressable,
} from 'react-native';

import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import { useTranslation } from 'react-i18next';
import { ScaledText } from './ui/ScaledText';
import { getShopDestination } from '../src/utils/shopDestination';
import { ExpoImage } from '../src/components/MediaImage';


export interface ShopSheetItem {
  id: string | number;
  image?: string;
  imageUrl?: string;
  name?: string;
  type?: string;
  brand?: string;
  price?: number;
  currency?: string;
  shopUrl?: string;
  sourceUrl?: string;
  macroCategory?: string;
  category?: string;
  isShopItem?: boolean;
}

interface OutfitShopSheetProps {
  visible: boolean;
  onClose: () => void;
  items: ShopSheetItem[];
  occasion?: string;
  style?: string;
}

const getItemImage = (item: ShopSheetItem): string =>
  (typeof item.image === 'string' && item.image) ||
  (typeof item.imageUrl === 'string' && item.imageUrl) ||
  '';

const getShopUrl = (item: ShopSheetItem): string | undefined =>
  item.shopUrl || item.sourceUrl || undefined;

const getRetailerLabel = (url?: string): string | null => {
  if (!url) return null;
  try {
    const host = new URL(url).hostname.replace(/^www\./, '');
    return host;
  } catch {
    return null;
  }
};

const getCategoryLabel = (item: ShopSheetItem): string => {
  const raw = item.macroCategory || item.category || item.type || '';
  const lower = String(raw).toLowerCase();
  if (lower.includes('outer')) return 'Outerwear';
  if (lower === 'top' || lower.includes('upper') || lower.includes('shirt') || lower.includes('tee')) return 'Top';
  if (lower === 'bottom' || lower.includes('lower') || lower.includes('pant') || lower.includes('jean') || lower.includes('trouser')) return 'Bottom';
  if (lower.includes('shoe') || lower.includes('sneaker') || lower.includes('loafer') || lower.includes('boot')) return 'Shoes';
  return raw ? String(raw) : 'Item';
};


export const OutfitShopSheet = ({ visible, onClose, items, occasion, style }: OutfitShopSheetProps) => {
  const { t, i18n } = useTranslation();

  const openShopUrl = useCallback(async (url?: string, itemName?: string) => {
    if (!url && !itemName) return;
    // Only http(s) retailer links may reach Linking; hostile schemes
    // (tel:, javascript:, app deep links) fall back to a Shopping search.
    const target = getShopDestination({ shopUrl: url, name: itemName });
    try {
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
      await Linking.openURL(target);
    } catch {
      // Linking failures are non-fatal — user stays on the sheet.
    }
  }, []);

  return (
    <Modal visible={visible} transparent animationType="slide" onRequestClose={onClose} statusBarTranslucent>
      <View style={styles.overlay}>
        <Pressable style={StyleSheet.absoluteFill} onPress={onClose} />
        <View style={styles.sheet}>
          <View style={styles.handle} />
          <View style={styles.header}>
            <View style={styles.headerCopy}>
              <ScaledText style={styles.title}>{t('home.shopThisLook', 'Shop this look')}</ScaledText>
              {(occasion || style) && (
                <ScaledText style={styles.subtitle} numberOfLines={1}>
                  {[occasion, style].filter(Boolean).join(' · ')}
                </ScaledText>
              )}
            </View>
            <TouchableOpacity onPress={onClose} style={styles.closeButton} accessibilityRole="button" accessibilityLabel={t('common.close', 'Close')}>
              <Ionicons name="close" size={20} color="#0A1931" />
            </TouchableOpacity>
          </View>
          <ScaledText style={styles.hint}>{t('home.shopSheetHint', 'Pick an item, then choose where to buy it.')}</ScaledText>

          <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.list} style={styles.scroll}>
            {items.map((item, index) => {
              const image = getItemImage(item);
              const shopUrl = getShopUrl(item);
              const retailer = getRetailerLabel(shopUrl);
              const price = shopPriceLabel(item, i18n?.resolvedLanguage || i18n?.language);
              const name = item.name || item.type || `Item ${index + 1}`;
              return (
                <View key={String(item.id ?? `${name}-${index}`)} style={styles.row}>
                  <View style={styles.thumbWrap}>
                    {image ? (
                      <ExpoImage source={{ uri: image }} style={styles.thumb} contentFit="cover" cachePolicy="memory-disk" />
                    ) : (
                      <View style={[styles.thumb, styles.thumbPlaceholder]}>
                        <Ionicons name="shirt-outline" size={22} color="#8A97A8" />
                      </View>
                    )}
                  </View>
                  <View style={styles.rowInfo}>
                    <ScaledText style={styles.rowCategory}>{getCategoryLabel(item).toUpperCase()}</ScaledText>
                    <ScaledText style={styles.rowName} numberOfLines={2}>{name}</ScaledText>
                    {(item.brand || price) && (
                      <ScaledText style={styles.rowMeta} numberOfLines={1}>
                        {[item.brand, price].filter(Boolean).join(' · ')}
                      </ScaledText>
                    )}
                    {retailer ? (
                      <View style={styles.retailerRow}>
                        <Ionicons name="storefront-outline" size={12} color="#58734E" />
                        <ScaledText style={styles.retailerText} numberOfLines={1}>{retailer}</ScaledText>
                      </View>
                    ) : (
                      <View style={styles.retailerRow}>
                        <Ionicons name="shirt-outline" size={12} color="#8A97A8" />
                        <ScaledText style={styles.wardrobeText}>{t('home.fromWardrobe', 'From your wardrobe')}</ScaledText>
                      </View>
                    )}
                  </View>
                  <TouchableOpacity
                    style={styles.buyButton}
                    onPress={() => openShopUrl(shopUrl, name)}
                    accessibilityRole="button"
                    accessibilityLabel={t('home.buyItem', 'Buy {{name}}', { name })}
                  >
                    <Ionicons name="cart-outline" size={15} color="#FFFFFF" />
                    <ScaledText style={styles.buyButtonText}>{t('home.buy', 'Buy')}</ScaledText>
                  </TouchableOpacity>
                </View>
              );
            })}
          </ScrollView>
        </View>
      </View>
    </Modal>
  );
};

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    justifyContent: 'flex-end',
    backgroundColor: 'rgba(10,25,49,0.45)',
  },
  sheet: {
    maxHeight: '78%',
    backgroundColor: '#FFFFFF',
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    paddingHorizontal: 18,
    paddingTop: 10,
    paddingBottom: 28,
    shadowColor: '#0A1931',
    shadowOffset: { width: 0, height: -8 },
    shadowOpacity: 0.18,
    shadowRadius: 24,
    elevation: 12,
  },
  handle: {
    alignSelf: 'center',
    width: 44,
    height: 5,
    borderRadius: 3,
    backgroundColor: '#E2E8F0',
    marginBottom: 12,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    gap: 12,
  },
  headerCopy: {
    flex: 1,
  },
  title: {
    fontSize: 19,
    fontWeight: '800',
    color: '#0A1931',
  },
  subtitle: {
    fontSize: 12,
    color: '#718096',
    marginTop: 2,
  },
  closeButton: {
    width: 34,
    height: 34,
    borderRadius: 17,
    backgroundColor: '#F1F5FA',
    alignItems: 'center',
    justifyContent: 'center',
  },
  hint: {
    fontSize: 12,
    color: '#718096',
    marginTop: 6,
    marginBottom: 12,
  },
  scroll: {
    flexGrow: 0,
  },
  list: {
    gap: 12,
    paddingBottom: 8,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    padding: 10,
    borderRadius: 18,
    backgroundColor: '#F7F9FC',
    borderWidth: 1,
    borderColor: 'rgba(23,58,101,0.07)',
  },
  thumbWrap: {
    width: 62,
    height: 72,
    borderRadius: 14,
    overflow: 'hidden',
    backgroundColor: '#FFFFFF',
  },
  thumb: {
    width: '100%',
    height: '100%',
  },
  thumbPlaceholder: {
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#EEF2F7',
  },
  rowInfo: {
    flex: 1,
    minWidth: 0,
  },
  rowCategory: {
    fontSize: 9,
    fontWeight: '800',
    letterSpacing: 0.8,
    color: '#58734E',
  },
  rowName: {
    fontSize: 13,
    fontWeight: '700',
    color: '#0A1931',
    marginTop: 2,
    lineHeight: 17,
  },
  rowMeta: {
    fontSize: 11,
    color: '#4A5568',
    marginTop: 2,
  },
  retailerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    marginTop: 4,
  },
  retailerText: {
    fontSize: 11,
    fontWeight: '600',
    color: '#58734E',
    flexShrink: 1,
  },
  wardrobeText: {
    fontSize: 11,
    color: '#8A97A8',
  },
  buyButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    backgroundColor: '#173A65',
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 999,
  },
  buyButtonText: {
    color: '#FFFFFF',
    fontSize: 13,
    fontWeight: '800',
  },
});

export default OutfitShopSheet;
