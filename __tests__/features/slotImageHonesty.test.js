/**
 * Gate G image honesty: a slot tile without a loadable photo must render an
 * explicit labeled 'photo unavailable' placeholder — never a silent grey box
 * presented as the product photo. Exactly ONE bounded automatic retry with a
 * fresh cache-busting query param may follow a load error.
 *
 * The real SlotCell/OutfitSlotGrid sources are extracted from the screen AST
 * (same approach as outfitPriceDisplay.test.js) and executed with the real
 * retry hook so the state machine — not a replica — is what passes.
 */
import React from 'react';
import fs from 'fs';
import ts from 'typescript';
import { act, render, screen } from '@testing-library/react-native';
import { View, TouchableOpacity, Text } from 'react-native';
import { formatMoney, resolveShopQuote } from '../../src/utils/outfitBudget';
import { canonicalizeMacroCategory } from '../../src/utils/categoryMapper';
import { getShopDestination } from '../../src/utils/shopDestination';
import { offerCaveatLabel } from '../../src/services/stylingCopy';
import {
  ImageUnavailablePlaceholder,
  useProductImageRetry,
  usableProductImageUri,
  PRODUCT_IMAGE_RETRY_DELAY_MS,
} from '../../src/components/ProductImageFallback';

jest.mock('@expo/vector-icons', () => ({ Ionicons: () => null }));

const extract = (file, pick) => {
  const source = fs.readFileSync(file, 'utf8');
  const ast = ts.createSourceFile('subject', source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  let code;
  const visit = (node) => {
    if (ts.isFunctionDeclaration(node) && node.name?.text === pick) {
      code = `const value = ${node.getText(ast)};`;
    }
    if (ts.isVariableStatement(node)) {
      for (const d of node.declarationList.declarations) {
        if (d.name.getText(ast) === pick) code = `const value = ${d.initializer.getText(ast)};`;
      }
    }
    ts.forEachChild(node, visit);
  };
  visit(ast);
  if (!code) throw new Error(`Missing production component: ${pick}`);
  return ts.transpileModule(code, { compilerOptions: { target: ts.ScriptTarget.ES2020, jsx: ts.JsxEmit.React } }).outputText;
};

// expo-image mock: records every render's props so tests can read the current
// source and deliver loader errors to the real hook.
let imageLog = [];
const MockExpoImage = (props) => { imageLog.push(props); return null; };
const currentImage = () => imageLog[imageLog.length - 1];
const MockScaledText = ({ children }) => <Text>{children}</Text>;
const MockIonicons = () => null;

const build = (code, scope) => Function(...Object.keys(scope), `${code};return value;`)(...Object.values(scope));

const screenSource = require.resolve('../../screens/AIOutfitmaker.tsx');
const getSlotImageSource = build(extract(screenSource, 'getSlotImageSource'), {});
const slotCell = build(extract(screenSource, 'SlotCell'), {
  React, View, TouchableOpacity,
  ExpoImage: MockExpoImage,
  ScaledText: MockScaledText,
  Ionicons: MockIonicons,
  slotStyles: {},
  formatMoney,
  ImageUnavailablePlaceholder,
  useProductImageRetry,
});
const outfitSlotGrid = build(extract(screenSource, 'OutfitSlotGrid'), {
  React, View,
  SlotCell: slotCell,
  useDesignTokens: () => ({ textSecondary: '#000', textPrimary: '#000' }),
  useTranslation: () => ({ t: (_key, fallback) => fallback }),
  useCallback: (fn) => fn,
  canonicalizeMacroCategory,
  getSlotImageSource,
  resolveShopQuote,
  getShopDestination,
  offerCaveatLabel,
  slotStyles: {},
});

const sheetSource = require.resolve('../../components/OutfitShopSheet.tsx');
const sheetThumb = build(extract(sheetSource, 'SheetThumb'), {
  React,
  ExpoImage: MockExpoImage,
  ImageUnavailablePlaceholder,
  useProductImageRetry,
  styles: { thumb: {} },
});

const lookSource = require.resolve('../../components/CompleteYourLook.tsx');
const COLORS = {
  background: require('../../constants/AppColors').default.background,
  surface: require('../../constants/AppColors').default.surface,
  primary: require('../../constants/AppColors').default.primary,
  accent: require('../../constants/AppColors').default.accent,
  text: require('../../constants/AppColors').default.text,
  textSecondary: require('../../constants/AppColors').default.textSecondary,
  border: require('../../constants/AppColors').default.border,
  sale: '#FF5252',
};
const productCard = build(extract(lookSource, 'ProductCard'), {
  React, View, TouchableOpacity,
  Image: MockExpoImage,
  ScaledText: MockScaledText,
  Ionicons: MockIonicons,
  styles: {},
  COLORS,
  ImageUnavailablePlaceholder,
  useProductImageRetry,
  useTranslation: () => ({ t: (key) => key }),
});

const cellProps = (imageUri) => ({
  slotKey: 'second-top-1',
  label: 'TOP',
  name: 'Wool blazer',
  imageSource: { uri: imageUri ?? '' },
  imageUri,
  itemId: 'sku-1',
  buyUrl: '',
  showBuyButton: false,
  buyLabel: 'Buy',
  photoUnavailableLabel: 'Photo unavailable',
  textSecondary: '#000',
  textPrimary: '#000',
  onBuy: () => {},
});

beforeEach(() => {
  imageLog = [];
  jest.useFakeTimers();
});
afterEach(() => jest.useRealTimers());

describe('SlotCell image honesty', () => {
  it('renders the labeled placeholder, not an image tile, for a missing or invalid URL', () => {
    for (const missing of [undefined, '', 'not-a-url', '/local/relative.jpg']) {
      imageLog = [];
      const view = render(React.createElement(slotCell, cellProps(missing)));
      expect(view.getByTestId('slot-photo-unavailable-second-top-1')).toBeTruthy();
      expect(view.getByText('Photo unavailable')).toBeTruthy();
      // No image element at all: nothing can present itself as the photo.
      expect(imageLog).toHaveLength(0);
      view.unmount();
    }
  });

  it('renders the real photo for a usable URL and never shows the placeholder', () => {
    const view = render(React.createElement(slotCell, cellProps('https://cdn.example/coat.jpg')));
    expect(currentImage().source).toEqual({ uri: 'https://cdn.example/coat.jpg' });
    expect(view.queryByTestId('slot-photo-unavailable-second-top-1')).toBeNull();
  });

  it('retries a load error exactly once with a fresh cache-busting param, then shows the placeholder', () => {
    const view = render(React.createElement(slotCell, cellProps('https://cdn.example/coat.jpg?token=abc')));
    act(() => { currentImage().onError(); });
    // Inside the retry window nothing changed yet — the original source stands.
    expect(currentImage().source).toEqual({ uri: 'https://cdn.example/coat.jpg?token=abc' });
    act(() => { jest.advanceTimersByTime(PRODUCT_IMAGE_RETRY_DELAY_MS); });
    // Exactly one retry, on a fresh cache-busting URL (existing query kept).
    expect(currentImage().source.uri).toMatch(/^https:\/\/cdn\.example\/coat\.jpg\?token=abc&aiwr=\d+$/);
    // A second load error lands on the explicit unavailable placeholder.
    act(() => { currentImage().onError(); });
    expect(view.getByTestId('slot-photo-unavailable-second-top-1')).toBeTruthy();
    expect(view.getByText('Photo unavailable')).toBeTruthy();
    // A third error is ignored: no further image elements ever render.
    const stale = currentImage();
    act(() => { stale.onError?.(); });
    act(() => { jest.advanceTimersByTime(5000); });
    expect(currentImage()).toBe(stale);
  });

  it('never cache-busts local or private wardrobe photos — one error is the placeholder', () => {
    const view = render(React.createElement(slotCell, cellProps('file:///local/wardrobe.jpg')));
    act(() => { currentImage().onError(); });
    expect(view.getByTestId('slot-photo-unavailable-second-top-1')).toBeTruthy();
    act(() => { jest.advanceTimersByTime(PRODUCT_IMAGE_RETRY_DELAY_MS * 4); });
    expect(imageLog).toHaveLength(1); // no re-attempt rendered a second image
    expect(usableProductImageUri('https://x/y')).toBe('https://x/y');
  });

  it('resets the retry cycle when the slot content changes', () => {
    const view = render(React.createElement(slotCell, cellProps('https://cdn.example/coat.jpg')));
    act(() => { currentImage().onError(); });
    view.rerender(React.createElement(slotCell, cellProps('https://cdn.example/other.jpg')));
    expect(currentImage().source).toEqual({ uri: 'https://cdn.example/other.jpg' });
    expect(view.queryByTestId('slot-photo-unavailable-second-top-1')).toBeNull();
    // The failed first photo must not consume the new photo's single retry.
    act(() => { currentImage().onError(); });
    act(() => { jest.advanceTimersByTime(PRODUCT_IMAGE_RETRY_DELAY_MS); });
    expect(String(currentImage().source.uri)).toContain('cdn.example/other.jpg?aiwr=');
  });
});

describe('OutfitSlotGrid image honesty', () => {
  const item = (extra = {}) => ({ id: 'shop_coat', name: 'Coat', type: 'outerwear', isShopItem: true, ...extra });

  it('renders a usable photo tile for a real product image', () => {
    const view = render(React.createElement(outfitSlotGrid, { items: [item({ imageUrl: 'https://cdn.example/coat.jpg' })] }));
    expect(currentImage().source).toEqual({ uri: 'https://cdn.example/coat.jpg' });
    expect(screen.queryByText('Photo unavailable')).toBeNull();
    expect(view.toJSON()).toBeTruthy();
  });

  it('renders the localized unavailable label instead of a bare grey tile when the URL is missing', () => {
    for (const broken of [{}, { image: '' }, { image: 'garbage' }, { imageUrl: 'not-a-url' }]) {
      imageLog = [];
      const view = render(React.createElement(outfitSlotGrid, { items: [item(broken)] }));
      expect(view.getByText('Photo unavailable')).toBeTruthy();
      expect(imageLog).toHaveLength(0);
      view.unmount();
    }
  });
});

describe('OutfitShopSheet thumbnail image honesty', () => {
  it('shows the labeled placeholder for a missing or invalid image string', () => {
    for (const raw of ['', 'not-a-url']) {
      imageLog = [];
      const view = render(React.createElement(sheetThumb, { rawImage: raw, unavailableLabel: 'Фото недоступно' }));
      expect(view.getByText('Фото недоступно')).toBeTruthy();
      expect(imageLog).toHaveLength(0);
      view.unmount();
    }
  });

  it('retries once with a cache-busted URL, then falls back to the placeholder', () => {
    const view = render(React.createElement(sheetThumb, { rawImage: 'https://cdn.example/shoe.jpg', unavailableLabel: 'Photo unavailable' }));
    act(() => { currentImage().onError(); });
    act(() => { jest.advanceTimersByTime(PRODUCT_IMAGE_RETRY_DELAY_MS); });
    expect(String(currentImage().source.uri)).toMatch(/^https:\/\/cdn\.example\/shoe\.jpg\?aiwr=\d+$/);
    act(() => { currentImage().onError(); });
    expect(view.getByText('Photo unavailable')).toBeTruthy();
    expect(imageLog).toHaveLength(2);
  });
});

describe('CompleteYourLook product card image honesty', () => {
  const product = (imageUrl, extra = {}) => ({
    id: 'p1', name: 'Belt', brand: 'B', price: 20, imageUrl, ...extra,
  });

  it('renders the photo when loadable and keeps onError wired to the retry hook', () => {
    const view = render(React.createElement(productCard, { product: product('https://cdn.example/belt.jpg'), onPress: () => {}, onWishlist: () => {} }));
    expect(currentImage().source).toEqual({ uri: 'https://cdn.example/belt.jpg' });
    expect(view.queryByText('common.photoUnavailable')).toBeNull();
  });

  it('replaces a failed photo with the labeled placeholder after the single retry', () => {
    const view = render(React.createElement(productCard, { product: product('https://cdn.example/belt.jpg'), onPress: () => {}, onWishlist: () => {} }));
    act(() => { currentImage().onError(); });
    act(() => { jest.advanceTimersByTime(PRODUCT_IMAGE_RETRY_DELAY_MS); });
    act(() => { currentImage().onError(); });
    expect(view.getByText('common.photoUnavailable')).toBeTruthy();
  });

  it('shows the labeled placeholder immediately for a missing URL', () => {
    const view = render(React.createElement(productCard, { product: product(''), onPress: () => {}, onWishlist: () => {} }));
    expect(view.getByText('common.photoUnavailable')).toBeTruthy();
    expect(imageLog).toHaveLength(0);
  });
});
