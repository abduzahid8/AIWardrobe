/** Isolated simulator QA entry, never a shipping entry point. No App/auth/provider imports. */
import React, { useEffect, useRef, useState } from 'react';
import {
  AppState,
  FlatList,
  Image,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { registerRootComponent } from 'expo';
import * as FileSystem from 'expo-file-system/legacy';
import * as SplashScreen from 'expo-splash-screen';
import { Image as ExpoImage } from 'expo-image';
import i18next from 'i18next';
import { initReactI18next } from 'react-i18next';
import en from '../../i18n/locales/en.json';
import { BetaFeedbackButton } from '../../components/BetaFeedbackButton';
import { authEvents } from '../../store/authEvents';
import { buyCheck } from '../../src/services/purchase/buyCheck';
import { outfitChoices } from '../../backend/supabase/functions/_shared/outfitChoices';
import { fourSlotOutfitReason } from '../../backend/supabase/functions/_shared/outfitStructure';
import type { ShopCatalogItem } from '../../src/types/shop';

const moduleStarted = performance.now();
const resultFile = `${FileSystem.documentDirectory}rc-local-native.json`;
const imageDirectory = `${FileSystem.cacheDirectory}rc-local-images/`;
let networkAttempts = 0;
// Keep this block before any test action. The entry imports no service that starts network work.
globalThis.fetch = (async () => {
  networkAttempts++;
  throw new Error('RC fixture forbids network');
}) as typeof fetch;
XMLHttpRequest.prototype.open = function () {
  networkAttempts++;
  throw new Error('RC fixture forbids XHR');
};
void i18next
  .use(initReactI18next)
  .init({
    lng: 'en',
    fallbackLng: 'en',
    resources: { en: { translation: en } },
    interpolation: { escapeValue: false },
  });
const fixturePhoto = require('./fixtures/opaque-photo/photo-1.jpg');
const apparel = [
  ['White cotton oxford shirt', 'upper_body', 'top'],
  ['Navy cotton chino trousers', 'lower_body', 'bottom'],
  ['Brown leather penny loafers', 'shoes', 'shoes'],
  ['Beige merino cardigan', 'upper_body', 'outerwear'],
] as const;
function makeWardrobe(uri: string): ShopCatalogItem[] {
  return Array.from({ length: 220 }, (_, index) => {
    const [name, garmentType, macroCategory] = apparel[index % 4];
    return {
      id: `synthetic-native-${index}`,
      name,
      garmentType,
      macroCategory,
      brand: 'Synthetic',
      price: 0,
      currency: 'USD',
      imageUrl: uri,
      isWardrobe: true,
    };
  });
}
interface Report {
  schemaVersion: 1;
  scope: string;
  platform: string;
  releaseBuild: boolean;
  moduleToFirstLayoutMs?: number;
  status: string;
  networkAttempts: number;
  checks: string[];
  appStates: string[];
  imports?: number;
  itemCount?: number;
  buyCheckMs?: number[];
  deterministicGenerationMs?: number[];
  scrollStepMs?: number[];
  imageLoads?: number;
  imageErrors?: number;
  memoryWarnings?: number;
  frameGapsMs?: number[];
  failure?: string;
}
const report: Report = {
  schemaVersion: 1,
  scope: 'isolated native harness; excludes full app startup/auth/billing/media-owner wrapper',
  platform: Platform.OS,
  releaseBuild: !__DEV__,
  status: 'idle',
  networkAttempts: 0,
  checks: [],
  appStates: [],
};
let writeQueue = Promise.resolve();
function persist() {
  report.networkAttempts = networkAttempts;
  const snapshot = JSON.stringify(report);
  writeQueue = writeQueue.then(() => FileSystem.writeAsStringAsync(resultFile, snapshot));
  return writeQueue;
}
const pause = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));
function Audit() {
  const [status, setStatus] = useState('Isolated RC fixture ready');
  const [rows, setRows] = useState<Array<{ id: string; imageUrl: string }>>([]);
  const [phase, setPhase] = useState('Wardrobe fixture');
  const list = useRef<FlatList<{ id: string; imageUrl: string }> | null>(null);
  const running = useRef(false),
    laidOut = useRef(false);
  useEffect(() => {
    void SplashScreen.hideAsync();
    const subscription = AppState.addEventListener('change', (state) => {
      report.appStates.push(state);
      void persist();
    });
    const memory = AppState.addEventListener('memoryWarning', () => {
      report.memoryWarnings = (report.memoryWarnings || 0) + 1;
      void persist();
    });
    return () => {
      subscription.remove();
      memory.remove();
    };
  }, []);
  async function stress() {
    if (running.current) return;
    running.current = true;
    report.status = 'running';
    report.checks = [];
    report.imageLoads = 0;
    report.imageErrors = 0;
    report.buyCheckMs = [];
    report.deterministicGenerationMs = [];
    report.scrollStepMs = [];
    report.frameGapsMs = [];
    setStatus('Isolated stress running');
    await persist();
    let frame = 0,
      previous = performance.now();
    const onFrame = (now: number) => {
      if (report.frameGapsMs!.length < 2000) report.frameGapsMs!.push(now - previous);
      previous = now;
      frame = requestAnimationFrame(onFrame);
    };
    frame = requestAnimationFrame(onFrame);
    try {
      await FileSystem.deleteAsync(imageDirectory, { idempotent: true });
      await FileSystem.makeDirectoryAsync(imageDirectory, { intermediates: true });
      const source = Image.resolveAssetSource(fixturePhoto).uri;
      if (!source.startsWith('file:') && !source.startsWith('/'))
        throw new Error('Photo fixture is not bundled locally');
      const uris: string[] = [];
      for (let index = 0; index < 30; index++) {
        const uri = `${imageDirectory}import-${index}.jpg`;
        await FileSystem.copyAsync({ from: source, to: uri });
        uris.push(uri);
      }
      report.imports = uris.length;
      report.checks.push('30 local synthetic image copies imported; no picker/provider used');
      const wardrobe = makeWardrobe(uris[0]);
      report.itemCount = wardrobe.length;
      setRows(
        wardrobe.map((item, index) => ({ id: item.id, imageUrl: uris[index % uris.length] }))
      );
      setPhase('220-item wardrobe fixture');
      await pause(800);
      for (let index = 0; index < 30; index++) {
        const start = performance.now();
        list.current?.scrollToOffset({ offset: index % 2 ? 0 : 32000, animated: false });
        await pause(100);
        report.scrollStepMs.push(performance.now() - start);
      }
      report.checks.push(
        '220-item native FlatList repeatedly scrolled; excludes production wardrobe screen'
      );
      const candidate = { ...wardrobe[2], id: 'synthetic-candidate', price: 100 };
      for (let index = 0; index < 25; index++) {
        const start = performance.now();
        const result = buyCheck(candidate, wardrobe, {
          style: 'classic',
          occasion: 'Work',
          budget: { amount: 5, currency: 'USD' },
        });
        report.buyCheckMs.push(performance.now() - start);
        if (result.verdict !== 'SKIP' || result.fit.confidence !== 'low')
          throw new Error('BuyCheck violated hard budget or sparse-fit evidence');
        const generatedAt = performance.now();
        const looks = outfitChoices(
          wardrobe,
          true,
          (parts) => !fourSlotOutfitReason(parts),
          () => false,
          [],
          1,
          1024
        );
        report.deterministicGenerationMs.push(performance.now() - generatedAt);
        if (looks.length !== 1 || fourSlotOutfitReason(looks[0].items))
          throw new Error('Deterministic generation returned an invalid four-slot outfit');
        await pause(0);
      }
      report.checks.push(
        '25 actual BuyCheck calls rejected hard-budget conflict and retained unknown fit'
      );
      report.checks.push(
        '25 production deterministic outfitChoices calls validated four slots; no provider generation'
      );
      for (const [label, count] of [
        ['24-reference image fixture', 24],
        ['20 synthetic VTON-history images (no provider)', 20],
      ] as const) {
        setPhase(label);
        setRows(
          Array.from({ length: count }, (_, index) => ({
            id: `${label}-${index}`,
            imageUrl: uris[index % uris.length],
          }))
        );
        await pause(500);
        for (let index = 0; index < 10; index++) {
          list.current?.scrollToOffset({ offset: index % 2 ? 0 : 4000, animated: false });
          await pause(100);
        }
        report.checks.push(label);
      }
      setRows([]);
      await pause(100);
      await ExpoImage.clearMemoryCache();
      await ExpoImage.clearDiskCache();
      await FileSystem.deleteAsync(imageDirectory, { idempotent: true });
      const info = await FileSystem.getInfoAsync(imageDirectory);
      if (info.exists || report.imageErrors! > 0 || networkAttempts > 0 || report.imageLoads! < 1)
        throw new Error('Image cleanup/decode/network guard failed');
      report.checks.push('native Expo image caches cleared and fixture directory removed');
      report.status = 'passed';
      setStatus('Isolated stress passed');
    } catch (error) {
      report.status = 'failed';
      report.failure = error instanceof Error ? error.message : 'unknown';
      setStatus('Isolated stress failed');
    } finally {
      cancelAnimationFrame(frame);
      running.current = false;
      await persist();
    }
  }
  return (
    <View
      style={styles.root}
      onLayout={() => {
        if (!laidOut.current) {
          laidOut.current = true;
          report.moduleToFirstLayoutMs = performance.now() - moduleStarted;
          void persist();
        }
      }}
    >
      <Text style={styles.title}>Local RC validation only</Text>
      <Text accessibilityLabel="rcHarnessStatus">{status}</Text>
      <Text>
        No account, backend, billing, analytics sink or paid provider. Fixture results are not
        full-app QA.
      </Text>
      <Pressable
        accessibilityRole="button"
        accessibilityLabel="Run isolated stress"
        onPress={() => {
          void stress();
        }}
        style={styles.button}
      >
        <Text>Run isolated stress</Text>
      </Pressable>
      <Pressable
        accessibilityRole="button"
        accessibilityLabel="Change synthetic owner"
        onPress={() => authEvents.emitLogin('synthetic-native-b')}
        style={styles.button}
      >
        <Text>Change synthetic owner</Text>
      </Pressable>
      <BetaFeedbackButton
        feature="buycheck"
        diagnostics={{
          requestId: 'synthetic-native-request',
          ruleVersion: 'purchase-local-v1',
          confidence: 'low',
          candidateCount: 220,
          fallback: 'none',
        }}
      />
      <Text>{phase}</Text>
      <FlatList
        ref={list}
        data={rows}
        keyExtractor={(item) => item.id}
        getItemLayout={(_, index) => ({ length: 160, offset: 160 * index, index })}
        renderItem={({ item }) => (
          <View style={styles.row}>
            <ExpoImage
              source={{ uri: item.imageUrl }}
              style={styles.image}
              contentFit="contain"
              recyclingKey={item.id}
              onLoad={() => {
                report.imageLoads!++;
              }}
              onError={() => {
                report.imageErrors!++;
              }}
            />
            <Text>{item.id}</Text>
          </View>
        )}
      />
    </View>
  );
}
const styles = StyleSheet.create({
  root: { flex: 1, padding: 20, paddingTop: 64, backgroundColor: '#fff' },
  title: { fontSize: 22, fontWeight: '700' },
  button: { minHeight: 44, padding: 12, backgroundColor: '#e5eaff', marginVertical: 6 },
  row: { height: 160, flexDirection: 'row', alignItems: 'center' },
  image: { width: 130, height: 150 },
});
registerRootComponent(Audit);
