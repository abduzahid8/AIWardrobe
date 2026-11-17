import { BetaFeedbackButton } from '../components/BetaFeedbackButton';
import { CatalogVariantPicker } from '../components/CatalogVariantPicker';
import { hasProductOnlyCatalogProvenance } from '../src/utils/productOnlyCatalog';
import { purchasePriceReads } from '../src/services/purchase/marketData';
import { newPurchaseRecordId } from '../src/services/purchase/localId';
import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  View,
  ScrollView,
  TouchableOpacity,
  TextInput,
  Switch,
  StyleSheet,
  ActivityIndicator,
  Linking,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRoute, type RouteProp } from '@react-navigation/native';
import { useTranslation } from 'react-i18next';
import { ScaledText } from '../components/ui/ScaledText';
import { useAppNavigation } from '../hooks/useAppNavigation';
import { withRequestDeadline } from '../src/lib/requestDeadline';
import type { RootStackParamList } from '../navigation/types';
import useWardrobeStore from '../store/wardrobeStore';
import { useBodyProfileStore } from '../store/bodyProfileStore';
import { useStylePreferenceStore } from '../store/stylePreferenceStore';
import useShopCatalogStore from '../store/shopCatalogStore';
import { usePurchaseIntelligenceStore } from '../store/purchaseIntelligenceStore';
import { authEvents } from '../store/authEvents';
import { catalogRepository } from '../src/services/shoppingService';
import { deriveFitProfile } from '../src/services/purchase/fitEvidence';
import { buyCheck } from '../src/services/purchase/buyCheck';
import {
  findBetterV2,
  maximumWardrobeGain,
  sameLookLessMoney,
  upgradeThisLook,
  detectMissingPieces,
} from '../src/services/purchase/wardrobeTools';
import { resolvePurchaseInput } from '../src/services/purchase/merchantProducts';
import { parseIncomingProduct } from '../src/services/purchase/merchantProducts';
import { parseShoppingIntent } from '../src/services/purchase/shareReceiving';
import {
  knownCharge,
  truePrice,
  evaluatePriceWatch,
} from '../src/services/purchase/priceIntelligence';
import { formatMoney } from '../src/utils/outfitBudget';
import { getShopDestination } from '../src/utils/shopDestination';
import type { ShopCatalogItem } from '../src/types/shop';
import type {
  BuyCheckResult,
  FindBetterMode,
  MerchantOffer,
  PurchaseContext,
} from '../src/types/purchase';
type Intent = 'check' | 'find' | 'complete' | 'budget' | 'save' | 'upgrade';
export default function PurchaseIntelligenceScreen() {
  const { t } = useTranslation(),
    navigation = useAppNavigation(),
    route = useRoute<RouteProp<RootStackParamList, 'PurchaseIntelligence'>>();
  const privateStore = usePurchaseIntelligenceStore(),
    closet = useWardrobeStore((s) => s.items),
    catalog = useShopCatalogStore((s) => s.items),
    bodyStore = useBodyProfileStore();
  const [intent, setIntent] = useState<Intent>(route.params?.intent || 'check'),
    [input, setInput] = useState(route.params?.input || ''),
    [candidate, setCandidate] = useState<ShopCatalogItem | undefined>(route.params?.candidate),
    [variants, setVariants] = useState<ShopCatalogItem[]>([]),
    [results, setResults] = useState<ShopCatalogItem[]>([]);
  const [size, setSize] = useState(''),
    [budget, setBudget] = useState(''),
    [currency, setCurrency] = useState('USD'),
    [occasion, setOccasion] = useState('Work'),
    [style, setStyle] = useState('classic');
  const [lockedLookIds, setLockedLookIds] = useState<string[]>([]);
  const [allowMultiples, setAllowMultiples] = useState(false);
  const [temperature, setTemperature] = useState('');
  const [shipping, setShipping] = useState(''),
    [tax, setTax] = useState(''),
    [fees, setFees] = useState(''),
    [threshold, setThreshold] = useState('');
  const [analysis, setAnalysis] = useState<BuyCheckResult>(),
    [tool, setTool] = useState<{ lines: string[]; items?: ShopCatalogItem[] }>(),
    [busy, setBusy] = useState(false),
    [message, setMessage] = useState(''),
    [offers, setOffers] = useState<MerchantOffer[]>([]),
    [lookSnapshots, setLookSnapshots] = useState<Record<string, ShopCatalogItem>>({}),
    [manualTitle, setManualTitle] = useState(''),
    [manualBrand, setManualBrand] = useState(''),
    [manualPrice, setManualPrice] = useState(''),
    [manualType, setManualType] = useState<ShopCatalogItem['garmentType']>('upper_body'),
    [lookIds, setLookIds] = useState<string[]>(route.params?.outfitIds || []);
  const preferences = useStylePreferenceStore((state) => state.preferences);
  const request = useRef(0),
    alive = useRef(true),
    previousOwner = useRef(privateStore.ownerId),
    analysisOwner = useRef<string | null>(null),
    toolOwner = useRef<string | null>(null);
  const owned = useMemo<ShopCatalogItem[]>(
    () =>
      closet.map((item) => ({
        id: item.id,
        name: item.name || item.subCategory,
        brand: item.brand || '',
        price: 0,
        imageUrl: item.imageUrl,
        garmentType:
          item.category === 'shoes'
            ? 'shoes'
            : item.category === 'bottom'
              ? 'lower_body'
              : item.category === 'dress'
                ? 'dresses'
                : item.category === 'accessory'
                  ? 'accessory'
                  : 'upper_body',
        category: item.category,
        colorName: item.primaryColor,
        description: [item.material, item.pattern, item.subCategory].join(' '),
        physicalProfiles: Object.values(bodyStore.garmentProfiles).filter(
          (profile) => profile.garmentId === item.id
        ),
      })),
    [closet, bodyStore.garmentProfiles]
  );
  const fitProfile = useMemo(
    () =>
      deriveFitProfile(privateStore.evidence.filter((e) => e.userId === authEvents.getUserId())),
    [privateStore.evidence]
  );
  const body = bodyStore.activeProfileId
    ? bodyStore.profiles[bodyStore.activeProfileId]
    : undefined;
  const context: PurchaseContext & { lockedIds: string[] } = {
    lockedIds: lockedLookIds,
    allowMultiples,
    style,
    occasion,
    avoidedColors: preferences.avoidColors,
    avoidedPatterns: preferences.dislikedPatterns,
    avoidedBrands: preferences.avoidBrands,
    prompt: !temperature.trim() ? parseIncomingProduct(input).title : undefined,
    weather:
      temperature.trim() !== '' && Number.isFinite(Number(temperature))
        ? { temp: Number(temperature) }
        : undefined,
    contexts: [...new Set([occasion, ...preferences.primaryOccasions])]
      .slice(0, 4)
      .map((value) => ({
        id: value,
        occasion: value,
        style,
        weather:
          temperature.trim() !== '' && Number.isFinite(Number(temperature))
            ? { temp: Number(temperature) }
            : undefined,
      })),
    selectedSize: size || undefined,
    fitProfile,
    bodyProfile: body?.userId === authEvents.getUserId() ? body : undefined,
    ...(budget.trim() && Number.isFinite(Number(budget)) && Number(budget) >= 0
      ? { budget: { amount: Number(budget), currency: currency.toUpperCase() } }
      : {}),
  };
  const formToken = useMemo(
    () => ({}),
    [
      candidate,
      size,
      budget,
      currency,
      occasion,
      style,
      temperature,
      shipping,
      tax,
      fees,
      allowMultiples,
      privateStore.evidence,
      closet,
      body,
      preferences.avoidColors.join('|'),
      preferences.avoidBrands.join('|'),
      preferences.dislikedPatterns.join('|'),
    ]
  );
  const formTokenRef = useRef(formToken),
    analysisToken = useRef<object | null>(null),
    toolToken = useRef<object | null>(null);
  formTokenRef.current = formToken;
  useEffect(() => {
    if (preferences.stylePersonality) setStyle(preferences.stylePersonality);
    if (preferences.primaryOccasions[0]) setOccasion(preferences.primaryOccasions[0]);
  }, []);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
      request.current++;
    };
  }, []);
  useEffect(() => {
    void privateStore
      .hydrate()
      .catch(() =>
        setMessage(t('purchase.loadFailed', 'Could not load private shopping preferences'))
      );
  }, [privateStore.hydrate, t]);
  useEffect(() => {
    if (previousOwner.current && previousOwner.current !== privateStore.ownerId) {
      request.current++;
      setAnalysis(undefined);
      setTool(undefined);
      setBusy(false);
      setInput('');
      setCandidate(undefined);
      setVariants([]);
      setResults([]);
      setOffers([]);
      setLookIds([]);
      setLookSnapshots({});
      setSize('');
      setBudget('');
      setShipping('');
      setTax('');
      setFees('');
      setThreshold('');
      setTemperature('');
      setAllowMultiples(false);
      setLockedLookIds([]);
      setManualTitle('');
      setManualBrand('');
      setManualPrice('');
      setStyle('classic');
      setOccasion('Work');
      setMessage('');
    }
    previousOwner.current = privateStore.ownerId;
  }, [privateStore.ownerId]);
  useEffect(() => {
    const id = route.params?.candidateId;
    if (!id || candidate?.id === id) return;
    let active = true;
    void catalogRepository.getVariant(id).then((item) => {
      if (active) {
        if (item) setCandidate(item);
        else
          setMessage(
            t(
              'purchase.offlineProduct',
              'Product unavailable offline. Paste its title or select a loaded product.'
            )
          );
      }
    });
    return () => {
      active = false;
    };
  }, [route.params?.candidateId]);

  useEffect(() => {
    if (route.params?.input) {
      setInput(route.params.input);
      load(route.params.input);
    }
  }, [route.params?.input, route.params?.inputRevision]);
  const run = (work: (current: () => boolean) => Promise<void> | void) => {
    const version = ++request.current,
      owner = authEvents.getUserId();
    setBusy(true);
    setMessage('');
    setTimeout(() => {
      if (version !== request.current || owner !== authEvents.getUserId()) return;
      Promise.resolve()
        .then(() =>
          work(
            () => alive.current && version === request.current && owner === authEvents.getUserId()
          )
        )
        .catch(() => {
          if (version === request.current)
            setMessage(
              t('purchase.failed', 'Could not complete this check. You can continue shopping.')
            );
        })
        .finally(() => {
          if (version === request.current) setBusy(false);
        });
    }, 0);
  };
  const contextValid = () => {
    if (
      temperature.trim() &&
      (!Number.isFinite(Number(temperature)) ||
        Number(temperature) < -60 ||
        Number(temperature) > 60)
    ) {
      setMessage('Enter an intended temperature between -60 and 60 °C');
      return false;
    }
    if (budget && (!Number.isFinite(Number(budget)) || Number(budget) < 0)) {
      setMessage(t('purchase.invalidBudget', 'Enter a valid non-negative budget'));
      return false;
    }
    return true;
  };
  const load = (sharedInput?: string) =>
    run(async (current) => {
      const value = sharedInput || input;
      setAnalysis(undefined);
      setTool(undefined);
      setCandidate(undefined);
      setOffers([]);
      const revision = authEvents.getRevision();
      const parsed = parseShoppingIntent(value);
      if (parsed.budget) {
        setBudget(String(parsed.budget.amount));
        setCurrency(parsed.budget.currency);
      }
      if (!route.params?.intent) setIntent(parsed.mode);
      const resolved = await resolvePurchaseInput(value, {
        catalog: [...catalog, ...privateStore.bookmarks],
        lookupUrl: async (url) => {
          // A bounded catalog read; no merchant scraping or source-code upload.
          const { supabase } = await import('../lib/supabase');
          const { data, error } = await withRequestDeadline((signal) =>
            supabase
              .from('shop_catalog')
              .select(
                'id,name,brand,price,currency,image_url,garment_type,category,description,source,source_url'
              )
              .eq('is_active', true)
              .eq('source_url', url)
              .limit(5)
              .abortSignal(signal)
          );
          if (error) throw error;
          const { hasProductOnlyCatalogProvenance } =
            await import('../src/utils/productOnlyCatalog');
          const { catalogRowToItem } = await import('../src/services/catalogRepository');
          return (data || []).filter(hasProductOnlyCatalogProvenance).map(catalogRowToItem);
        },
      });
      if (!current() || revision !== authEvents.getRevision()) return;
      setVariants(resolved.variants);
      setOffers(resolved.offers);
      setCandidate(resolved.candidate);
      setSize('');
      setMessage(resolved.message || resolved.evidence.join('. '));
      if (resolved.title || intent === 'find' || parsed.mode === 'find') {
        const page = await catalogRepository.search({
          query: resolved.title || parsed.query,
          limit: 24,
          ...(parsed.budget
            ? { currency: parsed.budget.currency, priceMax: parsed.budget.amount }
            : {}),
        });
        if (current() && revision === authEvents.getRevision()) setResults(page.items);
      }
    });
  const itemOffers = (observedOffers: MerchantOffer[] = []): MerchantOffer[] => {
    if (!candidate) return [];
    const now = new Date().toISOString();
    const base = [
      ...new Map(
        [...observedOffers, ...offers]
          .filter((offer) => offer.variantId === candidate.id)
          .map((offer) => [offer.id, offer])
      ).values(),
    ];
    const exact = base.findIndex(
      (offer) =>
        parseIncomingProduct(offer.url).url === parseIncomingProduct(candidate.sourceUrl || '').url
    );
    if (exact > 0) base.unshift(...base.splice(exact, 1));
    const offer: MerchantOffer = base[0] || {
      id: `catalog-offer:${candidate.id}`,
      quoteKind: 'estimate',
      variantId: candidate.id,
      merchant: candidate.brand || 'Catalog merchant',
      url: candidate.sourceUrl || '',
      currency: candidate.currency || '',
      currentPrice: candidate.price,
      lastSeenAt: now,
      merchantConfidence: 'low',
    };
    const charge = (text: string) =>
      text.trim() !== '' && Number.isFinite(Number(text)) && Number(text) >= 0
        ? {
            ...knownCharge(Number(text), offer.currency, 'user-entered checkout amount', now),
            confidence: 'medium' as const,
          }
        : undefined;
    return [
      {
        ...offer,
        shipping: charge(shipping) || offer.shipping,
        tax: charge(tax) || offer.tax,
        fees: charge(fees) || offer.fees,
      },
      ...base.slice(1),
    ];
  };
  const check = () =>
    run(async (current) => {
      if (!candidate || !contextValid()) return;
      const startedWith = formTokenRef.current;
      const observed = await purchasePriceReads.read(candidate.id);
      if (!current() || startedWith !== formTokenRef.current) return;
      const market = itemOffers(observed.offers);
      analysisOwner.current = authEvents.getUserId();
      analysisToken.current = formTokenRef.current;
      setAnalysis(
        buyCheck(candidate, owned, context, {
          offers: market,
          history: observed.history,
        })
      );
    });
  const explore = (mode?: FindBetterMode) =>
    run(async (current) => {
      if (!contextValid()) return;
      const startedWith = formTokenRef.current;
      const revision = authEvents.getRevision();
      const page = await catalogRepository.search({
        query: intent === 'find' ? parseShoppingIntent(input).query : '',
        limit: 48,
        ...(context.budget
          ? { currency: context.budget.currency, priceMax: context.budget.amount }
          : {}),
      });
      if (
        !current() ||
        revision !== authEvents.getRevision() ||
        startedWith !== formTokenRef.current
      )
        return;
      toolToken.current = startedWith;
      toolOwner.current = authEvents.getUserId();
      setResults(page.items);
      if (mode && candidate) {
        const alternatives = findBetterV2(candidate, page.items, owned, mode, context);
        setTool({
          lines: alternatives.length
            ? alternatives.flatMap((alt) => [
                `${alt.item.name}: ${alt.why.join('. ')}`,
                alt.tradeoffs.join('. '),
              ])
            : [
                t(
                  'purchase.noAlternatives',
                  'No valid alternatives with enough evidence were found'
                ),
              ],
          items: alternatives.map((alt) => alt.item),
        });
        return;
      }
      if (intent === 'complete') {
        const missing = detectMissingPieces(owned, page.items, context);
        setTool({
          lines: [
            ...missing.evidence,
            ...missing.suggestions.map(
              (s) => `${s.item.name}: ${s.unlocks.total} validated sampled looks`
            ),
          ],
          items: missing.suggestions.map((s) => s.item),
        });
      } else if (intent === 'budget') {
        const gain = maximumWardrobeGain(page.items, owned, context);
        setTool({
          lines: [
            ...gain.evidence,
            ...gain.candidates.map(
              (r) =>
                `${r.candidate.name}: ${r.wardrobe.usefulOutfitUnlocks.total} validated looks; fit ${r.fit.confidence}`
            ),
          ],
          items: gain.candidates.map((r) => r.candidate),
        });
      } else if (intent === 'save' || intent === 'upgrade') {
        const rows = new Map(
          [
            ...owned,
            ...catalog,
            ...page.items,
            ...privateStore.bookmarks,
            ...Object.values(lookSnapshots),
            ...(candidate ? [candidate] : []),
          ].map((item) => [item.id, item])
        );
        const look = lookIds
          .map((id) => rows.get(id))
          .filter((item): item is ShopCatalogItem => !!item);
        if (look.length !== 4) {
          setMessage(t('purchase.selectLook', 'Choose four exact look pieces first'));
          return;
        }
        if (intent === 'save') {
          const result = sameLookLessMoney(
            look,
            page.items,
            owned,
            { ...context, selectedSizes: candidate && size ? { [candidate.id]: size } : {} },
            context.budget ? { budget: context.budget.amount } : undefined
          );
          setTool({
            lines: [
              ...result.evidence,
              `${formatMoney(result.total, result.currency)}; item-price savings ${formatMoney(result.savings, result.currency)}`,
            ],
            items: result.items,
          });
        } else {
          const upgraded = upgradeThisLook(look, page.items, owned, context);
          setTool({
            lines: upgraded.length
              ? upgraded.flatMap((r) => [`${r.from.name} → ${r.to.name}`, ...r.evidence])
              : [t('purchase.noUpgrade', 'No supported single-piece improvement found')],
            items: upgraded.map((r) => r.to),
          });
        }
      }
    });
  const button = (label: string, action: () => void, key = label, blockWhileBusy = true) => (
    <TouchableOpacity
      key={key}
      accessibilityRole="button"
      onPress={() => action()}
      disabled={blockWhileBusy && busy}
      style={styles.button}
    >
      <ScaledText>{label}</ScaledText>
    </TouchableOpacity>
  );
  if (previousOwner.current && previousOwner.current !== privateStore.ownerId)
    return (
      <SafeAreaView style={styles.container}>
        <ActivityIndicator accessibilityRole="progressbar" />
      </SafeAreaView>
    );
  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
        {button(
          t('common.back', 'Back'),
          () => {
            request.current++;
            navigation.goBack();
          },
          'back',
          false
        )}
        <ScaledText style={styles.title}>
          {t('purchase.hubTitle', 'Shopping & fit intelligence')}
        </ScaledText>
        <View style={styles.row}>
          {(
            [
              ['complete', 'Complete my wardrobe'],
              ['find', 'Find something'],
              ['check', 'Check before I buy'],
              ['upgrade', 'Improve an outfit'],
              ['save', 'Save money'],
              ['budget', 'Spend my budget wisely'],
            ] as const
          ).map(([value, label]) =>
            button(
              t(`purchase.intent.${value}`, label),
              () => {
                setIntent(value);
                setTool(undefined);
              },
              value
            )
          )}
        </View>
        <TextInput
          editable={!busy}
          accessibilityLabel={t('purchase.input', 'Product URL, title or shopping intent')}
          placeholder={t('purchase.input', 'Product URL, title or shopping intent')}
          value={input}
          onChangeText={setInput}
          maxLength={4096}
          style={styles.input}
        />
        {button(t('purchase.resolve', 'Resolve / search'), load)}
        <View style={styles.row}>
          <TextInput
            editable={!busy}
            accessibilityLabel="Budget"
            placeholder="Budget (optional)"
            value={budget}
            onChangeText={setBudget}
            keyboardType="decimal-pad"
            style={styles.input}
          />
          <TextInput
            editable={!busy}
            accessibilityLabel="Currency"
            value={currency}
            onChangeText={setCurrency}
            autoCapitalize="characters"
            maxLength={3}
            style={styles.input}
          />
        </View>
        <View style={styles.row}>
          <TextInput
            editable={!busy}
            accessibilityLabel="Occasion"
            value={occasion}
            onChangeText={setOccasion}
            style={styles.input}
          />
          <TextInput
            editable={!busy}
            accessibilityLabel="Style"
            value={style}
            onChangeText={setStyle}
            style={styles.input}
          />
        </View>
        <TextInput
          editable={!busy}
          accessibilityLabel="Intended temperature Celsius"
          placeholder="Intended temperature °C (optional)"
          value={temperature}
          onChangeText={setTemperature}
          keyboardType="numbers-and-punctuation"
          style={styles.input}
        />
        {message !== '' && <ScaledText accessibilityRole="alert">{message}</ScaledText>}
        {busy && <ActivityIndicator accessibilityRole="progressbar" />}
        {!candidate && variants.length > 1 && (
          <View style={styles.card}>
            <ScaledText>
              {t('purchase.chooseVariant', 'Choose the exact variant before checking')}
            </ScaledText>
            <CatalogVariantPicker
              selectedId=""
              variants={variants.map((variant) =>
                hasProductOnlyCatalogProvenance(variant) ? variant : { ...variant, imageUrl: '' }
              )}
              onSelect={(id) => {
                const variant = variants.find((item) => item.id === id);
                if (variant) {
                  setCandidate(variant);
                  setSize('');
                  setAnalysis(undefined);
                }
              }}
            />
          </View>
        )}
        {intent === 'check' && (
          <View style={styles.card}>
            <ScaledText>
              {t('purchase.manual', 'Product details when URL metadata is unavailable')}
            </ScaledText>
            <TextInput
              editable={!busy}
              accessibilityLabel="Product title"
              placeholder="Product title"
              value={manualTitle}
              onChangeText={setManualTitle}
              maxLength={200}
              style={styles.input}
            />
            <TextInput
              editable={!busy}
              accessibilityLabel="Product brand"
              placeholder="Brand (optional)"
              value={manualBrand}
              onChangeText={setManualBrand}
              maxLength={80}
              style={styles.input}
            />
            <TextInput
              editable={!busy}
              accessibilityLabel="Native item price"
              placeholder="Native item price (optional)"
              value={manualPrice}
              onChangeText={setManualPrice}
              keyboardType="decimal-pad"
              style={styles.input}
            />
            <View style={styles.row}>
              {(['upper_body', 'lower_body', 'shoes', 'dresses', 'accessory'] as const).map(
                (type) => button(type.replace('_', ' '), () => setManualType(type), type)
              )}
            </View>
            {button(t('purchase.useDetails', 'Use these product details'), () => {
              if (
                !manualTitle.trim() ||
                (manualPrice && (!Number.isFinite(Number(manualPrice)) || Number(manualPrice) <= 0))
              ) {
                setMessage('Enter a title and valid native price when known');
                return;
              }
              const parsed = parseIncomingProduct(input);
              setCandidate({
                id: newPurchaseRecordId('manual'),
                name: manualTitle.trim(),
                brand: manualBrand.trim(),
                price: manualPrice ? Number(manualPrice) : 0,
                currency: manualPrice ? currency.toUpperCase() : undefined,
                garmentType: manualType,
                imageUrl: '',
                source: 'manual-user-product',
                sourceUrl: parsed.url,
              });
              setVariants([]);
              setOffers([]);
              setAnalysis(undefined);
            })}
          </View>
        )}
        {candidate && (
          <View style={styles.card}>
            <ScaledText style={styles.subtitle}>
              {candidate.brand} · {candidate.name}
            </ScaledText>
            <ScaledText>
              {candidate.currency
                ? formatMoney(candidate.price, candidate.currency)
                : t('purchase.priceUnknown', 'Price unknown')}
            </ScaledText>
            <TextInput
              editable={!busy}
              accessibilityLabel="Selected size"
              placeholder="Selected size (optional)"
              value={size}
              onChangeText={setSize}
              maxLength={24}
              style={styles.input}
            />
            <View style={styles.row}>
              {candidate.availableSizes?.map((value) => button(value, () => setSize(value), value))}
            </View>
            {variants.length > 1 && (
              <CatalogVariantPicker
                selectedId={candidate.id}
                variants={variants.map((variant) =>
                  hasProductOnlyCatalogProvenance(variant) ? variant : { ...variant, imageUrl: '' }
                )}
                onSelect={(id) => {
                  const variant = variants.find((item) => item.id === id);
                  if (variant) {
                    setCandidate(variant);
                    setSize('');
                    setAnalysis(undefined);
                  }
                }}
              />
            )}

            <ScaledText>
              {t('purchase.checkout', 'Known checkout costs (leave unknown fields empty)')}
            </ScaledText>
            <View style={styles.row}>
              {(
                [
                  ['Shipping', shipping, setShipping],
                  ['Tax / duty', tax, setTax],
                  ['Required fees', fees, setFees],
                ] as const
              ).map(([label, value, setter]) => (
                <TextInput
                  editable={!busy}
                  key={label}
                  accessibilityLabel={label}
                  placeholder={label}
                  value={value}
                  onChangeText={setter}
                  keyboardType="decimal-pad"
                  style={styles.input}
                />
              ))}
            </View>
            <View style={styles.row}>
              <ScaledText>
                {t('purchase.intentionalMultiple', 'I intentionally want a similar item')}
              </ScaledText>
              <Switch
                accessibilityLabel="Intentional similar purchase"
                disabled={busy}
                value={allowMultiples}
                onValueChange={(value) => {
                  setAllowMultiples(value);
                  setAnalysis(undefined);
                }}
              />
            </View>
            {button(t('purchase.runCheck', 'Run BuyCheck'), check)}
            {button(t('purchase.save', 'Save this product locally'), () =>
              run(() => privateStore.saveCandidate(candidate))
            )}
            {button(t('purchase.openMerchant', 'Open exact merchant product'), () => {
              const target = getShopDestination({ ...candidate, shopUrl: candidate.sourceUrl });
              void Linking.openURL(target).catch(() =>
                setMessage(t('purchase.openFailed', 'Could not open the merchant'))
              );
            })}
            <View style={styles.row}>
              <TextInput
                editable={!busy}
                accessibilityLabel="Watch threshold"
                placeholder="Item-price threshold"
                value={threshold}
                onChangeText={setThreshold}
                keyboardType="decimal-pad"
                style={styles.input}
              />
              {button(t('purchase.watch', 'Watch item price locally'), () =>
                run(async () => {
                  if (!threshold || !Number.isFinite(Number(threshold)) || Number(threshold) < 0) {
                    setMessage('Enter a valid threshold');
                    return;
                  }
                  await privateStore.saveWatch({
                    productId: candidate.id,
                    variantId: candidate.id,
                    selectedSize: size || undefined,
                    currency: candidate.currency || currency,
                    threshold: Number(threshold),
                    status: 'active',
                  });
                  setMessage(
                    t(
                      'purchase.watchLocal',
                      'Saved on this device. Background monitoring is not active.'
                    )
                  );
                })
              )}
            </View>
          </View>
        )}
        {analysis &&
          analysisOwner.current === authEvents.getUserId() &&
          analysisToken.current === formToken && (
            <View style={styles.card}>
              <ScaledText style={styles.title}>{analysis.verdict.replace(/_/g, ' ')}</ScaledText>
              <BetaFeedbackButton key={analysis.selectedVariantId} feature="buycheck" diagnostics={{ confidence: analysis.overallConfidence, candidateCount: analysis.wardrobe.usefulOutfitUnlocks.evaluations, ruleVersion: 'purchase-local-v1', fallback: 'none' }} />
              <ScaledText>
                {t('purchase.confidence', 'Confidence')}: {analysis.overallConfidence}
              </ScaledText>
              <ScaledText>
                {t('purchase.size', 'Recommended size')}:{' '}
                {analysis.fit.recommendedSize || t('purchase.unknown', 'Unknown')} ·{' '}
                {analysis.fit.confidence}
              </ScaledText>
              <ScaledText>
                {analysis.fit.evidenceCount} fit observations ·{' '}
                {analysis.fit.provenance.join(', ') || 'No personal fit evidence'}
              </ScaledText>
              <ScaledText>
                {analysis.fit.notes.join('\n')}
                {analysis.fit.zoneNotes.join('\n')}
              </ScaledText>
              <ScaledText>
                {analysis.wardrobe.wardrobeValue} wardrobe value ·{' '}
                {analysis.wardrobe.duplicateRisk.level} duplicate risk
              </ScaledText>
              <ScaledText>
                {analysis.wardrobe.usefulOutfitUnlocks.total} validated sampled new looks
              </ScaledText>
              <ScaledText>{analysis.explanation.join('\n')}</ScaledText>
              {analysis.wardrobe.duplicateRisk.substitutes.map((substitute) => (
                <ScaledText key={substitute.item.id}>
                  Already owned: {substitute.item.name} · {substitute.evidence.join(', ')}
                </ScaledText>
              ))}
              <ScaledText>
                {analysis.wardrobe.usefulOutfitUnlocks.confidence}
                {analysis.wardrobe.usefulOutfitUnlocks.truncated ? ' · capped sample' : ''}
              </ScaledText>
              {Object.entries(analysis.wardrobe.usefulOutfitUnlocks.contextBreakdown).map(
                ([label, count]) => (
                  <ScaledText key={label}>
                    {label}: {count} validated looks in this context
                  </ScaledText>
                )
              )}
              {analysis.wardrobe.usefulOutfitUnlocks.looks.slice(0, 3).map((look, index) => (
                <ScaledText key={index}>
                  {look.items.map((item) => item.name).join(' · ')}
                </ScaledText>
              ))}
              <ScaledText>
                {analysis.purchase.priceAssessment?.evidence.join('\n') ||
                  'Not enough price history'}
              </ScaledText>
              {analysis.purchase.offers?.map(({ offer, truePrice: price }) => (
                <View key={offer.id} style={{ gap: 6 }}>
                  <ScaledText>
                    {offer.merchant}:{' '}
                    {price.total === undefined
                      ? `Known subtotal ${formatMoney(price.knownSubtotal, price.currency)}; unknown ${price.unknownComponents.join(', ')}`
                      : formatMoney(price.total, price.currency)}
                  </ScaledText>
                  <ScaledText>
                    {size
                      ? offer.sizes?.[size] === true
                        ? 'Selected size reported in stock'
                        : offer.sizes?.[size] === false
                          ? 'Selected size reported out of stock'
                          : 'Selected-size stock unknown'
                      : offer.inStock === true
                        ? 'Reported in stock'
                        : offer.inStock === false
                          ? 'Reported out of stock'
                          : 'Stock unknown'}
                  </ScaledText>
                  {offer.returnPolicy?.source && <ScaledText>{offer.returnPolicy.text}</ScaledText>}
                  {button(
                    `${t('purchase.openMerchant', 'Open merchant product')}: ${offer.merchant}`,
                    () => {
                      void Linking.openURL(
                        getShopDestination({ ...candidate, shopUrl: offer.url })
                      ).catch(() =>
                        setMessage(t('purchase.openFailed', 'Could not open the merchant'))
                      );
                    },
                    `offer-${offer.id}`
                  )}
                </View>
              ))}
              {analysis.wardrobe.usefulOutfitUnlocks.looks[0] &&
                button(t('purchase.cheaperLook', 'Make a representative look cheaper'), () => {
                  const items = analysis.wardrobe.usefulOutfitUnlocks.looks[0].items;
                  setLookIds(items.map((item) => item.id));
                  setLookSnapshots(Object.fromEntries(items.map((item) => [item.id, item])));
                  setIntent('save');
                })}
              <View style={styles.row}>
                {(
                  [
                    'overall',
                    'cheaper',
                    'closest-style',
                    'fit',
                    'wardrobe',
                    'lowest-duplicate',
                    'missing-piece',
                  ] as const
                ).map((mode) =>
                  button(
                    t(`purchase.mode.${mode}`, mode.replace(/-/g, ' ')),
                    () => explore(mode),
                    mode
                  )
                )}
              </View>
            </View>
          )}
        {intent !== 'check' &&
          button(t('purchase.compare', 'Compare using my wardrobe'), () => explore())}
        {tool &&
          toolOwner.current === authEvents.getUserId() &&
          toolToken.current === formToken && (
            <View style={styles.card}>
              {tool.lines.map((line, index) => (
                <ScaledText key={index}>{line}</ScaledText>
              ))}
              {tool.items?.map((item) =>
                button(
                  item.name,
                  () => {
                    setCandidate(item);
                    setIntent('check');
                    setAnalysis(undefined);
                  },
                  item.id
                )
              )}
            </View>
          )}
        {(intent === 'save' || intent === 'upgrade') && (
          <View style={styles.card}>
            <ScaledText>
              {t('purchase.lookPieces', 'Choose exact look pieces (four required)')}
            </ScaledText>
            {lookIds.map((id) =>
              button(
                `${lockedLookIds.includes(id) ? '✓ ' : ''}Keep ${lookSnapshots[id]?.name || owned.find((item) => item.id === id)?.name || id}`,
                () =>
                  setLockedLookIds((previous) =>
                    previous.includes(id) ? previous.filter((key) => key !== id) : [...previous, id]
                  ),
                `keep-${id}`
              )
            )}
            {[...owned, ...results].slice(0, 40).map((item) =>
              button(
                `${lookIds.includes(item.id) ? '✓ ' : ''}${item.name}`,
                () => {
                  setLookSnapshots((previous) => ({ ...previous, [item.id]: item }));
                  setLookIds((previous) =>
                    previous.includes(item.id)
                      ? previous.filter((id) => id !== item.id)
                      : previous.length < 4
                        ? [...previous, item.id]
                        : previous
                  );
                },
                item.id
              )
            )}
          </View>
        )}
        {results.slice(0, 24).map((item) =>
          button(
            `${item.brand} · ${item.name}`,
            () => {
              setCandidate(item);
              setVariants([]);
              setAnalysis(undefined);
              setIntent('check');
            },
            item.id
          )
        )}
        <View style={styles.card}>
          <ScaledText>{t('purchase.feedbackSetting', 'Optional fit feedback prompts')}</ScaledText>
          <Switch
            accessibilityLabel="Fit feedback prompts"
            disabled={busy}
            value={privateStore.feedbackEnabled}
            onValueChange={(value) => run(() => privateStore.setFeedbackEnabled(value))}
          />
          <ScaledText>
            {fitProfile.evidenceCount} recent fit observations ·{' '}
            {fitProfile.referenceGarments.length} reference garments
          </ScaledText>
          {fitProfile.referenceGarments.slice(0, 8).map((reference) => (
            <ScaledText key={reference.evidenceId}>
              {reference.brand} · {reference.subcategory || reference.category} ·{' '}
              {reference.sizeLabel}
              {reference.productId
                ? ` · ${owned.find((item) => item.id === reference.productId)?.name || 'Reference garment'}`
                : ''}
            </ScaledText>
          ))}
          <ScaledText>
            {t(
              'purchase.localPrivacy',
              'Fit evidence stays on this device. It is not sent to merchants, analytics or an AI prompt.'
            )}
          </ScaledText>
          {privateStore.bookmarks.map((item) =>
            button(
              `${t('purchase.savedProduct', 'Saved product')}: ${item.name}`,
              () => setCandidate(item),
              item.id
            )
          )}
          {privateStore.watches.map((watch) => (
            <View key={watch.id}>
              <ScaledText>
                {formatMoney(watch.threshold, watch.currency)} · {watch.status} ·{' '}
                {watch.selectedSize || 'any size'}
              </ScaledText>
              {button(
                watch.status === 'active' ? 'Pause watch' : 'Resume watch',
                () =>
                  run(() =>
                    privateStore.updateWatch(
                      watch.id,
                      watch.status === 'active' ? 'paused' : 'active'
                    )
                  ),
                watch.id
              )}
              {button(
                'Check observed prices',
                () =>
                  run(async (current) => {
                    const observed = await purchasePriceReads.read(
                      watch.variantId || watch.productId
                    );
                    if (!current()) return;
                    const result = evaluatePriceWatch(watch, observed.offers);
                    await privateStore.markWatchChecked(watch.id, result.triggered);
                    if (current()) setMessage(result.reason);
                  }),
                `check-${watch.id}`
              )}
            </View>
          ))}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}
const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fff' },
  content: { padding: 20, gap: 12, paddingBottom: 50 },
  title: { fontSize: 24, fontWeight: '700', color: '#0A1931' },
  subtitle: { fontSize: 18, fontWeight: '600' },
  row: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  input: {
    minWidth: 90,
    borderWidth: 1,
    borderColor: '#CCD4DF',
    borderRadius: 10,
    padding: 12,
    color: '#0A1931',
  },
  button: { padding: 12, borderRadius: 10, backgroundColor: '#EBF1F8' },
  card: { backgroundColor: '#F5F7FA', padding: 16, borderRadius: 16, gap: 10 },
});
