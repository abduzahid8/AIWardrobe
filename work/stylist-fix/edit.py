from pathlib import Path
p=Path('screens/InspoScreen.tsx');s=p.read_text();a=s.index('// shop_catalog rows are priced');b=s.index('const formatShopPrice',a);s=s[:a]+s[b:];s=s.replace("return formatMoney(n, 'UZS');","return formatMoney(n, item.currency || 'USD');");s=s.replace("    const {\n        items: syncedShopItems,","    const [searchQuery, setSearchQuery] = useState('');\n    const deferredSearchQuery = useDeferredValue(searchQuery);\n    const {\n        items: syncedShopItems,",1);s=s.replace("} = useShopCatalog({ source: 'all' });","} = useShopCatalog({ source: 'all', search: deferredSearchQuery });",1);s=s.replace("    const [searchQuery, setSearchQuery] = useState('');",'',1) if False else s
# Remove later duplicate declarations.
start=s.index('    const [searchQuery, setSearchQuery]');pos=s.find("    const [searchQuery, setSearchQuery] = useState('');",start+1);s=s[:pos]+s[pos:].replace("    const [searchQuery, setSearchQuery] = useState('');",'',1)
pos=s.find('    const deferredSearchQuery = useDeferredValue(searchQuery);',s.index('    const deferredSearchQuery')+1);s=s[:pos]+s[pos:].replace('    const deferredSearchQuery = useDeferredValue(searchQuery);','',1);p.write_text(s)
p=Path('hooks/useShopCatalog.ts');s=p.read_text().replace('function cacheKey(source: string, cat: string) { return `${source}|${cat}`; }','function cacheKey(source: string, cat: string, search = \'\') { return JSON.stringify([source, cat, search]); }');s=s.replace('    enabled?: boolean;','    enabled?: boolean;\n    /** Search the entire active catalog before pagination. */\n    search?: string;');s=s.replace('activeSource: string, signal?: AbortSignal)',"activeSource: string, signal?: AbortSignal, search = '')");s=s.replace(".order('sort_order', { ascending: true })",".order('sort_order', { ascending: true })\n        .order('id', { ascending: true })");s=s.replace('    return q;','''    if (search.trim()) {
        // Quote filter values so punctuation cannot alter the PostgREST grammar.
        const pattern = `%${search.trim().replace(/[%_*\\\\]/g, c => `\\\\${c}`)}%`;
        const value = JSON.stringify(pattern);
        q = q.or(`name.ilike.${value},brand.ilike.${value},description.ilike.${value}`);
    }
    return q;''',1);s=s.replace('    append: boolean,\n): Promise',"    append: boolean,\n    search = '',\n): Promise");s=s.replace('cacheKey(activeSource, cat);','cacheKey(activeSource, cat, search);');s=s.replace('buildQuery(page, cat, activeSource, controller.signal)','buildQuery(page, cat, activeSource, controller.signal, search)');s=s.replace('    enabled = true,','    enabled = true,\n    search: rawSearch = \'\',');s=s.replace('    const [items, setItems]',"    const search = rawSearch.trim().toLowerCase();\n    const searchRef = useRef(search);\n    searchRef.current = search;\n    const requestVersion = useRef(0);\n    const [items, setItems]",1);s=s.replace('append: boolean, silent = false) => {',"append: boolean, silent = false) => {\n        const search = searchRef.current;\n        const version = ++requestVersion.current;\n        const current = () => version === requestVersion.current && search === searchRef.current && cat === categoryRef.current && activeSource === sourceRef.current && enabledRef.current;");s=s.replace('loadCatalogPage(page, cat, activeSource, append);','loadCatalogPage(page, cat, activeSource, append, search);\n            if (!current()) return;\n            pageRef.current = page;');s=s.replace('        } catch (err: any) {\n            // Already logged','        } catch (err: any) {\n            if (!current()) return;\n            // Already logged');s=s.replace('        } finally {\n            setLoading(false);','        } finally {\n            if (current()) {\n            setLoading(false);');s=s.replace('            setLoadingMore(false);\n        }\n    }, []);','            setLoadingMore(false);\n            }\n        }\n    }, []);');s=s.replace('const key = cacheKey(source, category);\n        const cached', 'const key = cacheKey(source, category, search);\n        const cached') # only hook? prefetch uses no search variable! revert first
s=s.replace('        const key = cacheKey(source, category, search);\n        const cached', '        const key = cacheKey(source, category);\n        const cached',1)
s=s.replace('const seed = getStaticSeed(category, source);',"const seed = search ? [] : getStaticSeed(category, source);");s=s.replace('}, [category, source, enabled]);','}, [category, source, enabled, search]);');s=s.replace('        pageRef.current = nextPage;\n','');s=s.replace('        if (!enabled) {','        requestVersion.current += 1;\n        setLoadingMore(false);\n        if (!enabled) {',1);p.write_text(s)
p=Path('src/utils/mappingUtils.ts');s=p.read_text();a=s.index('  if (sec ===');b=s.index("  return 'tops'",a);s=s[:a]+'''  // Specific garment identity wins over stale generic sections.
  if (/trouser|pant|jean|chino|slack|skirt|short|брюк|джинс/.test(cat)) return 'bottoms';
  if (/jacket|coat|blazer|puffer|parka|куртк|пиджак/.test(cat)) return 'outerwear';
  if (/shoe|sneaker|boot|sandal|loafer|кроссов/.test(cat)) return 'shoes';
  if (/shirt|blouse|sweater|hoodie|polo|рубаш|футбол/.test(cat)) return 'tops';
  if (/bag|hat|scarf|belt|watch/.test(cat)) return 'accessories';
  if (/sport|gym|legging/.test(cat)) return 'sportswear';
  const aliases: Record<string, string> = {top:'tops',tops:'tops',upper_body:'tops',bottom:'bottoms',bottoms:'bottoms',lower_body:'bottoms',shoes:'shoes',outerwear:'outerwear',accessory:'accessories',accessories:'accessories',sportswear:'sportswear',homewear:'homewear'};
  if (aliases[cat]) return aliases[cat];
  if (aliases[sec]) return aliases[sec];
''' +s[b:];p.write_text(s)
p=Path('screens/AIOutfitmaker.tsx');s=p.read_text().replace("occasion: edgePrompt || occasionToUse || 'Any',","occasion: occasionToUse || 'Any',");a=s.index('          // Lock the anchor item into every outfit');b=s.index('          // When source',a);s=s[:a]+'''          // A locked owned piece remains the anchor; the other slots use the catalog.
          anchorItemId: baseItemId,
'''+s[b:];s=s.replace('''              {error ? (
                <View style={[styles.errorBox, { marginTop: 16 }]}>
                  <ScaledText style={styles.errorText}>{error}</ScaledText>
                </View>
              ) : null}''','');needle='''          {/* Sticky bottom CTA */}
          <View style={styles.stickyFooter}>''';s=s.replace(needle,'''          {/* Keep failures visible beside the retry action, outside scrolling choices. */}
          <View style={styles.stickyFooter}>
            {error ? <View accessibilityRole="alert" accessibilityLiveRegion="assertive" style={[styles.errorBox, { marginBottom: 12 }]}>
              <ScaledText style={styles.errorText}>{error}</ScaledText>
            </View> : null}''');p.write_text(s)
