import React, { useCallback, useEffect, useState } from 'react';
import { ActivityIndicator, Alert, Modal, ScrollView, StyleSheet, Switch, TextInput, TouchableOpacity, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { ScaledText } from '../../components/ui/ScaledText';
import { Image } from '../../src/components/MediaImage';
import { supabase } from '../../lib/supabase';

type LookRow = {
    id: string; brand: string; collection_name: string; title: string; image_url: string;
    source_url: string; sort_order: number; is_active: boolean;
};
type ProductRow = {
    id: string; look_id: string; name: string; category: string; image_url: string;
    price: number | null; currency: string; product_url: string; sort_order: number; is_active: boolean;
};

const blankLook = (): LookRow => ({ id: '', brand: '', collection_name: '', title: '', image_url: '', source_url: '', sort_order: 0, is_active: true });
const blankProduct = (lookId: string): ProductRow => ({ id: '', look_id: lookId, name: '', category: 'TOP', image_url: '', price: null, currency: 'USD', product_url: '', sort_order: 0, is_active: true });
const TextField = ({ label, value, onChangeText, placeholder, keyboardType = 'default' }: { label: string; value: string; onChangeText: (value: string) => void; placeholder: string; keyboardType?: 'default' | 'url' | 'decimal-pad' | 'number-pad' }) => (
    <View style={s.field}><ScaledText style={s.label}>{label}</ScaledText><TextInput style={s.input} value={value} onChangeText={onChangeText} placeholder={placeholder} autoCapitalize={keyboardType === 'url' ? 'none' : 'sentences'} autoCorrect={keyboardType !== 'url'} keyboardType={keyboardType} /></View>
);

export const AdminLooksTab = () => {
    const [looks, setLooks] = useState<LookRow[]>([]);
    const [products, setProducts] = useState<ProductRow[]>([]);
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [editingLook, setEditingLook] = useState<LookRow | null>(null);
    const [managingLook, setManagingLook] = useState<LookRow | null>(null);
    const [editingProduct, setEditingProduct] = useState<ProductRow | null>(null);

    const load = useCallback(async () => {
        setLoading(true);
        const [lookResult, productResult] = await Promise.all([
            supabase.from('inspo_looks').select('*').order('sort_order', { ascending: true }),
            supabase.from('inspo_look_products').select('*').order('sort_order', { ascending: true }),
        ]);
        if (lookResult.error || productResult.error) {
            Alert.alert('Could not load outfit previews', lookResult.error?.message || productResult.error?.message);
        } else {
            setLooks((lookResult.data ?? []) as LookRow[]);
            setProducts((productResult.data ?? []) as ProductRow[]);
        }
        setLoading(false);
    }, []);

    useEffect(() => { void load(); }, [load]);

    const saveLook = async () => {
        if (!editingLook) return;
        if (!editingLook.brand.trim() || !editingLook.title.trim() || !editingLook.image_url.trim() || !editingLook.source_url.trim()) {
            Alert.alert('Missing information', 'Brand, title, preview image URL, and collection URL are required.');
            return;
        }
        setSaving(true);
        const row = { ...editingLook, id: editingLook.id || `look-${Date.now()}`, brand: editingLook.brand.trim(), title: editingLook.title.trim(), collection_name: editingLook.collection_name.trim(), image_url: editingLook.image_url.trim(), source_url: editingLook.source_url.trim() };
        const { error } = editingLook.id
            ? await supabase.from('inspo_looks').update(row).eq('id', editingLook.id)
            : await supabase.from('inspo_looks').insert(row);
        setSaving(false);
        if (error) { Alert.alert('Save failed', error.message); return; }
        setEditingLook(null);
        await load();
    };

    const saveProduct = async () => {
        if (!editingProduct) return;
        if (!editingProduct.name.trim() || !editingProduct.image_url.trim() || !editingProduct.product_url.trim()) {
            Alert.alert('Missing information', 'Product name, white-background image URL, and product page URL are required.');
            return;
        }
        setSaving(true);
        const row = { ...editingProduct, id: editingProduct.id || `product-${Date.now()}`, name: editingProduct.name.trim(), category: editingProduct.category.trim() || 'ITEM', image_url: editingProduct.image_url.trim(), product_url: editingProduct.product_url.trim() };
        const { error } = editingProduct.id
            ? await supabase.from('inspo_look_products').update(row).eq('id', editingProduct.id)
            : await supabase.from('inspo_look_products').insert(row);
        setSaving(false);
        if (error) { Alert.alert('Save failed', error.message); return; }
        setEditingProduct(null);
        await load();
    };

    const deleteLook = (look: LookRow) => Alert.alert('Delete preview?', `${look.brand} · ${look.title} and its product cards will be deleted.`, [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Delete', style: 'destructive', onPress: async () => {
            const { error } = await supabase.from('inspo_looks').delete().eq('id', look.id);
            if (error) Alert.alert('Delete failed', error.message); else await load();
        } },
    ]);

    const deleteProduct = (product: ProductRow) => Alert.alert('Delete product card?', product.name, [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Delete', style: 'destructive', onPress: async () => {
            const { error } = await supabase.from('inspo_look_products').delete().eq('id', product.id);
            if (error) Alert.alert('Delete failed', error.message); else await load();
        } },
    ]);

    return (
        <View style={s.container}>
            <View style={s.intro}>
                <ScaledText style={s.introTitle}>Remote outfit previews</ScaledText>
                <ScaledText style={s.introText}>Changes save to Supabase and appear in the app without a store update. Use product image URLs from the brand’s white-background product pages.</ScaledText>
            </View>
            <TouchableOpacity style={s.addLook} onPress={() => setEditingLook(blankLook())}>
                <Ionicons name="add-circle" size={20} color="#FFF" /><ScaledText style={s.addLookText}>Add outfit preview</ScaledText>
            </TouchableOpacity>
            {loading ? <View style={s.center}><ActivityIndicator size="large" color="#0A1931" /></View> : (
                <ScrollView contentContainerStyle={s.list}>
                    {looks.map((look) => {
                        const count = products.filter((product) => product.look_id === look.id).length;
                        return <View key={look.id} style={[s.lookRow, !look.is_active && s.inactive]}>
                            <Image source={{ uri: look.image_url }} style={s.lookImage} />
                            <View style={s.lookInfo}>
                                <ScaledText style={s.brand}>{look.brand} · {look.collection_name}</ScaledText>
                                <ScaledText style={s.lookTitle}>{look.title}</ScaledText>
                                <ScaledText style={s.meta}>{count} product cards · order {look.sort_order}</ScaledText>
                                <View style={s.actions}>
                                    <TouchableOpacity style={s.action} onPress={() => setEditingLook({ ...look })}><Ionicons name="create-outline" size={17} color="#0A1931" /><ScaledText style={s.actionText}>Edit preview</ScaledText></TouchableOpacity>
                                    <TouchableOpacity style={s.action} onPress={() => setManagingLook(look)}><Ionicons name="grid-outline" size={17} color="#0A1931" /><ScaledText style={s.actionText}>Products</ScaledText></TouchableOpacity>
                                    <TouchableOpacity style={s.action} onPress={() => deleteLook(look)}><Ionicons name="trash-outline" size={17} color="#D23C3C" /></TouchableOpacity>
                                </View>
                            </View>
                        </View>;
                    })}
                </ScrollView>
            )}

            <Modal visible={!!editingLook} animationType="slide" presentationStyle="pageSheet" onRequestClose={() => setEditingLook(null)}>
                <SafeAreaView style={s.modal}>
                    <View style={s.modalHeader}><TouchableOpacity onPress={() => setEditingLook(null)}><ScaledText style={s.cancel}>Cancel</ScaledText></TouchableOpacity><ScaledText style={s.modalTitle}>Preview details</ScaledText><TouchableOpacity onPress={saveLook} disabled={saving}>{saving ? <ActivityIndicator /> : <ScaledText style={s.save}>Save</ScaledText>}</TouchableOpacity></View>
                    {editingLook && <ScrollView contentContainerStyle={s.form}>
                        {editingLook.image_url ? <Image source={{ uri: editingLook.image_url }} style={s.formImagePreview} resizeMode="contain" /> : null}
                        <TextField label="Brand" value={editingLook.brand} onChangeText={(brand) => setEditingLook({ ...editingLook, brand })} placeholder="Canali" />
                        <TextField label="Collection / season" value={editingLook.collection_name} onChangeText={(collection_name) => setEditingLook({ ...editingLook, collection_name })} placeholder="Spring/Summer 2027 preview" />
                        <TextField label="Title" value={editingLook.title} onChangeText={(title) => setEditingLook({ ...editingLook, title })} placeholder="Look 01" />
                        <TextField label="Preview image URL" value={editingLook.image_url} onChangeText={(image_url) => setEditingLook({ ...editingLook, image_url })} placeholder="https://brand.com/look.jpg" keyboardType="url" />
                        <TextField label="Collection page URL" value={editingLook.source_url} onChangeText={(source_url) => setEditingLook({ ...editingLook, source_url })} placeholder="https://brand.com/collection" keyboardType="url" />
                        <TextField label="Display order" value={String(editingLook.sort_order)} onChangeText={(value) => setEditingLook({ ...editingLook, sort_order: Number(value) || 0 })} placeholder="10" keyboardType="number-pad" />
                        <View style={s.switchRow}><ScaledText style={s.label}>Visible in the app</ScaledText><Switch value={editingLook.is_active} onValueChange={(is_active) => setEditingLook({ ...editingLook, is_active })} /></View>
                    </ScrollView>}
                </SafeAreaView>
            </Modal>

            <Modal visible={!!managingLook} animationType="slide" presentationStyle="pageSheet" onRequestClose={() => setManagingLook(null)}>
                <SafeAreaView style={s.modal}>
                    <View style={s.modalHeader}><TouchableOpacity onPress={() => setManagingLook(null)}><ScaledText style={s.cancel}>Back</ScaledText></TouchableOpacity><ScaledText style={s.modalTitle} numberOfLines={1}>Product cards</ScaledText><TouchableOpacity onPress={() => managingLook && setEditingProduct(blankProduct(managingLook.id))}><ScaledText style={s.save}>Add</ScaledText></TouchableOpacity></View>
                    <ScaledText style={s.productSubtitle}>{managingLook?.brand} · {managingLook?.title}</ScaledText>
                    <ScrollView contentContainerStyle={s.list}>
                        {products.filter((product) => product.look_id === managingLook?.id).map((product) => <View key={product.id} style={s.productRow}>
                            <Image source={{ uri: product.image_url }} style={s.productImage} />
                            <View style={s.lookInfo}><ScaledText style={s.brand}>{product.category}</ScaledText><ScaledText style={s.lookTitle}>{product.name}</ScaledText><ScaledText style={s.meta}>{product.price == null ? 'No price' : `${product.price} ${product.currency}`}</ScaledText></View>
                            <View style={s.productActions}><TouchableOpacity onPress={() => setEditingProduct({ ...product })}><Ionicons name="create-outline" size={20} color="#0A1931" /></TouchableOpacity><TouchableOpacity onPress={() => deleteProduct(product)}><Ionicons name="trash-outline" size={20} color="#D23C3C" /></TouchableOpacity></View>
                        </View>)}
                        {products.filter((product) => product.look_id === managingLook?.id).length === 0 && <ScaledText style={s.empty}>No product cards yet. Add a card using the exact product photo, name, price, and web link.</ScaledText>}
                    </ScrollView>
                </SafeAreaView>
            </Modal>

            <Modal visible={!!editingProduct} animationType="slide" presentationStyle="pageSheet" onRequestClose={() => setEditingProduct(null)}>
                <SafeAreaView style={s.modal}>
                    <View style={s.modalHeader}><TouchableOpacity onPress={() => setEditingProduct(null)}><ScaledText style={s.cancel}>Cancel</ScaledText></TouchableOpacity><ScaledText style={s.modalTitle}>Product card</ScaledText><TouchableOpacity onPress={saveProduct} disabled={saving}>{saving ? <ActivityIndicator /> : <ScaledText style={s.save}>Save</ScaledText>}</TouchableOpacity></View>
                    {editingProduct && <ScrollView contentContainerStyle={s.form}>
                        {editingProduct.image_url ? <Image source={{ uri: editingProduct.image_url }} style={s.formProductPreview} resizeMode="contain" /> : null}
                        <TextField label="Product name" value={editingProduct.name} onChangeText={(name) => setEditingProduct({ ...editingProduct, name })} placeholder="Brown wool trousers" />
                        <TextField label="Category" value={editingProduct.category} onChangeText={(category) => setEditingProduct({ ...editingProduct, category })} placeholder="BOTTOM" />
                        <TextField label="White-background image URL" value={editingProduct.image_url} onChangeText={(image_url) => setEditingProduct({ ...editingProduct, image_url })} placeholder="https://brand.com/product.jpg" keyboardType="url" />
                        <TextField label="Price" value={editingProduct.price == null ? '' : String(editingProduct.price)} onChangeText={(value) => setEditingProduct({ ...editingProduct, price: value.trim() ? Number(value) || 0 : null })} placeholder="450" keyboardType="decimal-pad" />
                        <TextField label="Currency" value={editingProduct.currency || ''} onChangeText={(currency) => setEditingProduct({ ...editingProduct, currency: currency.toUpperCase() })} placeholder="GBP" />
                        <TextField label="Product page URL" value={editingProduct.product_url} onChangeText={(product_url) => setEditingProduct({ ...editingProduct, product_url })} placeholder="https://brand.com/product" keyboardType="url" />
                        <TextField label="Display order" value={String(editingProduct.sort_order)} onChangeText={(value) => setEditingProduct({ ...editingProduct, sort_order: Number(value) || 0 })} placeholder="10" keyboardType="number-pad" />
                        <View style={s.switchRow}><ScaledText style={s.label}>Visible in the app</ScaledText><Switch value={editingProduct.is_active} onValueChange={(is_active) => setEditingProduct({ ...editingProduct, is_active })} /></View>
                    </ScrollView>}
                </SafeAreaView>
            </Modal>
        </View>
    );
};

const s = StyleSheet.create({
    container: { flex: 1, backgroundColor: '#F4F6FA' },
    intro: { padding: 16, backgroundColor: '#FFF', marginHorizontal: 16, marginTop: 8, borderRadius: 16 },
    introTitle: { color: '#0A1931', fontSize: 17, fontWeight: '700' },
    introText: { color: '#68717E', fontSize: 13, lineHeight: 19, marginTop: 5 },
    addLook: { marginHorizontal: 16, marginVertical: 10, borderRadius: 13, backgroundColor: '#0A1931', minHeight: 46, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8 },
    addLookText: { color: '#FFF', fontSize: 14, fontWeight: '700' },
    list: { padding: 16, gap: 12, paddingBottom: 40 },
    lookRow: { flexDirection: 'row', padding: 10, borderRadius: 16, backgroundColor: '#FFF', gap: 12, alignItems: 'center' },
    inactive: { opacity: 0.55 },
    lookImage: { width: 76, height: 96, borderRadius: 10, backgroundColor: '#F5F5F3' },
    lookInfo: { flex: 1 },
    brand: { color: '#77808C', fontSize: 10, fontWeight: '700', letterSpacing: 0.5, textTransform: 'uppercase' },
    lookTitle: { color: '#0A1931', fontSize: 14, fontWeight: '700', marginTop: 3 },
    meta: { color: '#7A8492', fontSize: 11, marginTop: 4 },
    actions: { flexDirection: 'row', gap: 12, marginTop: 10, alignItems: 'center' },
    action: { flexDirection: 'row', alignItems: 'center', gap: 4 },
    actionText: { color: '#0A1931', fontSize: 11, fontWeight: '600' },
    center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
    modal: { flex: 1, backgroundColor: '#F5F6F8' },
    modalHeader: { minHeight: 52, paddingHorizontal: 16, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: '#D6D9E0' },
    modalTitle: { color: '#0A1931', fontSize: 16, fontWeight: '700', maxWidth: '58%' },
    cancel: { color: '#647080', fontSize: 15 },
    save: { color: '#087BE3', fontSize: 15, fontWeight: '700' },
    form: { padding: 18, paddingBottom: 50 },
    formImagePreview: { width: '100%', height: 210, backgroundColor: '#FFFFFF', borderRadius: 14, marginBottom: 12 },
    formProductPreview: { width: '100%', height: 230, backgroundColor: '#FFFFFF', borderRadius: 14, marginBottom: 12 },
    field: { marginBottom: 10 },
    label: { color: '#344257', fontSize: 12, fontWeight: '700', marginBottom: 5 },
    input: { minHeight: 44, backgroundColor: '#FFF', borderWidth: 1, borderColor: '#DCE1E8', borderRadius: 10, paddingHorizontal: 12, color: '#0A1931', fontSize: 14 },
    switchRow: { marginTop: 6, paddingVertical: 8, flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
    productSubtitle: { color: '#68717E', fontSize: 12, paddingHorizontal: 18, paddingTop: 12 },
    productRow: { backgroundColor: '#FFF', borderRadius: 14, padding: 10, flexDirection: 'row', alignItems: 'center', gap: 10 },
    productImage: { width: 68, height: 68, backgroundColor: '#FFF', borderRadius: 10 },
    productActions: { gap: 14, paddingHorizontal: 4 },
    empty: { color: '#68717E', fontSize: 14, lineHeight: 20, padding: 20, textAlign: 'center' },
});
