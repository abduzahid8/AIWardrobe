/**
 * Confirmation Helpers — wraps destructive actions in Alert.alert confirmations.
 *
 * Use these instead of calling store actions directly for delete/remove actions.
 *
 * @example
 * import { confirmRemoveItem } from '../src/utils/confirmations';
 * <TouchableOpacity onPress={() => confirmRemoveItem(item.id, item.name)}>
 */
import { Alert } from 'react-native';
import { useTranslation } from 'react-i18next';
import useWardrobeStore from '../../store/wardrobeStore';

/**
 * Show confirmation dialog before removing a clothing item.
 */
export const confirmRemoveItem = (itemId: string, itemName?: string): void => {
    const { t } = useTranslation();
    const label = itemName || t('confirmations.thisItem');
    Alert.alert(
        t('confirmations.removeItem'),
        t('confirmations.removeItemConfirm', { label }),
        [
            { text: t('common.cancel'), style: 'cancel' },
            {
                text: t('common.remove'),
                style: 'destructive',
                onPress: () => useWardrobeStore.getState().removeItem(itemId),
            },
        ],
    );
};

/**
 * Show confirmation dialog before deleting an outfit.
 */
export const confirmDeleteOutfit = (outfitId: string, outfitName?: string): void => {
    const { t } = useTranslation();
    const label = outfitName || t('confirmations.thisOutfit');
    Alert.alert(
        t('confirmations.deleteOutfit'),
        t('confirmations.deleteOutfitConfirm', { label }),
        [
            { text: t('common.cancel'), style: 'cancel' },
            {
                text: t('common.delete'),
                style: 'destructive',
                onPress: () => {
                    // Remove from store
                    const { outfits } = useWardrobeStore.getState();
                    useWardrobeStore.setState({
                        outfits: outfits.filter(o => o.id !== outfitId),
                    });
                },
            },
        ],
    );
};

/**
 * Generic confirmation for any destructive action.
 */
export const confirmDestructive = (
    title: string,
    message: string,
    onConfirm: () => void,
    confirmLabel?: string,
): void => {
    const { t } = useTranslation();
    const label = confirmLabel || t('common.delete');
    Alert.alert(title, message, [
        { text: t('common.cancel'), style: 'cancel' },
        { text: label, style: 'destructive', onPress: onConfirm },
    ]);
};

export default {
    confirmRemoveItem,
    confirmDeleteOutfit,
    confirmDestructive,
};
