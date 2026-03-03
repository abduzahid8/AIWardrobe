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
import useWardrobeStore from '../../store/wardrobeStore';

/**
 * Show confirmation dialog before removing a clothing item.
 */
export const confirmRemoveItem = (itemId: string, itemName?: string): void => {
    const label = itemName || 'this item';
    Alert.alert(
        'Remove Item',
        `Are you sure you want to remove ${label} from your wardrobe? This action cannot be undone.`,
        [
            { text: 'Cancel', style: 'cancel' },
            {
                text: 'Remove',
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
    const label = outfitName || 'this outfit';
    Alert.alert(
        'Delete Outfit',
        `Are you sure you want to delete ${label}?`,
        [
            { text: 'Cancel', style: 'cancel' },
            {
                text: 'Delete',
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
    confirmLabel = 'Delete',
): void => {
    Alert.alert(title, message, [
        { text: 'Cancel', style: 'cancel' },
        { text: confirmLabel, style: 'destructive', onPress: onConfirm },
    ]);
};

export default {
    confirmRemoveItem,
    confirmDeleteOutfit,
    confirmDestructive,
};
