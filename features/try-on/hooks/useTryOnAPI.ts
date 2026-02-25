/**
 * useTryOnAPI — handles the virtual try-on API call and save-to-wardrobe
 */

import { useState } from 'react';
import { Alert } from 'react-native';
import { useTranslation } from 'react-i18next';
import { useNavigation } from '@react-navigation/native';
import { supabase } from '../../../lib/supabase';
import useAuthStore from '../../../store/auth';

export function useTryOnAPI() {
    const { t } = useTranslation();
    const navigation = useNavigation();
    const { user } = useAuthStore();

    const [loading, setLoading] = useState(false);
    const [saving, setSaving] = useState(false);
    const [resultImage, setResultImage] = useState<string | null>(null);

    const handleTryOn = async (humanImage: string | null, clothImage: string | null) => {
        if (!humanImage || !clothImage) {
            Alert.alert(t('aiTryOn.errors.missingPhotos'), t('aiTryOn.errors.missingPhotos'));
            return;
        }
        if (!user) {
            Alert.alert('Error', t('aiTryOn.loginRequired'));
            return;
        }

        setLoading(true);
        setResultImage(null);

        try {
            const { data, error } = await supabase.functions.invoke('try-on', {
                body: {
                    person_image: humanImage,
                    garment_image: clothImage,
                    garment_type: 'upper_body',
                },
            });

            if (error) throw error;

            if (data?.success && data?.resultImage) {
                setResultImage(data.resultImage);
                if (data.methodUsed === 'mock') {
                    Alert.alert(t('aiTryOn.demoTitle'), t('aiTryOn.demoMessage'));
                }
            } else {
                throw new Error(data?.error || 'Try-On failed');
            }
        } catch (err: any) {
            console.error('Try-On Error:', err);
            Alert.alert(t('aiTryOn.errorTitle'), `${t('aiTryOn.errorMessage')} ${err?.message || ''}`);
        } finally {
            setLoading(false);
        }
    };

    const handleSaveToWardrobe = async () => {
        if (!resultImage || !user) return;
        setSaving(true);
        try {
            const { error } = await supabase.from('clothing_items').insert({
                user_id: user.id,
                type: 'AI Try-On Result',
                color: 'Mixed',
                style: 'Casual',
                description: 'Virtual try-on generated outfit',
                season: 'All Seasons',
                image_url: resultImage,
                category: 'outfit',
            });
            if (error) throw error;
            Alert.alert(t('aiTryOn.savedTitle'), t('aiTryOn.savedMessage'), [
                { text: t('aiTryOn.viewWardrobe'), onPress: () => (navigation as any).navigate('Profile') },
                { text: 'OK' },
            ]);
        } catch (err: any) {
            console.error('Save error:', err);
            Alert.alert(t('aiTryOn.errorTitle'), t('aiTryOn.saveFailed'));
        } finally {
            setSaving(false);
        }
    };

    return {
        loading,
        saving,
        resultImage,
        handleTryOn,
        handleSaveToWardrobe,
    };
}
