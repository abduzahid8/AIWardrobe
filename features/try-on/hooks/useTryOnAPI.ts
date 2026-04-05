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
    const [isMock, setIsMock] = useState(false);

    const handleTryOn = async (humanImage: string | null, clothImage: string | null, garmentType: string = 'upper_body') => {
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
            const replicateToken = process.env.EXPO_PUBLIC_REPLICATE_TOKEN;

            if (replicateToken) {
                // Bypass Edge Function and hit Replicate API directly
                const predictionRes = await fetch("https://api.replicate.com/v1/predictions", {
                    method: "POST",
                    headers: {
                        "Authorization": `Token ${replicateToken}`,
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({
                        version: "0513734a452173b8173e907e3a59d19a36266e55b48528559432bd21c7d7e985",
                        input: {
                            human_img: humanImage,
                            garm_img: clothImage,
                            garment_des: "clothing",
                            category: garmentType === 'lower_body' ? 'lower_body' : 'upper_body',
                            n_samples: 1,
                            seed: 42
                        }
                    })
                });

                if (!predictionRes.ok) {
                    const err = await predictionRes.text();
                    throw new Error("Replicate API Error: " + err);
                }

                const prediction = await predictionRes.json();
                let result = prediction;

                // Poll Replicate
                while (result.status !== "succeeded" && result.status !== "failed") {
                    await new Promise(r => setTimeout(r, 2000));
                    const pollRes = await fetch(result.urls.get, {
                        headers: { "Authorization": `Token ${replicateToken}` }
                    });
                    result = await pollRes.json();
                }

                if (result.status === "failed") {
                    throw new Error(`AI Processing Failed: ${result.error || 'Unknown error from Replicate'}`);
                }

                const outputImage = Array.isArray(result.output) ? result.output[0] : result.output;
                setResultImage(outputImage);
                setIsMock(false);

            } else {
                // Fallback to missing Edge Function / Mock call
                const { data, error } = await supabase.functions.invoke('try-on', {
                    body: {
                        person_image: humanImage,
                        garment_image: clothImage,
                        garment_type: garmentType,
                    },
                });

                if (error) throw error;

                if (data?.success && data?.resultImage) {
                    setResultImage(data.resultImage);
                    setIsMock(data.methodUsed === 'mock');
                    if (data.methodUsed === 'mock') {
                        Alert.alert(t('aiTryOn.demoTitle'), t('aiTryOn.demoMessage'));
                    }
                } else {
                    throw new Error(data?.error || 'Try-On failed');
                }
            }
        } catch (err: any) {
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
            Alert.alert(t('aiTryOn.errorTitle'), t('aiTryOn.saveFailed'));
        } finally {
            setSaving(false);
        }
    };

    return {
        loading,
        saving,
        resultImage,
        isMock,
        handleTryOn,
        handleSaveToWardrobe,
    };
}
