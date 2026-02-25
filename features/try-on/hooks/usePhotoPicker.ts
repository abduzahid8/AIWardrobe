/**
 * usePhotoPicker — encapsulates camera/gallery photo picking logic
 */

import { useState } from 'react';
import { Alert, ActionSheetIOS, Platform } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { useTranslation } from 'react-i18next';

export function usePhotoPicker() {
    const { t } = useTranslation();
    const [humanImage, setHumanImage] = useState<string | null>(null);
    const [clothImage, setClothImage] = useState<string | null>(null);

    const pickPhoto = async (
        source: 'camera' | 'library',
        aspect: [number, number],
        onResult: (base64DataUri: string) => void
    ) => {
        if (source === 'camera') {
            const { status } = await ImagePicker.requestCameraPermissionsAsync();
            if (status !== 'granted') {
                Alert.alert(t('aiTryOn.permissionTitle'), t('aiTryOn.cameraPermission'));
                return;
            }
            const result = await ImagePicker.launchCameraAsync({
                mediaTypes: ['images'],
                allowsEditing: true,
                aspect,
                quality: 0.7,
                base64: true,
            });
            if (!result.canceled && result.assets?.[0]?.base64) {
                onResult(`data:image/jpeg;base64,${result.assets[0].base64}`);
            }
            return;
        }

        const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
        if (status !== 'granted') {
            Alert.alert(t('aiTryOn.permissionTitle'), t('aiTryOn.photoPermission'));
            return;
        }
        const result = await ImagePicker.launchImageLibraryAsync({
            mediaTypes: ['images'],
            allowsEditing: true,
            aspect,
            quality: 0.7,
            base64: true,
        });
        if (!result.canceled && result.assets?.[0]?.base64) {
            onResult(`data:image/jpeg;base64,${result.assets[0].base64}`);
        }
    };

    const pickFullLengthPhoto = (source: 'camera' | 'library', onSuccess?: () => void) =>
        pickPhoto(source, [3, 4], (uri) => {
            setHumanImage(uri);
            if (onSuccess) onSuccess();
        });

    const pickGarmentPhoto = (source: 'camera' | 'library') =>
        pickPhoto(source, [1, 1], (uri) => setClothImage(uri));

    const showPhotoOptions = (
        title: string,
        onCamera: () => void,
        onGallery: () => void
    ) => {
        if (Platform.OS === 'ios') {
            ActionSheetIOS.showActionSheetWithOptions(
                {
                    options: [t('aiTryOn.cancel'), t('aiTryOn.takePhoto'), t('aiTryOn.choosePhoto')],
                    cancelButtonIndex: 0,
                },
                (buttonIndex) => {
                    if (buttonIndex === 1) onCamera();
                    if (buttonIndex === 2) onGallery();
                }
            );
        } else {
            Alert.alert(title, undefined, [
                { text: t('aiTryOn.cancel'), style: 'cancel' },
                { text: t('aiTryOn.takePhoto'), onPress: onCamera },
                { text: t('aiTryOn.choosePhoto'), onPress: onGallery },
            ]);
        }
    };

    const showFullLengthPhotoOptions = (onSuccess?: () => void) =>
        showPhotoOptions(
            t('aiTryOn.fullLengthPhoto'),
            () => pickFullLengthPhoto('camera', onSuccess),
            () => pickFullLengthPhoto('library', onSuccess)
        );

    const showGarmentPhotoOptions = () =>
        showPhotoOptions(
            t('aiTryOn.addGarmentPhoto') || 'Add garment photo',
            () => pickGarmentPhoto('camera'),
            () => pickGarmentPhoto('library')
        );

    return {
        humanImage,
        setHumanImage,
        clothImage,
        setClothImage,
        pickFullLengthPhoto,
        pickGarmentPhoto,
        showFullLengthPhotoOptions,
        showGarmentPhotoOptions,
    };
}
