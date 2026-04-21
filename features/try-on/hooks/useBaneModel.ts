/**
 * useBaneModel — loads bane_mannequin.glb from the local asset bundle
 * and returns a base64 data URI that can be passed directly to
 * generate3Dhtml(). GLTFLoader supports data URIs natively.
 */
import { useState, useEffect } from 'react';
import { Asset } from 'expo-asset';
import * as FileSystem from 'expo-file-system/legacy';

const BANE_GLB = require('../../../assets/models/bane_mannequin.glb');

export interface BaneModelState {
    modelDataUri: string | null;
    modelLoading: boolean;
    modelError: string | null;
}

export function useBaneModel(): BaneModelState {
    const [modelDataUri, setModelDataUri] = useState<string | null>(null);
    const [modelLoading, setModelLoading] = useState(true);
    const [modelError, setModelError] = useState<string | null>(null);

    useEffect(() => {
        let cancelled = false;
        (async () => {
            try {
                const asset = Asset.fromModule(BANE_GLB);
                await asset.downloadAsync();
                if (!asset.localUri) throw new Error('GLB asset localUri is null');
                const base64 = await FileSystem.readAsStringAsync(asset.localUri, {
                    encoding: 'base64' as any,
                });
                if (!cancelled) {
                    setModelDataUri(`data:model/gltf-binary;base64,${base64}`);
                }
            } catch (e: any) {
                if (!cancelled) {
                    setModelError(e?.message || 'Failed to load 3D model');
                }
            } finally {
                if (!cancelled) {
                    setModelLoading(false);
                }
            }
        })();
        return () => { cancelled = true; };
    }, []);

    return { modelDataUri, modelLoading, modelError };
}
