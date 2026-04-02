/**
 * useMannequin — mannequin configuration, local asset management,
 * size presets, and base64 conversion for IDM-VTON API.
 *
 * Uses the project's built-in mannequin_front.png / mannequin_side.png
 * assets instead of remote URLs so the display is always reliable.
 * getMannequinBase64() converts the selected local asset to a base64
 * data URI accepted by the Replicate IDM-VTON model.
 */

import { useState } from 'react';
import { Asset } from 'expo-asset';
import * as FileSystem from 'expo-file-system/legacy';
import type { ShopCatalogItem } from '../types';

export type MannequinSize   = 'XS' | 'S' | 'M' | 'L' | 'XL';
export type MannequinGender = 'female' | 'male';
export type MannequinView   = 'front' | 'side';

// ── Local mannequin assets ─────────────────────────────────────────────────
const MANNEQUIN_FRONT = require('../../../assets/images/mannequin_front.png');
const MANNEQUIN_SIDE  = require('../../../assets/images/mannequin_side.png');

export interface MannequinPreset {
    size:        MannequinSize;
    label:       string;
    heightCm:    string;
    bustCm:      string;
    waistCm:     string;
    hipsCm:      string;
    frontAsset:  number;
    sideAsset:   number;
}

// ── Convert a require() asset to a base64 data URI for Replicate API ───────
export async function getMannequinBase64(assetModule: number): Promise<string> {
    const asset = Asset.fromModule(assetModule);
    await asset.downloadAsync();
    if (!asset.localUri) throw new Error('Mannequin asset not available');
    const base64 = await FileSystem.readAsStringAsync(asset.localUri, {
        encoding: 'base64' as any,
    });
    const mimeType = asset.type === 'jpg' ? 'image/jpeg' : 'image/png';
    return `data:${mimeType};base64,${base64}`;
}

// ── Size presets — EU/ISO 8559-1:2017 measurements ────────────────────────
export const FEMALE_PRESETS: MannequinPreset[] = [
    { size: 'XS', label: 'XS', heightCm: '162', bustCm: '80',  waistCm: '60', hipsCm: '86',  frontAsset: MANNEQUIN_FRONT, sideAsset: MANNEQUIN_SIDE },
    { size: 'S',  label: 'S',  heightCm: '165', bustCm: '84',  waistCm: '64', hipsCm: '90',  frontAsset: MANNEQUIN_FRONT, sideAsset: MANNEQUIN_SIDE },
    { size: 'M',  label: 'M',  heightCm: '168', bustCm: '88',  waistCm: '68', hipsCm: '94',  frontAsset: MANNEQUIN_FRONT, sideAsset: MANNEQUIN_SIDE },
    { size: 'L',  label: 'L',  heightCm: '170', bustCm: '96',  waistCm: '76', hipsCm: '102', frontAsset: MANNEQUIN_FRONT, sideAsset: MANNEQUIN_SIDE },
    { size: 'XL', label: 'XL', heightCm: '172', bustCm: '104', waistCm: '84', hipsCm: '110', frontAsset: MANNEQUIN_FRONT, sideAsset: MANNEQUIN_SIDE },
];

export const MALE_PRESETS: MannequinPreset[] = [
    { size: 'XS', label: 'XS', heightCm: '168', bustCm: '84',  waistCm: '74', hipsCm: '88',  frontAsset: MANNEQUIN_FRONT, sideAsset: MANNEQUIN_SIDE },
    { size: 'S',  label: 'S',  heightCm: '173', bustCm: '92',  waistCm: '78', hipsCm: '94',  frontAsset: MANNEQUIN_FRONT, sideAsset: MANNEQUIN_SIDE },
    { size: 'M',  label: 'M',  heightCm: '178', bustCm: '100', waistCm: '84', hipsCm: '100', frontAsset: MANNEQUIN_FRONT, sideAsset: MANNEQUIN_SIDE },
    { size: 'L',  label: 'L',  heightCm: '182', bustCm: '108', waistCm: '90', hipsCm: '106', frontAsset: MANNEQUIN_FRONT, sideAsset: MANNEQUIN_SIDE },
    { size: 'XL', label: 'XL', heightCm: '184', bustCm: '116', waistCm: '98', hipsCm: '112', frontAsset: MANNEQUIN_FRONT, sideAsset: MANNEQUIN_SIDE },
];

const PRESET_MAP: Record<MannequinGender, MannequinPreset[]> = {
    female: FEMALE_PRESETS,
    male:   MALE_PRESETS,
};

// ── Hook ──────────────────────────────────────────────────────────────────
export function useMannequin() {
    const [mannequinSize,    setMannequinSize]    = useState<MannequinSize>('M');
    const [mannequinGender,  setMannequinGender]  = useState<MannequinGender>('female');
    const [mannequinView,    setMannequinView]    = useState<MannequinView>('front');
    const [selectedShopItem, setSelectedShopItem] = useState<ShopCatalogItem | null>(null);
    const [shopFilter,       setShopFilter]       = useState<string>('all');

    const presets       = PRESET_MAP[mannequinGender];
    const currentPreset = presets.find((p) => p.size === mannequinSize) ?? presets[2];
    const currentAsset  = mannequinView === 'front'
        ? currentPreset.frontAsset
        : currentPreset.sideAsset;

    return {
        mannequinSize,
        setMannequinSize,
        mannequinGender,
        setMannequinGender,
        mannequinView,
        setMannequinView,
        selectedShopItem,
        setSelectedShopItem,
        shopFilter,
        setShopFilter,
        currentPreset,
        currentAsset,
        presets,
    };
}
