/**
 * qualityGate.ts — Image quality gate (Phase 4C).
 *
 * Pure metadata-driven assessment: the caller supplies measured signals
 * (brightness, blur score, garment frame share, occlusion flags). Never
 * forces retake for mildly degraded images — `degraded` still allows
 * low-confidence extraction with reasons attached.
 */
import type { QualityGateResult } from '../../types/visualOS';
import { isSupportedMime } from './visualAssets';

export interface QualitySignals {
  mimeType?: string | null;
  brightness01?: number | null;
  blurScore01?: number | null;
  garmentFrameShare01?: number | null;
  occluded?: boolean;
  multipleGarments?: boolean;
  mirrorObstruction?: boolean;
  croppedItem?: boolean;
  badAngle?: boolean;
}

export function assessImageQuality(signals: QualitySignals): QualityGateResult {
  const reasons: string[] = [];
  if (!isSupportedMime(signals.mimeType)) {
    reasons.push(`unsupported format ${signals.mimeType || 'unknown'}`);
    return { usability: 'unusable', reasons, recommendedCorrection: 'Use JPEG, PNG, WEBP or HEIC.', analyzedAt: new Date().toISOString() };
  }
  if (signals.brightness01 !== undefined && signals.brightness01 !== null && signals.brightness01 < 0.15)
    reasons.push('too dark');
  if (signals.blurScore01 !== undefined && signals.blurScore01 !== null && signals.blurScore01 > 0.7)
    reasons.push('too blurry');
  if (signals.garmentFrameShare01 !== undefined && signals.garmentFrameShare01 !== null && signals.garmentFrameShare01 < 0.12)
    reasons.push('garment too small in frame');
  if (signals.occluded) reasons.push('heavy occlusion');
  if (signals.mirrorObstruction) reasons.push('mirror obstruction');
  if (signals.badAngle) reasons.push('bad angle');
  if (signals.croppedItem) reasons.push('cropped item');
  if (signals.multipleGarments) reasons.push('multiple garments');

  const hardFails = ['too blurry', 'garment too small in frame', 'heavy occlusion', 'bad angle'].filter((r) => reasons.includes(r));
  const severeBlur = (signals.blurScore01 ?? 0) > 0.85;
  const usability = hardFails.length >= 2 || severeBlur || reasons.includes('too blurry') && reasons.includes('garment too small in frame')
    ? 'unusable'
    : reasons.length ? 'degraded' : 'usable';
  const correction = usability === 'usable'
    ? null
    : usability === 'unusable'
      ? 'Retake with the garment filling the frame in good light.'
      : 'Usable with reduced confidence; retake in better light for higher accuracy.';
  return { usability, reasons, recommendedCorrection: correction, analyzedAt: new Date().toISOString() };
}
