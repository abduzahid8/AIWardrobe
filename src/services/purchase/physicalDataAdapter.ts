import type {
  GarmentPhysicalProfile,
  GarmentCategory,
  FitIntent,
  Stretch,
} from '../../types/garment';
export interface MerchantMeasurementChart {
  verified: boolean;
  sourceUrl: string;
  productReference: string;
  rows: Array<{
    sizeLabel: string;
    category: GarmentCategory;
    fitIntent?: FitIntent;
    stretch?: Stretch;
    material?: string[];
    measurements: Record<string, { value: number; unit: 'cm' | 'in' }>;
  }>;
}
/** Exact, explicitly verified units only. Unknown metrics/units never become
 * inferred centimeters or get mixed with category/cut priors. */
export function adaptMerchantMeasurements(
  chart: MerchantMeasurementChart,
  product: { id: string; reference: string; url: string }
): GarmentPhysicalProfile[] {
  if (
    !chart.verified ||
    !chart.productReference ||
    chart.productReference !== product.reference ||
    chart.sourceUrl !== product.url
  )
    return [];
  const allowed = new Set([
    'chest',
    'waist',
    'hips',
    'shoulderWidth',
    'sleeveLength',
    'bodyLength',
    'inseam',
    'rise',
    'thigh',
    'hemOpening',
    'shoeLength',
    'shoeWidth',
  ]);
  return chart.rows.slice(0, 30).flatMap((row) => {
    if (!row.sizeLabel || row.sizeLabel.length > 24) return [];
    const measurements: GarmentPhysicalProfile['measurements'] = {};
    for (const [zone, value] of Object.entries(row.measurements || {}).slice(0, 16)) {
      if (
        !allowed.has(zone) ||
        !value ||
        !['cm', 'in'].includes(value.unit) ||
        !Number.isFinite(value.value) ||
        value.value <= 0
      )
        continue;
      const cm = value.unit === 'in' ? value.value * 2.54 : value.value;
      if (cm > 400 || (zone.startsWith('shoe') && cm > 60)) continue;
      (measurements as Record<string, unknown>)[zone] = {
        valueCm: Math.round(cm * 100) / 100,
        unitSource: value.unit === 'in' ? 'inch' : 'cm',
        confidence: 'high',
      };
    }
    if (!Object.keys(measurements).length) return [];
    return [
      {
        id: `chart:${product.id}:${row.sizeLabel}`,
        garmentId: product.id,
        sizeLabel: row.sizeLabel,
        category: row.category,
        fitIntent: row.fitIntent,
        stretch: row.stretch,
        material: row.material?.slice(0, 10),
        measurements,
        source: 'manufacturer_size_chart' as const,
      },
    ];
  });
}
