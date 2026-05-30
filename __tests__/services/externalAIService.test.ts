import { ExternalAIService } from '../../src/services/externalAIService';
import { supabase } from '../../src/lib/supabase';

jest.mock('../../src/lib/supabase', () => ({
  supabase: {
    functions: {
      invoke: jest.fn(),
    },
  },
}));

const mockedInvoke = supabase.functions.invoke as jest.Mock;

describe('ExternalAIService', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('uses the studio_photo operation and returns the cutout image', async () => {
    mockedInvoke.mockResolvedValue({
      data: {
        success: true,
        cutoutUrl: 'https://cdn.example.com/cutout.png',
        classification: {
          category: 'shirt',
          section: 'tops',
          confidence: 0.91,
          attributes: {
            style: 'casual',
            color: 'blue',
            material: 'cotton',
          },
        },
        description: 'Blue cotton shirt',
      },
      error: null,
    });

    const result = await ExternalAIService.processStudioPhoto('abc123');

    expect(mockedInvoke).toHaveBeenCalledWith('ai-process', {
      body: {
        image: 'abc123',
        operation: 'classify',
      },
    });
    expect(result.success).toBe(true);
    expect(result.imageUrl).toBe('data:image/jpeg;base64,abc123');
    expect(result.cutoutUrl).toBe('https://cdn.example.com/cutout.png');
    expect(result.steps).toEqual(['nvidia_classify']);
  });

  it('falls back to the original image when studio processing fails', async () => {
    mockedInvoke.mockResolvedValue({
      data: {
        success: false,
        error: 'processing failed',
      },
      error: null,
    });

    const result = await ExternalAIService.processStudioPhoto('abc123');

    expect(result.success).toBe(false);
    expect(result.imageUrl).toBe('data:image/jpeg;base64,abc123');
    expect(result.steps).toEqual(['error_fallback']);
  });
});
