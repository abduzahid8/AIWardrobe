import axios from 'axios';
import aiService from '../../src/services/aiService';

jest.mock('axios');
jest.mock('../../src/config/env', () => ({
  __esModule: true,
  default: {
    api: {
      url: 'https://test-api.example.com',
      alicevisionUrl: 'https://test-alicevision.example.com'
    }
  },
  Config: {
    api: {
      url: 'https://test-api.example.com',
      alicevisionUrl: 'https://test-alicevision.example.com'
    }
  }
}));

const mockedAxios = axios as jest.Mocked<typeof axios>;

beforeEach(() => {
  jest.clearAllMocks();
});

const getService = () => {
  return aiService;
};

describe('aiService', () => {
  describe('checkServerHealth', () => {
    it('returns healthy=true when server responds', async () => {
      mockedAxios.get = jest.fn().mockResolvedValue({ data: { status: 'ok' } });
      const service = getService();
      const result = await service.checkServerHealth();
      expect(result.healthy).toBe(true);
    });

    it('returns healthy=false when server is unreachable', async () => {
      mockedAxios.get = jest.fn().mockRejectedValue(new Error('ECONNREFUSED'));
      const service = getService();
      const result = await service.checkServerHealth();
      expect(result.healthy).toBe(false);
    });
  });

  describe('generateOutfitSuggestions — local fallback', () => {
    it('returns local suggestions when backend is unavailable', async () => {
      mockedAxios.post = jest.fn().mockRejectedValue(new Error('Network Error'));
      const service = getService();
      const suggestions = await service.generateOutfitSuggestions('date');
      expect(Array.isArray(suggestions)).toBe(true);
      expect(suggestions.length).toBeGreaterThan(0);
      expect(suggestions[0]).toHaveProperty('id');
      expect(suggestions[0]).toHaveProperty('description');
      expect(suggestions[0]).toHaveProperty('items');
    });

    it('returns local suggestions for unknown occasion (falls back to casual)', async () => {
      mockedAxios.post = jest.fn().mockRejectedValue(new Error('Network Error'));
      const service = getService();
      const suggestions = await service.generateOutfitSuggestions('unknown_occasion_xyz');
      expect(suggestions[0].occasion).toBe('Casual');
    });

    it('returns cached suggestions on repeated calls with same key', async () => {
      mockedAxios.post = jest.fn().mockResolvedValue({
        data: {
          success: true,
          outfits: [{ id: 'server-1', description: 'Server outfit', occasion: 'date', confidence: 0.9, items: [], stylingTips: [] }],
        },
      });
      const service = getService();
      await service.generateOutfitSuggestions('date');
      await service.generateOutfitSuggestions('date');
      // Second call should use cache — axios.post called only once
      expect(mockedAxios.post).toHaveBeenCalledTimes(1);
    });
  });

  describe('sendChatMessage — local fallback', () => {
    it('returns local response for date-related message when backend unavailable', async () => {
      mockedAxios.post = jest.fn().mockRejectedValue(new Error('Network Error'));
      const service = getService();
      const response = await service.sendChatMessage('What should I wear for a date?');
      expect(response.text).toBeTruthy();
      expect(Array.isArray(response.suggestions)).toBe(true);
    });

    it('returns local response for work-related message when backend unavailable', async () => {
      mockedAxios.post = jest.fn().mockRejectedValue(new Error('Network Error'));
      const service = getService();
      const response = await service.sendChatMessage('Help me dress for a job interview');
      expect(response.text).toContain('professional');
    });

    it('returns generic fallback for unrecognized message', async () => {
      mockedAxios.post = jest.fn().mockRejectedValue(new Error('Network Error'));
      const service = getService();
      const response = await service.sendChatMessage('random unrelated message');
      expect(response.text).toBeTruthy();
      expect(response.suggestions?.length).toBeGreaterThan(0);
    });
  });

  describe('getWeatherBasedOutfit', () => {
    it('includes cold weather tip when temp < 10', async () => {
      mockedAxios.post = jest.fn().mockRejectedValue(new Error('Network Error'));
      const service = getService();
      const outfit = await service.getWeatherBasedOutfit(5, 'clear');
      expect(outfit.stylingTips.some((t: string) => t.toLowerCase().includes('warm') || t.toLowerCase().includes('layer'))).toBe(true);
    });

    it('includes rain tip when condition includes rain', async () => {
      mockedAxios.post = jest.fn().mockRejectedValue(new Error('Network Error'));
      const service = getService();
      const outfit = await service.getWeatherBasedOutfit(18, 'light rain');
      expect(outfit.stylingTips.some((t: string) => t.toLowerCase().includes('umbrella') || t.toLowerCase().includes('waterproof'))).toBe(true);
    });
  });
});
