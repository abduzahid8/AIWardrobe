import * as cheerio from 'cheerio';
import axios from 'axios';

export const scrapeProduct = async (url) => {
  try {
    // 1. Скачиваем HTML страницы магазина
    // Добавляем User-Agent, чтобы магазин думал, что мы обычный браузер, а не бот
    const { data } = await axios.get(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
      }
    });

    // 2. Загружаем HTML в Cheerio
    const $ = cheerio.load(data);

    // 3. Ищем данные (обычно магазины хранят всё в Open Graph тегах)
    let title = $('meta[property="og:title"]').attr('content') || $('title').text();
    let image = $('meta[property="og:image"]').attr('content');
    
    // Если og:image нет, пробуем найти первую большую картинку
    if (!image) {
        image = $('img').first().attr('src');
    }

    // Очищаем название от лишнего мусора (пробелы по краям)
    title = title ? title.trim() : "Unknown Item";

    return {
      success: true,
      data: {
        title,
        image,
        url
      }
    };

  } catch (error) {
    console.error("Scraping Error:", error.message);
    return {
      success: false,
      error: "Could not parse this link. Try another one."
    };
  }
};