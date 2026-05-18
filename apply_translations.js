// Replace placeholder [RU] and [UZ] values with proper translations
const fs = require('fs');
const path = require('path');

const ruPath = path.join(__dirname, 'i18n', 'locales', 'ru.json');
const uzPath = path.join(__dirname, 'i18n', 'locales', 'uz.json');

// Helper to recursively find and replace [RU] and [UZ] placeholders
function replacePlaceholders(obj, lang) {
  if (Array.isArray(obj)) {
    return obj.map(item => replacePlaceholders(item, lang));
  }
  if (obj && typeof obj === 'object') {
    const result = {};
    for (const [k, v] of Object.entries(obj)) {
      result[k] = replacePlaceholders(v, lang);
    }
    return result;
  }
  if (typeof obj === 'string' && obj.startsWith(`[${lang}]`)) {
    return obj.slice(5).trim() || obj;
  }
  return obj;
}

// RU translations map
const ruTranslations = {
  'profile.analyticsSharing': 'Передача аналитики',
  'profile.analyticsOn': 'Вкл',
  'profile.analyticsOff': 'Выкл',
  'tripPlanner.subtitle': 'Отслеживайте свой модный путь',
  'tripPlanner.activeGoals': 'Активные цели',
  'tripPlanner.completed': 'Завершено',
  'tripPlanner.challenges': 'Челленджи',
  'tripPlanner.yourStyleGoals': 'Ваши цели стиля',
  'tripPlanner.weeklyChallenges': 'Еженедельные челленджи',
  'tripPlanner.alreadyActive': 'Уже активно',
  'tripPlanner.goalAlreadyProgress': 'Эта цель уже в процессе!',
  'tripPlanner.goalAchieved': '🎉 Цель достигнута!',
  'tripPlanner.congratulationsCompleted': 'Поздравляем! Вы завершили',
  'tripPlanner.challengeComplete': '🏆 Челлендж завершен!',
  'tripPlanner.amazingCompleted': 'Превосходно! Вы завершили',
  'tripPlanner.completedText': 'Завершено!',
  'tripPlanner.startGoal': 'Начать цель',
  'tripPlanner.logToday': 'Записать сегодня',
  'tripPlanner.challengeCompleteText': 'Челлендж завершен!',
  'tripPlanner.acceptChallenge': 'Принять челлендж',
  'tripPlanner.available[0].title': 'Создать капсульный гардероб',
  'tripPlanner.available[0].unit': 'универсальные вещи',
  'tripPlanner.available[0].description': 'Подберите 30 универсальных вещей, которые сочетаются друг с другом',
  'tripPlanner.available[1].title': 'Покупать более устойчиво',
  'tripPlanner.available[1].unit': 'устойчивые покупки',
  'tripPlanner.available[1].description': 'Сделайте 10 осознанных модных выборов в этом месяце',
  'tripPlanner.available[2].title': 'Добавить больше цвета',
  'tripPlanner.available[2].unit': 'цветные вещи',
  'tripPlanner.available[2].description': 'Выйдите из зоны комфорта с 5 цветными вещами',
  'tripPlanner.available[3].title': 'Принять минимализм',
  'tripPlanner.available[3].unit': 'убранные вещи',
  'tripPlanner.available[3].description': 'Уберите 20 вещей, которые вы больше не носите',
  'tripPlanner.available[4].title': 'Поднять рабочий стиль',
  'tripPlanner.available[4].unit': 'рабочие образы',
  'tripPlanner.available[4].description': 'Создайте 7 элегантных рабочих комбинаций образов',
  'tripPlanner.available[5].title': 'Планировать полные образы',
  'tripPlanner.available[5].unit': 'запланированные образы',
  'tripPlanner.available[5].description': 'Запланируйте 14 полных образов на следующие 2 недели',
  'tripPlanner.weeklyChallengesList[0].title': 'Неделя без повторов',
  'tripPlanner.weeklyChallengesList[0].description': 'Носите разные образы каждый день на этой неделе',
  'tripPlanner.weeklyChallengesList[1].title': 'Монохромный понедельник',
  'tripPlanner.weeklyChallengesList[1].description': 'Создайте одноцветный образ в понедельник',
  'tripPlanner.weeklyChallengesList[2].title': 'Фокус на аксессуарах',
  'tripPlanner.weeklyChallengesList[2].description': 'Добавьте новый аксессуар к каждому образу на этой неделе',
  'tripPlanner.weeklyChallengesList[3].title': 'Переоткрытие гардероба',
  'tripPlanner.weeklyChallengesList[3].description': 'Носите 3 вещи, которые вы не носили месяцами',
  'weeklyInsights.utilization60': 'Хорошо! Вы используете свой гардероб достаточно эффективно.',
  'weeklyInsights.utilization40': 'Средне. Некоторым вещам не помешало бы больше внимания.',
  'weeklyInsights.utilization20': 'Низко. Подумайте о том, чтобы отдать или обменять неиспользуемые вещи.',
  'signUp.invalidGender': 'Неверный пол',
  'signUp.genderOptions': 'Пожалуйста, выберите корректный вариант пола',
  'signUp.preferNotToSay': 'Предпочитаю не указывать',
  'signUp.skip': 'Пропустить',
  'signUp.profileImageUrlOptional': 'URL изображения профиля (необязательно)',
  'signUp.creatingAccount': 'Создание аккаунта...',
  'admin.inspo.addCapsule': 'Добавить капсулу',
  'admin.inspo.editCapsule': 'Редактировать капсулу',
  'admin.inspo.linkUrl': 'URL ссылки',
  'admin.inspo.alreadyInShop': 'Товар уже в каталоге магазина',
  'admin.inspo.photoUpdateNoAccess': 'Обновление заблокировано: нет прав на изменение капсулы',
  'admin.garmentTypes.outerwear': 'Верхняя одежда',
  'admin.garmentTypes.accessories': 'Аксессуары',
  'admin.guide.failedToSave': 'Не удалось сохранить содержание гида',
};

// UZ translations map
const uzTranslations = {
  'profile.analyticsSharing': 'Analitika ulashish',
  'profile.analyticsOn': 'Yoq',
  'profile.analyticsOff': 'O\'ch',
  'tripPlanner.subtitle': 'Moda sayohatingizni kuzating',
  'tripPlanner.activeGoals': 'Faol maqsadlar',
  'tripPlanner.completed': 'Tugatildi',
  'tripPlanner.challenges': 'Sinovlar',
  'tripPlanner.yourStyleGoals': 'Sizning uslub maqsadlaringiz',
  'tripPlanner.weeklyChallenges': 'Haftalik sinovlar',
  'tripPlanner.alreadyActive': 'Allaqachon faol',
  'tripPlanner.goalAlreadyProgress': 'Bu maqsad allaqachon jarayonda!',
  'tripPlanner.goalAchieved': '🎉 Maqsad erishildi!',
  'tripPlanner.congratulationsCompleted': 'Tabriklaymiz! Siz tugatdingiz',
  'tripPlanner.challengeComplete': '🏆 Sinov tugadi!',
  'tripPlanner.amazingCompleted': 'Ajoyib! Siz tugatdingiz',
  'tripPlanner.completedText': 'Tugatildi!',
  'tripPlanner.startGoal': 'Maqsadni boshlash',
  'tripPlanner.logToday': 'Bugun yozish',
  'tripPlanner.challengeCompleteText': 'Sinov tugadi!',
  'tripPlanner.acceptChallenge': 'Sinovni qabul qilish',
  'tripPlanner.available[0].title': 'Kapsula shkaf yaratish',
  'tripPlanner.available[0].unit': 'ko\'p qirrali narsalar',
  'tripPlanner.available[0].description': 'Bir-biri bilan mos keladigan 30 ko\'p qirrali narsani tanlang',
  'tripPlanner.available[1].title': 'Barqarorroq xarid qiling',
  'tripPlanner.available[1].unit': 'barqaror xaridlar',
  'tripPlanner.available[1].description': 'Bu oy 10 ta ongli moda tanlovini qiling',
  'tripPlanner.available[2].title': 'Ko\'proq rang qo\'shing',
  'tripPlanner.available[2].unit': 'rangli narsalar',
  'tripPlanner.available[2].description': '5 ta rangli narsa bilan qulay zonadan chiqing',
  'tripPlanner.available[3].title': 'Minimalizmni qabul qiling',
  'tripPlanner.available[3].unit': 'o\'chirilgan narsalar',
  'tripPlanner.available[3].description': 'Kiymaydigan 20 ta narsani o\'chiring',
  'tripPlanner.available[4].title': 'Ish uslubini ko\'taring',
  'tripPlanner.available[4].unit': 'ish liboslari',
  'tripPlanner.available[4].description': '7 ta sayqallangan ish libosi kombinatsiyasini yarating',
  'tripPlanner.available[5].title': 'To\'liq liboslarni rejalashtiring',
  'tripPlanner.available[5].unit': 'rejalashtirilgan liboslar',
  'tripPlanner.available[5].description': 'Keyingi 2 hafta uchun 14 ta to\'liq libosni rejalashtiring',
  'tripPlanner.weeklyChallengesList[0].title': 'Takrorlanmas hafta',
  'tripPlanner.weeklyChallengesList[0].description': 'Bu hafta har kuni boshqa libos kiying',
  'tripPlanner.weeklyChallengesList[1].title': 'Monokrom dushanba',
  'tripPlanner.weeklyChallengesList[1].description': 'Dushanba bir rangli libos yarating',
  'tripPlanner.weeklyChallengesList[2].title': 'Aksessuar fokus',
  'tripPlanner.weeklyChallengesList[2].description': 'Bu hafta har bir libosga yangi aksessuar qo\'shing',
  'tripPlanner.weeklyChallengesList[3].title': 'Shkafni qayta kashf qilish',
  'tripPlanner.weeklyChallengesList[3].description': 'Oylardan kiyilmagan 3 ta narsani kiying',
  'weeklyInsights.utilization60': 'Yaxshi! Shkafingizdan yaxshi foydalanmoqdasiz.',
  'weeklyInsights.utilization40': 'O\'rtacha. Ba\'zi narsalarga ko\'proq e\'tibor kerak.',
  'weeklyInsights.utilization20': 'Past. Ishlatilmaydigan narsalarni sovg\'a qilish yoki almashtirishni ko\'rib chiqing.',
  'admin.inspo.addCapsule': 'Kapsula qo\'shish',
  'admin.inspo.editCapsule': 'Kapsulani tahrirlash',
  'admin.inspo.linkUrl': 'Havola URL',
  'admin.inspo.alreadyInShop': 'Mahsulot allaqachon do\'kon katalogida',
  'admin.inspo.photoUpdateNoAccess': 'Yangilash bloklandi: kapsulani o\'zgartirish uchun ruxsat yo\'q',
  'admin.garmentTypes.outerwear': 'Tashqi kiyim',
  'admin.garmentTypes.accessories': 'Aksessuarlar',
  'admin.guide.failedToSave': 'Qo\'llanma mazmunini saqlab bo\'lmadi',
  'manage.deleteTitle': 'Mahsulotni o\'chirish',
  'manage.deleteConfirm': 'Do\'kondan \"{{name}}\" ni o\'chirish?',
  'manage.editItem': 'Mahsulotni tahrirlash',
  'manage.empty': 'Mahsulotlar topilmadi',
  'manage.videoSavedGallery': 'Video galereyangizga saqlandi!',
  'manage.failedPickGallery': 'Galereyadan tanlab bo\'lmadi',
};

function setDeep(obj, keyPath, value) {
  const parts = keyPath.split('.');
  let current = obj;
  for (let i = 0; i < parts.length; i++) {
    const p = parts[i];
    if (i === parts.length - 1) {
      // Handle array index notation like "available[0]"
      const arrMatch = p.match(/^(.+)\[(\d+)\]$/);
      if (arrMatch) {
        const arrName = arrMatch[1];
        const idx = parseInt(arrMatch[2]);
        current[arrName] = current[arrName] || [];
        if (Array.isArray(value) || typeof value === 'object') {
          current[arrName][idx] = { ...current[arrName][idx], ...value };
        } else {
          current[arrName][idx] = value;
        }
      } else {
        current[p] = value;
      }
    } else {
      const arrMatch = p.match(/^(.+)\[(\d+)\]$/);
      if (arrMatch) {
        const arrName = arrMatch[1];
        const idx = parseInt(arrMatch[2]);
        current[arrName] = current[arrName] || [];
        current[arrName][idx] = current[arrName][idx] || {};
        current = current[arrName][idx];
      } else {
        current[p] = current[p] || {};
        current = current[p];
      }
    }
  }
}

// Apply RU translations
const ru = JSON.parse(fs.readFileSync(ruPath, 'utf8'));
for (const [key, value] of Object.entries(ruTranslations)) {
  setDeep(ru, key, value);
}
fs.writeFileSync(ruPath, JSON.stringify(ru, null, 2) + '\n');

// Apply UZ translations
const uz = JSON.parse(fs.readFileSync(uzPath, 'utf8'));
for (const [key, value] of Object.entries(uzTranslations)) {
  setDeep(uz, key, value);
}
fs.writeFileSync(uzPath, JSON.stringify(uz, null, 2) + '\n');

console.log('Translations applied successfully!');
console.log(`Applied ${Object.keys(ruTranslations).length} RU translations`);
console.log(`Applied ${Object.keys(uzTranslations).length} UZ translations`);