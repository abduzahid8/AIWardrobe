import React from 'react';
import {render,fireEvent} from '@testing-library/react-native';
import i18next from 'i18next';
import StyleQuizScreen from '../../screens/StyleQuizScreen';
import {COLOR_OPTIONS} from '../../features/style-quiz/data';
import en from '../../i18n/locales/en.json';
import ru from '../../i18n/locales/ru.json';
import uz from '../../i18n/locales/uz.json';
const mockI18n=i18next.createInstance();
const mockSetPreferences=jest.fn(),mockComplete=jest.fn();
jest.mock('react-i18next',()=>({useTranslation:()=>({t:mockI18n.t.bind(mockI18n)})}));
jest.mock('@react-navigation/native',()=>({useNavigation:()=>({navigate:jest.fn()})}));
jest.mock('../../store/stylePreferenceStore',()=>({useStylePreferenceStore:()=>({setPreferences:mockSetPreferences,setOnboardingStep:jest.fn(),completeOnboarding:mockComplete})}));
jest.mock('../../components/ui/ScaledText',()=>({ScaledText:jest.requireActual<typeof import('react-native')>('react-native').Text}));
jest.mock('react-native-safe-area-context',()=>({SafeAreaView:jest.requireActual<typeof import('react-native')>('react-native').View}));
jest.mock('react-native-reanimated',()=>{
 const transition={delay:()=>transition,duration:()=>transition,springify:()=>transition};
 return {__esModule:true,default:{View:jest.requireActual<typeof import('react-native')>('react-native').View,createAnimatedComponent:(component:unknown)=>component},FadeIn:transition,FadeInDown:transition,FadeInUp:transition,SlideInRight:transition};
});
beforeAll(async()=>{await mockI18n.init({lng:'en',fallbackLng:'en',resources:{en:{translation:en},ru:{translation:ru},uz:{translation:uz}},interpolation:{escapeValue:false}});});
beforeEach(()=>jest.clearAllMocks());
const label=(key:string)=>String(mockI18n.t(key));
function colors(){
 const view=render(<StyleQuizScreen/>);
 fireEvent.press(view.getByRole('button',{name:label('styleQuiz.continue')}));
 fireEvent.press(view.getByRole('radio',{name:label('styleQuiz.personality.minimalist')}));
 fireEvent.press(view.getByRole('button',{name:label('styleQuiz.continue')}));
 return view;
}
it.each(['en','ru','uz'])('all colors have localized names and checked states in %s',async language=>{
 await mockI18n.changeLanguage(language);const view=colors();
 expect(view.getAllByRole('checkbox')).toHaveLength(12);
 for(const option of COLOR_OPTIONS){
  const name=label(option.tKey);expect(name).not.toBe(option.tKey);
  expect(view.getByRole('checkbox',{name,checked:false})).toBeTruthy();
 }
 expect(view.getByRole('button',{name:label('common.back')})).toBeTruthy();
 expect(view.getByRole('button',{name:label('styleQuiz.continue')}).props.accessibilityState.disabled).toBe(true);
 const white=label('styleQuiz.colors.names.white');
 fireEvent.press(view.getByRole('checkbox',{name:white}));
 expect(view.getByRole('checkbox',{name:white,checked:true})).toBeTruthy();
 expect(view.getByRole('button',{name:label('styleQuiz.continue')}).props.accessibilityState.disabled).toBe(false);
 fireEvent.press(view.getByRole('checkbox',{name:white}));
 expect(view.getByRole('checkbox',{name:white,checked:false})).toBeTruthy();
 expect(view.getByRole('button',{name:label('styleQuiz.continue')}).props.accessibilityState.disabled).toBe(true);
});
test('selection semantics survive navigation and save stable preference IDs',async()=>{
 await mockI18n.changeLanguage('en');const view=colors();const next=()=>fireEvent.press(view.getByRole('button',{name:'Continue'}));
 fireEvent.press(view.getByRole('checkbox',{name:'Midnight blue'}));
 fireEvent.press(view.getByRole('button',{name:label('common.back')}));
 expect(view.getByRole('radio',{name:'Minimalist',checked:true})).toBeTruthy();next();
 expect(view.getByRole('checkbox',{name:'Midnight blue',checked:true})).toBeTruthy();next();
 fireEvent.press(view.getByRole('checkbox',{name:'Casual'}));next();
 const fit=label('styleQuiz.fit.balanced')+'. '+label('styleQuiz.fit.balancedDesc');
 fireEvent.press(view.getByRole('radio',{name:fit}));next();
 fireEvent.press(view.getByRole('checkbox',{name:label('styleQuiz.goals.organizeCloset')}));next();
 fireEvent.press(view.getByRole('button',{name:label('styleQuiz.complete.unlockPlan')}));
 expect(mockSetPreferences).toHaveBeenCalledWith(expect.objectContaining({favoriteColors:['#0A1931'],stylePersonality:'minimalist',primaryOccasions:['casual'],fitPreference:'balanced',styleGoals:['organize_closet']}));
 expect(mockComplete).toHaveBeenCalledTimes(1);
});
