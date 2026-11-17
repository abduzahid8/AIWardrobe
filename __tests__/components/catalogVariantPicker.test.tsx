import i18n from 'i18next';
import {initReactI18next} from 'react-i18next';
import React from 'react';
import { render, fireEvent } from '@testing-library/react-native';
import { CatalogVariantPicker } from '../../components/CatalogVariantPicker';
import type { ShopCatalogItem } from '../../src/types/shop';
beforeAll(async()=>{await i18n.use(initReactI18next).init({lng:'en',fallbackLng:'en',resources:{en:{translation:{}}},interpolation:{escapeValue:false}});});
const variants: ShopCatalogItem[] = [
  {
    id: 'black-row',
    brand: 'Brand',
    name: 'Model Black',
    price: 100,
    currency: 'USD',
    imageUrl: 'black.jpg',
    garmentType: 'shoes',
  },
  {
    id: 'brown-row',
    brand: 'Brand',
    name: 'Model Brown',
    price: 120,
    currency: 'USD',
    imageUrl: 'brown.jpg',
    garmentType: 'shoes',
  },
];
test('accessible variant buttons show native price differences and emit the concrete row ID', () => {
  const select = jest.fn();
  const view = render(
    <CatalogVariantPicker variants={variants} selectedId="black-row" onSelect={select} />
  );
  expect(view.getByLabelText('black · 100 USD').props.accessibilityState.selected).toBe(true);
  fireEvent.press(view.getByLabelText('brown · 120 USD'));
  expect(select).toHaveBeenCalledWith('brown-row');
  view.rerender(
    <CatalogVariantPicker variants={variants} selectedId="brown-row" onSelect={select} />
  );
  expect(view.getByLabelText('brown · 120 USD').props.accessibilityState.selected).toBe(true);
});

test('generic-title variants remain distinguishable without invented colors',()=>{
 const generic=variants.map(item=>({...item,name:'Regular Fit Chino trousers'}));
 const select=jest.fn();
 const view=render(<CatalogVariantPicker variants={generic} selectedId={generic[0].id} onSelect={select}/>);
 fireEvent.press(view.getByLabelText('Regular Fit Chino trousers · Variant 2 · 120 USD'));
 expect(select).toHaveBeenCalledWith(generic[1].id);
});
