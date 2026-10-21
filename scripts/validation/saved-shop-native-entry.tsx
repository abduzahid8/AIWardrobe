/** Explicit-entry native audit: synthetic identities, real AsyncStorage, no cloud clients. */
import React, { useEffect, useState } from 'react';
import { Text, Platform } from 'react-native';
import { registerRootComponent } from 'expo';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as FileSystem from 'expo-file-system/legacy';
import store, { SAVED_SHOP_STORAGE_KEY, flushSavedShopPersistence } from '../../store/savedShopStore';
import { authEvents } from '../../store/authEvents';
import type { ShopCatalogItem } from '../../src/types/shop';
const phaseKey = 'readiness_saved_shop_phase';
const ownerA = '11111111-1111-4111-8111-111111111111';
const ownerB = '22222222-2222-4222-8222-222222222222';
const item = (id: string) => ({ id } as ShopCatalogItem);
const ids = () => store.getState().items.map(value => value.id);
function equal(actual: unknown, expected: unknown, label: string) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) throw new Error(label);
}
let started = false;
function Audit() {
  const [status, setStatus] = useState('Native saved-items audit running');
  useEffect(() => {
    if (started) return;
    started = true;
    void (async () => {
      let phase = 0;
      const report: {phase:number;platform:string;checks:string[];passed:boolean;failure?:string} = {phase,platform:Platform.OS,checks:[],passed:false};
      try {
        phase = Number(await AsyncStorage.getItem(phaseKey) || '1'); report.phase = phase;
        equal(authEvents.getUserId(), null, 'Fresh process starts without identity');
        await store.getState().hydrate(); equal(ids(), [], 'Unknown identity cannot hydrate');
        report.checks.push('fresh process has no identity or visible likes');
        if (phase === 1) {
          authEvents.emitLogin(ownerA);
          await store.getState().setItems([item('a-existing')]);
          await Promise.all([store.getState().toggleSave(item('a-first')),store.getState().toggleSave(item('a-second'))]);
          equal(ids(),['a-existing','a-first','a-second'],'Concurrent native saves merge');
          const disk=JSON.parse((await AsyncStorage.getItem(SAVED_SHOP_STORAGE_KEY))!);
          equal(disk.ownerId,ownerA,'Owner A tag');
          equal(disk.items.map((value:ShopCatalogItem)=>value.id),ids(),'Native disk matches confirmed list');
          report.checks.push('owner A tagged native disk matches three confirmed likes');
        } else if (phase === 2) {
          const disk=JSON.parse((await AsyncStorage.getItem(SAVED_SHOP_STORAGE_KEY))!);
          equal(disk.ownerId,ownerA,'Owner A record survives force-stop');
          authEvents.emitLogin(ownerB); await store.getState().hydrate();
          equal(ids(),[],'Owner B cannot read leftover owner A record');
          report.checks.push('owner A disk survives restart and owner B rejects it');
          await store.getState().setItems([item('b-existing')]);
          await store.getState().toggleSave(item('b-first'));
          equal(ids(),['b-existing','b-first'],'Owner B independent writes');
          report.checks.push('owner B writes confirmed independent likes');
        } else if (phase === 3) {
          authEvents.emitLogin(ownerB);
          await store.getState().toggleSave(item('b-second'));
          equal(ids(),['b-existing','b-first','b-second'],'First mutation preserves same-owner disk after restart');
          await store.getState().removeSave('b-existing');
          equal(ids(),['b-first','b-second'],'Removal preserves other likes');
          const disk=JSON.parse((await AsyncStorage.getItem(SAVED_SHOP_STORAGE_KEY))!);
          equal(disk.items.map((value:ShopCatalogItem)=>value.id),ids(),'Removal persists');
          report.checks.push('same-owner restart and first mutation preserve earlier likes');
          authEvents.emitLogout(); equal(ids(),[],'Logout clears memory synchronously');
          equal(await store.getState().toggleSave(item('anonymous')),false,'Anonymous toggle denied');
          await flushSavedShopPersistence();
          await AsyncStorage.multiRemove([SAVED_SHOP_STORAGE_KEY,phaseKey]);
          equal(await AsyncStorage.getItem(SAVED_SHOP_STORAGE_KEY),null,'Fixture likes removed');
          equal(await AsyncStorage.getItem(phaseKey),null,'Fixture phase removed');
          report.checks.push('logout, anonymous action denial and fixture key cleanup');
        } else throw new Error('Unexpected audit phase');
        if (phase < 3) await AsyncStorage.setItem(phaseKey,String(phase+1));
        report.passed=true;
      } catch(error) {report.failure=error instanceof Error?error.message:'Unknown failure';}
      await FileSystem.writeAsStringAsync(FileSystem.documentDirectory+`saved-shop-native-phase-${phase}.json`,JSON.stringify(report));
      console.warn('SAVED_SHOP_NATIVE_AUDIT '+JSON.stringify(report));
      setStatus(`Native saved-items phase ${phase}: ${report.passed?'passed':'failed'}`);
    })();
  }, []);
  return <Text style={{padding:40,paddingTop:80}}>{status}</Text>;
}
registerRootComponent(Audit);
