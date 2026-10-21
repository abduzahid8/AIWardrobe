import { confirmedImageReplacements,replaceGeneratedOutfitImages,replaceSavedOutfitImages } from '../../src/utils/confirmedOutfitMedia';
const before=[{id:'shirt',image:'file:///photo.jpg',imageUrl:'file:///thumb.jpg'}];
const after=[{id:'shirt',image:'https://storage.test/photo.jpg',imageUrl:'https://storage.test/thumb.jpg'}];
test('confirmed references replace preview, generated pieces, saved snapshots and daily suggestion shape',()=>{
 const replacements=confirmedImageReplacements(before,after);
 const generated={mainImage:before[0].image,items:[{...before[0],name:'shirt'}, {id:'shop',image:'https://shop.test/photo.jpg'}],description:'keep'};
 const normalized=replaceGeneratedOutfitImages(generated,replacements);
 expect(normalized).toEqual({...generated,mainImage:after[0].image,items:[{...after[0],name:'shirt'},generated.items[1]]});
 const saved={id:'look',previewImageUrl:before[0].image,snapshotItems:[{id:'shirt',image:before[0].image},{id:'shop',image:'https://shop.test/photo.jpg'}]};
 expect(replaceSavedOutfitImages(saved,replacements)).toMatchObject({id:'look',previewImageUrl:after[0].image,snapshotItems:[{id:'shirt',image:after[0].image},saved.snapshotItems[1]]});
 expect(generated.items[0].image).toBe(before[0].image);expect(saved.snapshotItems[0].image).toBe(before[0].image);
});
test('new photo selections and unrelated sources survive late normalization',()=>{
 const replacements=confirmedImageReplacements(before,after);
 const updated={mainImage:'file:///new.jpg',items:[{id:'shirt',image:'file:///new.jpg'}]};
 expect(replaceGeneratedOutfitImages(updated,replacements)).toEqual(updated);
});
test('mismatched or incomplete confirmation cannot reassign photo references',()=>{
 expect(()=>confirmedImageReplacements(before,[])).toThrow('do not match');
 expect(()=>confirmedImageReplacements(before,[{...after[0],id:'another'}])).toThrow('do not match');
});
test('no changed references preserve original object identity and optional fields',()=>{
 const generated={mainImage:'https://shop.test/a.jpg',items:[{image:'https://shop.test/a.jpg'}]};
 const saved={};
 expect(replaceGeneratedOutfitImages(generated,new Map())).toBe(generated);
 expect(replaceSavedOutfitImages(saved,new Map())).toBe(saved);
});
