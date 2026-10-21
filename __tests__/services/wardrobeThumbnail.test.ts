import {createNativeWardrobeThumbnail,flushNativeWardrobeThumbnails} from '../../src/services/wardrobeThumbnail';
import {Image} from '../../src/components/MediaImage';
import {ImageManipulator} from 'expo-image-manipulator';
import {stageNativeWardrobeImage} from '../../src/services/pendingWardrobeMedia';
import * as FileSystem from 'expo-file-system/legacy';
import {authEvents} from '../../store/authEvents';
jest.mock('../../src/components/MediaImage',()=>({Image:{getSize:jest.fn()}}));
jest.mock('expo-image-manipulator',()=>({ImageManipulator:{manipulate:jest.fn()},SaveFormat:{PNG:'png',JPEG:'jpeg'}}));
jest.mock('expo-file-system/legacy',()=>({documentDirectory:'file:///Documents/',cacheDirectory:'file:///cache/',deleteAsync:jest.fn()}));
jest.mock('../../src/services/pendingWardrobeMedia',()=>({...jest.requireActual('../../src/services/pendingWardrobeMedia'),stageNativeWardrobeImage:jest.fn()}));
jest.mock('../../lib/supabase',()=>({supabase:{}}));
const owner='11111111-1111-4111-8111-111111111111',id='aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',thumb='bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb';
const source=`file:///Documents/aiwardrobe-pending-media/${owner}/${id}.png`;
const temporary='file:///cache/ImageManipulator/output.png';
const resize=jest.fn(),save=jest.fn(),render=jest.fn(),releaseImage=jest.fn(),releaseContext=jest.fn();
beforeEach(async()=>{
 await flushNativeWardrobeThumbnails();
 jest.resetAllMocks();authEvents.emitLogin(owner);
 (Image.getSize as jest.Mock).mockResolvedValue({width:1125,height:1500});
 save.mockResolvedValue({uri:temporary,width:384,height:512});render.mockResolvedValue({saveAsync:save,release:releaseImage});
 (ImageManipulator.manipulate as jest.Mock).mockReturnValue({resize,renderAsync:render,release:releaseContext});
 (stageNativeWardrobeImage as jest.Mock).mockResolvedValue('file:///Documents/thumbnail.png');
 (FileSystem.deleteAsync as jest.Mock).mockResolvedValue(undefined);
});
test('portrait preview preserves PNG encoding and stages output before cache cleanup',async()=>{
 expect(await createNativeWardrobeThumbnail(owner,thumb,source)).toBe('file:///Documents/thumbnail.png');
 expect(resize).toHaveBeenCalledWith({height:512});expect(save).toHaveBeenCalledWith({format:'png',compress:1});
 expect(stageNativeWardrobeImage).toHaveBeenCalledWith(owner,thumb,temporary,'image/png');
 expect((stageNativeWardrobeImage as jest.Mock).mock.invocationCallOrder[0]).toBeLessThan((FileSystem.deleteAsync as jest.Mock).mock.invocationCallOrder[0]);
 expect(FileSystem.deleteAsync).toHaveBeenCalledWith(temporary,{idempotent:true});expect(releaseImage).toHaveBeenCalledTimes(1);expect(releaseContext).toHaveBeenCalledTimes(1);
});
it.each([[1500,1125,{width:512}],[100,80,undefined]])('resize bounds for %s by %s never upscale',async(width,height,expected)=>{
 (Image.getSize as jest.Mock).mockResolvedValue({width,height});const scale=Math.min(1,512/Math.max(width,height));save.mockResolvedValue({uri:temporary,width:Math.round(width*scale),height:Math.round(height*scale)});
 await createNativeWardrobeThumbnail(owner,thumb,source);if(expected)expect(resize).toHaveBeenCalledWith(expected);else expect(resize).not.toHaveBeenCalled();
});
it.each([[0,1],[1.5,20],[10000,10000]])('invalid or excessive dimensions %s by %s reject before manipulation',async(width,height)=>{
 (Image.getSize as jest.Mock).mockResolvedValue({width,height});await expect(createNativeWardrobeThumbnail(owner,thumb,source)).rejects.toMatchObject({code:'invalid_size'});expect(ImageManipulator.manipulate).not.toHaveBeenCalled();
});
test('foreign and unmanaged source files cannot be manipulated',async()=>{
 await expect(createNativeWardrobeThumbnail(owner,thumb,source.replace(owner,'22222222-2222-4222-8222-222222222222'))).rejects.toMatchObject({code:'access_denied'});
 await expect(createNativeWardrobeThumbnail(owner,thumb,'file:///picker/photo.png')).rejects.toMatchObject({code:'access_denied'});expect(Image.getSize).not.toHaveBeenCalled();
});
test('account change during encode blocks durable publication and removes only cache output',async()=>{
 save.mockImplementationOnce(async()=>{authEvents.emitLogout();return {uri:temporary,width:384,height:512};});
 await expect(createNativeWardrobeThumbnail(owner,thumb,source)).rejects.toMatchObject({code:'account_changed'});expect(stageNativeWardrobeImage).not.toHaveBeenCalled();expect(FileSystem.deleteAsync).toHaveBeenCalledWith(temporary,{idempotent:true});
});
test('failed staging removes temporary output and releases image resources',async()=>{
 (stageNativeWardrobeImage as jest.Mock).mockRejectedValueOnce(new Error('disk full'));
 await expect(createNativeWardrobeThumbnail(owner,thumb,source)).rejects.toThrow('disk full');expect(FileSystem.deleteAsync).toHaveBeenCalledWith(temporary,{idempotent:true});expect(releaseImage).toHaveBeenCalledTimes(1);expect(releaseContext).toHaveBeenCalledTimes(1);
});
test('a malformed codec result never deletes the durable original',async()=>{
 save.mockResolvedValueOnce({uri:source,width:384,height:512});await expect(createNativeWardrobeThumbnail(owner,thumb,source)).rejects.toMatchObject({code:'invalid_source'});expect(FileSystem.deleteAsync).not.toHaveBeenCalled();expect(stageNativeWardrobeImage).not.toHaveBeenCalled();
});

it.each([
 'file:///cache/',
 'file:///cache/ImageManipulator/',
 'file:///cache/../Documents/original.png',
 'file:///cache/ImageManipulator/%2e%2e%2foutside.png',
 'file:///cache/ImageManipulator/%2Foutside.png',
 'file:///cache/ImageManipulator/%00.png',
 'file:///cache/ImageManipulator/%invalid.png',
])('unsafe codec output %s is never staged or deleted',async uri=>{
 save.mockResolvedValueOnce({uri,width:384,height:512});
 await expect(createNativeWardrobeThumbnail(owner,thumb,source)).rejects.toMatchObject({code:'invalid_source'});
 expect(stageNativeWardrobeImage).not.toHaveBeenCalled();
 expect(FileSystem.deleteAsync).not.toHaveBeenCalled();
 expect(releaseImage).toHaveBeenCalledTimes(1);expect(releaseContext).toHaveBeenCalledTimes(1);
});


test('distorted codec dimensions cannot enter the upload queue',async()=>{
 save.mockResolvedValueOnce({uri:temporary,width:512,height:512});
 await expect(createNativeWardrobeThumbnail(owner,thumb,source)).rejects.toMatchObject({code:'invalid_source'});
 expect(stageNativeWardrobeImage).not.toHaveBeenCalled();expect(FileSystem.deleteAsync).toHaveBeenCalledWith(temporary,{idempotent:true});
});


test('overlapping saves never run two native decoders at once',async()=>{
 let finish!:(value:unknown)=>void;let started!:()=>void;
 const active=new Promise<void>(resolve=>{started=resolve;});
 render.mockImplementationOnce(()=>{started();return new Promise(resolve=>{finish=resolve;});});
 const first=createNativeWardrobeThumbnail(owner,thumb,source),second=createNativeWardrobeThumbnail(owner,thumb,source);
 await active;expect(Image.getSize).toHaveBeenCalledTimes(1);expect(ImageManipulator.manipulate).toHaveBeenCalledTimes(1);
 finish({saveAsync:save,release:releaseImage});await Promise.all([first,second]);
 expect(ImageManipulator.manipulate).toHaveBeenCalledTimes(2);
 expect(releaseContext.mock.invocationCallOrder[0]).toBeLessThan((ImageManipulator.manipulate as jest.Mock).mock.invocationCallOrder[1]);
});
test('queued old-session work is rejected after same-account relogin and does not poison later saves',async()=>{
 let finish!:(value:unknown)=>void;let started!:()=>void;
 const active=new Promise<void>(resolve=>{started=resolve;});
 render.mockImplementationOnce(()=>{started();return new Promise(resolve=>{finish=resolve;});});
 const first=createNativeWardrobeThumbnail(owner,thumb,source),second=createNativeWardrobeThumbnail(owner,thumb,source);
 const failed=[expect(first).rejects.toMatchObject({code:'account_changed'}),expect(second).rejects.toMatchObject({code:'account_changed'})];
 await active;authEvents.emitLogout();authEvents.emitLogin(owner);finish({saveAsync:save,release:releaseImage});await Promise.all(failed);
 expect(Image.getSize).toHaveBeenCalledTimes(1);
 await expect(createNativeWardrobeThumbnail(owner,thumb,source)).resolves.toBe('file:///Documents/thumbnail.png');
});


test('JPEG originals use an opaque JPEG preview with a matching MIME type',async()=>{
 const jpegSource=source.replace(/\.png$/,'.jpg');
 const jpegTemporary='file:///cache/ImageManipulator/output.jpg';
 save.mockResolvedValueOnce({uri:jpegTemporary,width:384,height:512});
 await createNativeWardrobeThumbnail(owner,thumb,jpegSource);
 expect(save).toHaveBeenCalledWith(expect.objectContaining({format:'jpeg'}));
 expect(save.mock.calls[0][0].compress).toBeGreaterThanOrEqual(0.9);
 expect(stageNativeWardrobeImage).toHaveBeenCalledWith(owner,thumb,jpegTemporary,'image/jpeg');
 expect(FileSystem.deleteAsync).toHaveBeenCalledWith(jpegTemporary,{idempotent:true});
 expect(FileSystem.deleteAsync).not.toHaveBeenCalledWith(jpegSource,expect.anything());
});

it.each(['png','webp','heic','heif'])('potentially transparent %s originals retain PNG output',async extension=>{
 await createNativeWardrobeThumbnail(owner,thumb,source.replace(/\.png$/,'.'+extension));
 expect(save).toHaveBeenCalledWith({format:'png',compress:1});
 expect(stageNativeWardrobeImage).toHaveBeenCalledWith(owner,thumb,temporary,'image/png');
});
