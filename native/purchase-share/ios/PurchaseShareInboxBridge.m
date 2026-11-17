#import <React/RCTBridgeModule.h>
@interface RCT_EXTERN_MODULE(PurchaseShareInbox, NSObject)
RCT_EXTERN_METHOD(setOwner:(NSString *)owner)
RCT_EXTERN_METHOD(consume:(NSString *)owner resolver:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject)
@end
