type InterestStoreState = {
  count: number;
};

declare global {
  var __fashionImagineInterestStore: InterestStoreState | undefined;
}

function getStore() {
  globalThis.__fashionImagineInterestStore ||= { count: 0 };
  return globalThis.__fashionImagineInterestStore;
}

export function recordInterestSignal() {
  const store = getStore();
  store.count += 1;
  return store.count;
}

export function getInterestSignalCount() {
  return getStore().count;
}
