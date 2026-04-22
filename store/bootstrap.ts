let initialized = false;

export function bootstrapStores() {
  if (initialized) {
    return;
  }

  initialized = true;
}
