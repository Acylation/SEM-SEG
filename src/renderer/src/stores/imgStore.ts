import { writable } from 'svelte/store';
import type { Writable } from 'svelte/store';
import type { ImageStoreState } from './types';
import { Steps } from './types';

// Initial state
const initialState: ImageStoreState = {
  currentImage: null,
  processingStep: Steps.Unknown,
  settings: {},
};

// Create a typed writable store
const createImageStore = () => {
  const { subscribe, set, update }: Writable<ImageStoreState> = writable(initialState);

  return {
    subscribe,
    setImage: (image: string) => update((state) => ({ ...state, currentImage: image })),
    updateProcessingStep: (step: Steps) =>
      update((state) => ({
        ...state,
        processingStep: step,
      })),
    clearProcessingStep: () => update((state) => ({ ...state, processingStep: Steps.Unknown })),
    reset: () => set(initialState),
  };
};

export const imageStore = createImageStore();
