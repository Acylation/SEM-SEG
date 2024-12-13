export enum Steps {
  Unknown,
  Loaded,
  Preview,
  SAMPredict,
  Hough,
  Save,
}

export interface ImageStoreState {
  currentImage: string | null; // Base64 string or URL
  processingStep: Steps; // Array of processing steps
  settings: Record<string, unknown>; // Additional app settings
}
