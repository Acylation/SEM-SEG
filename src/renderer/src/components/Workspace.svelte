<script lang="ts">
  import { imageStore } from '../stores/imgStore';
  import { Steps } from '../stores/types';
  let currentImage: string | null;
  let processingStep: Steps;

  import FileUpload from './FileUpload.svelte';
  import Form from './Form.svelte';

  // Subscribe to the store and infer types automatically
  imageStore.subscribe((state) => {
    currentImage = state.currentImage;
    processingStep = state.processingStep;
  });
</script>

<!-- <FileUpload /> -->
<Form />
