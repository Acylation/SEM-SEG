<script lang="ts">
  import { Spinner } from 'flowbite-svelte';
  let loading = false; // Tracks loading state
  let pythonOutput = ''; // Stores Python script output
  let filePath = ''; // Stores resolved file path
  let errorMessage = ''; // Tracks error messages

  async function runPython() {
    loading = true; // Show loader
    errorMessage = ''; // Clear previous errors
    pythonOutput = ''; // Clear previous output
    try {
      const resolvedPath = await resolveFilePath('./resources/backend/test.txt');
      const result = await window.api.runPython(['gen', 'shin', resolvedPath]);
      pythonOutput = result; // Store the Python output
    } catch (error) {
      errorMessage = `Error running Python: ${error.message || error}`;
    } finally {
      loading = false; // Hide loader
    }
  }

  async function resolveFilePath(path: string): Promise<string> {
    errorMessage = ''; // Clear previous errors
    try {
      const result = await window.api.resolveFilePath(path);
      filePath = result; // Store the resolved path
      return result;
    } catch (error) {
      errorMessage = `Error resolving file path: ${error.message || error}`;
      throw error;
    }
  }
</script>

<div class="space-y-4 p-4">
  <!-- Spinner -->
  {#if loading}
    <div class="flex items-center justify-center">
      <Spinner color="blue" size="xl" />
    </div>
  {/if}

  <!-- Buttons -->
  <button
    on:click={runPython}
    class="rounded bg-blue-600 px-4 py-2 font-semibold text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
    disabled={loading}
  >
    Run Python
  </button>
  <!-- <button
    on:click={() => resolveFilePath('./resources/backend/test.txt')}
    class="rounded bg-gray-600 px-4 py-2 font-semibold text-white hover:bg-gray-700 disabled:cursor-not-allowed disabled:opacity-50"
    disabled={loading}
  >
    Resolve File Path
  </button> -->

  <!-- Output Section -->
  {#if pythonOutput}
    <div class="rounded border-l-4 border-blue-600 bg-blue-50 p-4">
      <h3 class="font-bold text-blue-600">Python Output</h3>
      <pre class="text-sm">{pythonOutput}</pre>
    </div>
  {/if}

  <!-- File Path Section -->
  {#if filePath && !loading}
    <div class="rounded border-l-4 border-gray-600 bg-gray-50 p-4">
      <h3 class="font-bold text-gray-600">Resolved File Path</h3>
      <pre class="text-sm">{filePath}</pre>
    </div>
  {/if}

  <!-- Error Message -->
  {#if errorMessage}
    <div class="rounded border-l-4 border-red-600 bg-red-50 p-4">
      <h3 class="font-bold text-red-600">Error</h3>
      <p class="text-sm">{errorMessage}</p>
    </div>
  {/if}
</div>
