<script lang="ts">
  async function runPython() {
    try {
      const result = await window.api.runPython([
        'gen',
        'shin',
        await resolveFilePath('./resources/backend/test.txt'), // path starts from the project root, or app.asar
      ]);
      console.log('Python output:', result);
    } catch (error) {
      console.error('Error running Python:', error);
    }
  }

  async function resolveFilePath(path: string) {
    try {
      const result = await window.api.resolveFilePath(path); // starts from app.asar
      console.log('Actual file path:', result);
      return result;
    } catch (error) {
      console.error('Error resolving file path:', error);
      return error;
    }
  }
</script>

<button on:click={runPython}>Run Python</button>
<button on:click={() => resolveFilePath('./resources/backend/test.txt')}>Resolve File Path</button>
