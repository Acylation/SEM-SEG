<script lang="ts">
  const ipcHandle = (): void => window.electron.ipcRenderer.send('ping');
  import Button from './Button.svelte';
  import FileUpload from './components/FileUpload.svelte';
</script>

<div>
  <button
    class="rounded bg-blue-600 px-4 py-2 font-semibold text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
  >
    <a href="https://electron-vite.org/" target="_blank" rel="noreferrer">Documentation</a>
  </button>
  <button
    class="rounded bg-blue-600 px-4 py-2 font-semibold text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
  >
    <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions a11y-missing-attribute-->
    <a target="_blank" rel="noreferrer" on:click={ipcHandle}>Send IPC</a>
  </button>
</div>
<FileUpload></FileUpload>
<Button />
