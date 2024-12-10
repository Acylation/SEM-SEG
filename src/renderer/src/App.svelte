<script lang="ts">
  const ipcHandle = (): void => window.electron.ipcRenderer.send('ping');
  import Button from './Button.svelte';
  import FileUpload from './components/FileUpload.svelte';
</script>

<div class="actions">
  <div class="action">
    <a href="https://electron-vite.org/" target="_blank" rel="noreferrer">Documentation</a>
  </div>
  <div class="action">
    <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions a11y-missing-attribute-->
    <a target="_blank" rel="noreferrer" on:click={ipcHandle}>Send IPC</a>
  </div>
</div>
<FileUpload></FileUpload>
<Button />
