<script lang="ts">
  import { Dropzone } from 'flowbite-svelte';
  import { Upload } from 'lucide-svelte';
  import Table from './Table.svelte';
  import { Button, Modal } from 'flowbite-svelte';

  interface FileInfo {
    name: string;
    path: string; // Optional if not used
  }

  let value: FileInfo[] = [];
  let files: File[] = [];
  let showModal = false;
  let imgFile: string | undefined;

  const processFiles = (fileList: FileList | File[]) => {
    [...fileList].forEach((file) => {
      if (file.type.indexOf('image/') === 0) {
        value.push({ name: file.name, path: '' });
        value = value;
        files.push(file);
        file = file;
      }
    });
  };

  const dropHandle = (event: DragEvent) => {
    event.preventDefault();
    const fileList = [...(event.dataTransfer?.items || [])]
      .filter((item) => item.kind === 'file')
      .map((item) => item.getAsFile());
    processFiles(fileList);
  };

  const handleChange = (event: Event) => {
    const target = event.target as HTMLInputElement;
    if (target.files) {
      processFiles(target.files);
    }
  };

  // TODO: avoid duplicate
  // const updateValue = () => {
  //   value = [...new Map(value.map((v) => [v.name, v])).values()];
  // };

  const clearValue = () => {
    value = [];
    files = [];
  };

  const popImage = (id: number) => {
    const file = files[id];
    const reader = new FileReader();
    reader.onload = () => {
      imgFile = reader.result as string;
      showModal = true;
    };
    reader.readAsDataURL(file);
  };
</script>

<Modal title="Image Preview" bind:open={showModal} autoclose>
  {#if imgFile}
    <img width="600 px" src={imgFile} alt={''} />
  {/if}
  <svelte:fragment slot="footer">
    <Button on:click={() => (showModal = false)}>OK</Button>
  </svelte:fragment>
</Modal>

<div>
  <Dropzone
    id="dropzone"
    accept="image/*"
    multiple={true}
    on:drop={dropHandle}
    on:dragover={(event) => event.preventDefault()}
    on:change={handleChange}
  >
    <Upload class="mb-3 h-10 w-10 text-gray-400" fill="none" stroke="currentColor" />
    <p class="mb-2 text-sm text-gray-500 dark:text-gray-400">
      <span class="font-semibold">Click to upload</span> or drag and drop
    </p>
    <p class="text-xs text-gray-500 dark:text-gray-400">SVG, PNG, JPG or GIF (MAX. 800x400px)</p>
  </Dropzone>
  {#if value.length > 0}
    <Table tableData={value} on:clear={clearValue} on:preview={({ detail: id }) => popImage(id)} />
    <Button
      class="mt-5 rounded bg-blue-600 px-4 py-2 font-semibold text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
    >
      Next
    </Button>
  {/if}
</div>
