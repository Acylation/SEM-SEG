<script lang="ts">
  import {
    Table,
    TableBody,
    TableBodyCell,
    TableBodyRow,
    TableHead,
    TableHeadCell,
    Checkbox,
  } from 'flowbite-svelte';
  import { createEventDispatcher } from 'svelte';

  export let tableData = [];

  $: headers = tableData.length > 0 ? Object.keys(tableData[0]) : [];
  $: checked = Array(tableData.length).fill(true);
  $: headCheck = checked.every((v) => v);
  $: indeterminate = !headCheck && checked.some((v) => v);

  const handleHeadCheck = (event: Event) => {
    const target = event.target as HTMLInputElement;
    headCheck = target.checked;
    checked = checked.fill(target.checked);
  };

  const handleRowCheck = (event: Event, id: number) => {
    const target = event.target as HTMLInputElement;
    checked[id] = target.checked;
  };

  const dispatch = createEventDispatcher<{
    clear: void;
    preview: number;
    // remove: number;
  }>();
</script>

<Table hoverable={true} class="mt-5">
  <TableHead>
    <TableHeadCell class="!p-4">
      <Checkbox checked={headCheck} bind:indeterminate on:change={handleHeadCheck} />
    </TableHeadCell>
    {#each headers as header}
      <TableHeadCell>
        {header.charAt(0).toUpperCase() + header.slice(1).replace(/_/g, ' ')}
      </TableHeadCell>
    {/each}
    <TableHeadCell>
      <button
        class="font-medium text-primary-600 hover:underline dark:text-primary-500"
        on:click={() => dispatch('clear')}
        on:keypress={(event) => {
          if (event.key !== 'Enter') return;
          dispatch('clear');
        }}
      >
        CLEAR
      </button>
    </TableHeadCell>
  </TableHead>
  <TableBody tableBodyClass="divide-y">
    {#each tableData as item, id}
      <TableBodyRow>
        <TableBodyCell class="!p-4">
          <Checkbox checked={checked[id]} on:change={(e) => handleRowCheck(e, id)} />
        </TableBodyCell>
        {#each headers as key}
          <TableBodyCell>{item[key]}</TableBodyCell>
        {/each}
        <TableBodyCell>
          <button
            class="font-medium text-primary-600 hover:underline dark:text-primary-500"
            on:click={() => dispatch('preview', id)}
          >
            Preview
          </button>
        </TableBodyCell>
      </TableBodyRow>
    {/each}
  </TableBody>
</Table>
