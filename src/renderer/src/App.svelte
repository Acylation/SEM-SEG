<script>
  import Toolbar from './components/Toolbar.svelte';
  import Layout from './components/Layout.svelte';
  import Workspace from './components/Workspace.svelte';
  import Sidebar from './components/Sidebar.svelte';
</script>

<Layout>
  <Toolbar slot="toolbar" />
  <Sidebar slot="sidebar"></Sidebar>
  <Workspace slot="workspace"></Workspace>
</Layout>
