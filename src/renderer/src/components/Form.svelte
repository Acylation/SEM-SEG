<script>
  import { Label, Select } from 'flowbite-svelte';
  let selected;

  let countries = [
    { value: 'us', name: 'United States' },
    { value: 'ca', name: 'Canada' },
    { value: 'fr', name: 'France' },
  ];

  import { Input, Helper } from 'flowbite-svelte';
</script>

<Label>
  Select an option
  <Select class="mt-2" items={countries} bind:value={selected} />
</Label>

<Label class="mb-2 block">Your email</Label>
<Input label="Email" id="email" name="email" required placeholder="name@flowbite.com" />
<Helper class="mt-2 text-sm">
  We’ll never share your details. Read our <a
    href="/"
    class="font-medium text-primary-600 hover:underline dark:text-primary-500"
  >
    Privacy Policy
  </a>
  .
</Helper>
