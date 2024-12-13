<script>
  import { Stage, Layer, Rect } from 'svelte-konva';
</script>

<Stage config={{ width: 200, height: 200 }}>
  <Layer>
    <Rect config={{ x: 100, y: 100, width: 200, height: 100, fill: 'blue' }} />
  </Layer>
</Stage>
