<script>
  let { html, width } = $props();
</script>

<div class="preview-frame">
  <!--
    Sandboxed, srcdoc-rendered iframe: this is the correct technique for
    previewing generated HTML because it fully isolates the email markup's
    own styles/DOM from the builder app (a <div> render would leak the
    app's CSS into the "email"). `sandbox=""` (no allow-scripts) is
    maximally restrictive — the preview never needs to execute anything.
  -->
  <iframe title="Email preview" sandbox="" srcdoc={html} style="width: {width}px;"></iframe>
</div>

<style>
  .preview-frame {
    flex: 1;
    overflow: auto;
    display: flex;
    justify-content: center;
    padding: 16px 0;
    background: #e5e7eb;
    border-radius: 6px;
  }
  iframe {
    max-width: 100%;
    height: 100%;
    min-height: 500px;
    border: 1px solid #d1d5db;
    background: white;
    transition: width 0.2s ease;
  }
</style>
