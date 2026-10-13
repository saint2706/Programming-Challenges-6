// htmx 2 does not swap 4xx/5xx responses by default. Our error responses are the same
// fragment with a flash message (409 for refused actions, 502 for failed sends), so swap them
// and let the flash explain what happened.
document.body.addEventListener("htmx:beforeSwap", (event) => {
  const xhr = event.detail.xhr;
  if (xhr.status >= 400 && (xhr.getResponseHeader("content-type") || "").startsWith("text/html")) {
    event.detail.shouldSwap = true;
    event.detail.isError = false;
  }
});
