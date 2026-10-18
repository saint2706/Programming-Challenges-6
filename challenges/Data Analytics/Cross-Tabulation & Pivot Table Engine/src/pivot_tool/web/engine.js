"use strict";
// The pivot engine in the browser: the same Rust crate the Python CLI loads through ctypes,
// compiled to WebAssembly. One calling convention (pt_alloc / pt_call / pt_free, JSON in and out),
// so this wrapper is the whole binding. It also loads under Node, where a test runs it against the
// native engine and expects identical answers.

const PivotEngine = (() => {
  const enc = new TextEncoder();
  const dec = new TextDecoder();

  class Engine {
    constructor(instance) {
      this.x = instance.exports;
    }

    static async load(wasmBytes) {
      const { instance } = await WebAssembly.instantiate(wasmBytes, {});
      return new Engine(instance);
    }

    _put(bytes) {
      const ptr = this.x.pt_alloc(bytes.length);
      new Uint8Array(this.x.memory.buffer, ptr, bytes.length).set(bytes);
      return ptr;
    }

    // op: string; payload: JSON-able object; data: Uint8Array (the CSV for load_csv). Returns the
    // "ok" value or throws Error with the engine's message.
    call(op, payload, data) {
      const x = this.x;
      const name = enc.encode(op);
      const body = enc.encode(JSON.stringify(payload === undefined ? {} : payload));
      const bytes = data || new Uint8Array(0);
      const namePtr = this._put(name);
      const bodyPtr = this._put(body);
      const dataPtr = this._put(bytes);
      const lenPtr = x.pt_alloc(4);
      const outPtr = x.pt_call(namePtr, name.length, bodyPtr, body.length, dataPtr, bytes.length, lenPtr);
      // memory may have grown during the call: take fresh views
      const outLen = new DataView(x.memory.buffer).getUint32(lenPtr, true);
      const text = dec.decode(new Uint8Array(x.memory.buffer, outPtr, outLen).slice());
      x.pt_free(outPtr, outLen);
      x.pt_free(namePtr, name.length);
      x.pt_free(bodyPtr, body.length);
      x.pt_free(dataPtr, bytes.length);
      x.pt_free(lenPtr, 4);
      const answer = JSON.parse(text);
      if ("error" in answer) throw new Error(answer.error);
      return answer.ok;
    }
  }

  async function gunzipBase64(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
    return new Uint8Array(await new Response(stream).arrayBuffer());
  }

  return { Engine, gunzipBase64 };
})();

if (typeof module !== "undefined" && module.exports) {
  module.exports = PivotEngine;
}
